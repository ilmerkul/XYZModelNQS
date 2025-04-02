from typing import Dict

import jax
import netket as nk
import netket.nn
from flax import linen as nn
from flax import struct
from jax import lax
from jax import nn as jnn
from jax import numpy as jnp
from netket.utils.group import PermutationGroup


@struct.dataclass
class TransformerConfig:
    # Number of encoder layer pairs
    layers: int = 4

    # Number of units in dense layer
    mlp_dim_scale: int = 4

    # Tokens length
    length: int = 10

    # Number of embdedding dim
    features: int = 32

    # Number of heads in multihead attention
    num_heads: int = 1

    # Bias
    use_bias: bool = False

    # Droput rate
    dropout_rate: float = 0.1

    # Dropout or not
    training: bool = False

    # Random seed
    seed: int = 0

    n_state: int = 2

    dtype: jnp.dtype = jnp.float64

    automorphisms: PermutationGroup = nk.graph.Chain(length=10,
                                                     pbc=True).automorphisms()

    hilbert: nk.hilbert.Spin = nk.hilbert.Spin(N=10, s=1 / 2)

    symm: bool = True

    embed_concat: bool = False

    pos_embed: str = "rel"

    default_kernel_init: jnn.initializers.Initializer = jnn.initializers.normal(
        1e-1, dtype=jnp.float64)
    default_bias_init: jnn.initializers.Initializer = jnn.initializers.normal(
        1e-4, dtype=jnp.float64)


class MultiheadAttention(nn.Module):
    config: TransformerConfig
    decode: bool = False

    @nn.compact
    def __call__(self, kv, q, mask):
        # Initialize config
        cfg = self.config
        # Initialize key, query, value through Dense layer
        query = nn.Dense(features=cfg.features, use_bias=cfg.use_bias,
                         name='query', dtype=cfg.dtype,
                         kernel_init=cfg.default_kernel_init,
                         bias_init=cfg.default_bias_init)(q)
        key = nn.Dense(features=cfg.features, use_bias=cfg.use_bias,
                       name='key', dtype=cfg.dtype,
                       kernel_init=cfg.default_kernel_init,
                       bias_init=cfg.default_bias_init)(kv)
        value = nn.Dense(features=cfg.features, use_bias=cfg.use_bias,
                         name='value', dtype=cfg.dtype,
                         kernel_init=cfg.default_kernel_init,
                         bias_init=cfg.default_bias_init)(kv)

        batch_size = query.shape[0]

        # Layer norm
        # query = nn.LayerNorm()(query)
        # key = nn.LayerNorm()(key)
        # value = nn.LayerNorm()(value)

        head_dim = cfg.features // cfg.num_heads

        # Split head
        query = query.reshape(batch_size, -1, cfg.num_heads, head_dim)
        key = key.reshape(batch_size, -1, cfg.num_heads, head_dim)
        value = value.reshape(batch_size, -1, cfg.num_heads, head_dim)
        # Scaled dot-product attention
        logits = self.scaled_dot_product_attention(key, query, value, mask,
                                                   cfg.dtype,
                                                   cfg.pos_embed)

        # Concat
        logits = logits.reshape(batch_size, -1, cfg.features)

        # Linear
        logits = nn.Dense(features=cfg.features,
                          use_bias=cfg.use_bias,
                          name='attention_weights',
                          dtype=cfg.dtype,
                          kernel_init=cfg.default_kernel_init,
                          bias_init=cfg.default_bias_init
                          )(logits)
        logits = nn.Dropout(rate=cfg.dropout_rate,
                            deterministic=not cfg.training)(logits)

        return logits

    def scaled_dot_product_attention(self, key, query, value, mask, dtype,
                                     pos_embed):
        """ Matmul
            query: [batch, q_length, num_heads, qk_depth_per_head]
            key: [batch, kv_length, num_heads, qk_depth_per_head]
            -> qk: [batch, num_heads, q_length, kv_length]
        """
        length_q = query.shape[1]
        length_k = key.shape[1]
        hid_head = query.shape[3]

        if pos_embed == "rope":
            def get_rope(x):
                base = 1000
                d = hid_head
                length = x.shape[1]
                theta = 1. / (base ** (jnp.arange(0, d, 2) / d))
                seq_idx = jnp.arange(length)
                idx_theta = jnp.einsum('n,d->nd', seq_idx, theta)
                idx_theta2 = jnp.concatenate([idx_theta, idx_theta], axis=1)
                cos_cached = jnp.cos(idx_theta2)[:, None, :]
                sin_cached = jnp.sin(idx_theta2)[:, None, :]

                x_rope, x_pass = x[..., :d], x[..., d:]

                d_2 = d // 2
                neg_half_x = jnp.concatenate([-x_rope[:, :, :, d_2:],
                                              x_rope[:, :, :, :d_2]],
                                             axis=-1)

                x_rope = (x_rope * cos_cached[:x.shape[0]]) + (
                        neg_half_x * sin_cached[:x.shape[0]])

                return jnp.concatenate((x_rope, x_pass), axis=-1)

            length_q_2 = length_q // 2
            query = jnp.concatenate([get_rope(query[:, :length_q_2, :, :]),
                                     get_rope(
                                         query[:, length_q_2:, :, :][:, ::-1,
                                         ...])[:, ::-1, ...]],
                                    axis=1)
            key = jnp.concatenate([get_rope(key[:, :length_q_2, :, :]),
                                   get_rope(key[:, length_q_2:, :, :][:, ::-1,
                                            ...])[:, ::-1, ...]],
                                  axis=1)

        attention_weights = jnp.einsum("bqhd,bkhd->bhqk", query, key)

        if pos_embed == "rel":
            range_vec_q = jnp.arange(length_q)
            range_vec_k = jnp.arange(length_k)
            distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
            distance_mat += length_q

            embeddings = nn.Embed(num_embeddings=2 * length_q + 1,
                                  features=hid_head,
                                  name='rel_pos', dtype=dtype)(
                distance_mat.flatten())

            embeddings = nn.Dense(features=hid_head, use_bias=False,
                                  name='rel_pos_dense', dtype=dtype,
                                  kernel_init=self.config.default_kernel_init,
                                  bias_init=self.config.default_bias_init)(
                embeddings)

            # [q_length, kv_length, qk_depth_per_head]
            embeddings = embeddings.reshape(length_q, length_k, hid_head)

            attention_weights_rel_pos = jnp.einsum("bqhd,qkd->bhqk", query,
                                                   embeddings)

            attention_weights += attention_weights_rel_pos

        # Scale
        d_k = query.shape[-1]
        # attention_weights = (attention_weights + jnp.permute_dims(
        #    attention_weights, (0, 1, 3, 2))) / 2.
        attention_weights = attention_weights / jnp.sqrt(d_k)
        # Mask
        num_heads = attention_weights.shape[1]
        attention_weights = lax.select(
            jnp.repeat(mask, num_heads, axis=1) > 0,
            attention_weights,
            jnp.full(attention_weights.shape, -jnp.inf, dtype=dtype)
        )

        # Softmax
        attention_weights = nn.softmax(attention_weights).astype(dtype)
        """ Matmul
            qk: [batch, num_heads, q_length, kv_length]
            value: [batch, kv_length, num_heads, v_depth_per_head]
            -> Return: [batch, length, num_heads, v_depth_per_head]
        """

        return jnp.einsum('bhqk,bkhd->bqhd', attention_weights, value)

    @classmethod
    def attention_mask(cls, input_tokens):
        """Mask-making helper for attention weights (mask for padding)
        Args:
            input_tokens: [batch_size, tokens_length]
        return:
            mask: [batch_size, num_heads=1, query_length, key_value_length]
        """
        mask = jnp.multiply(input_tokens[:, :, None],
                            input_tokens[:, None, :])
        mask = mask[:, None, ...]
        mask = lax.select(
            mask > 0,
            jnp.full(mask.shape, 1),
            jnp.full(mask.shape, 0)
        )
        # mask -= jnp.eye(input_tokens.shape[1], input_tokens.shape[1])
        return mask

    @classmethod
    def causal_mask(cls, input_tokens):
        mask = cls.attention_mask(input_tokens)
        mask = jnp.full(mask.shape, -jnp.inf)
        mask = jnp.triu(mask, k=1)
        return mask


class MLP(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x):
        # Initialize config
        cfg = self.config

        x = nn.Dense(features=cfg.mlp_dim_scale * cfg.features,
                     use_bias=cfg.use_bias, dtype=cfg.dtype,
                     kernel_init=cfg.default_kernel_init,
                     bias_init=cfg.default_bias_init)(x)
        x = nn.Dropout(rate=cfg.dropout_rate,
                       deterministic=not cfg.training)(x)
        x = netket.nn.elu(x)
        x = nn.Dense(features=cfg.features, use_bias=cfg.use_bias,
                     dtype=cfg.dtype,
                     kernel_init=cfg.default_kernel_init,
                     bias_init=cfg.default_bias_init)(x)
        x = nn.Dropout(rate=cfg.dropout_rate,
                       deterministic=not cfg.training)(x)
        return x


class Encoder(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, input_embedding, encoder_mask):
        cfg = self.config

        # Encoder layer
        x = input_embedding
        for i in range(cfg.layers):
            # Multihead Attention
            y = nn.RMSNorm()(x)
            y = MultiheadAttention(config=cfg)(
                kv=y,
                q=y,
                mask=encoder_mask
            )
            # Add & Norm
            x = y + x
            y = nn.RMSNorm()(x)
            # MLP
            y = MLP(cfg)(y)
            # Add & Norm
            x = y + x

        return x


class Transformer(nn.Module):
    config: TransformerConfig
    states: Dict[int, int] = None

    @nn.compact
    def __call__(self, encoder_input_tokens):
        cfg = self.config

        encoder_input_tokens = (encoder_input_tokens > 0).astype(jnp.int32)

        padding_mask = MultiheadAttention.attention_mask(
            jnp.ones(encoder_input_tokens.shape))

        batch_size = encoder_input_tokens.shape[0]

        if cfg.embed_concat:
            features_2 = cfg.features // 2
        else:
            features_2 = cfg.features

        state_embedding = nn.Embed(num_embeddings=cfg.n_state,
                                   features=features_2,
                                   dtype=cfg.dtype)(encoder_input_tokens)
        input_embedding = state_embedding

        if cfg.pos_embed == "labs":
            if cfg.symm:
                pos_num_embeddings = (cfg.length + 1) // 2
                pos = jnp.concatenate([jnp.arange(pos_num_embeddings),
                                       jnp.arange(pos_num_embeddings - 1 -
                                                  (1 if cfg.length % 2 != 0
                                                   else 0), -1, -1)],
                                      axis=0)
            else:
                pos_num_embeddings = cfg.length
                pos = jnp.arange(cfg.length)
            pos = jnp.repeat(jnp.expand_dims(pos, axis=0), batch_size, axis=0)
            pos_embedding = nn.Embed(num_embeddings=pos_num_embeddings,
                                     features=features_2, dtype=cfg.dtype)(pos)

        elif cfg.pos_embed == "abs":
            base = 10000.0
            pos_embedding = jnp.zeros((cfg.length, features_2))
            position = jnp.arange(0, cfg.length)[:, None]
            div_term = jnp.exp((jnp.arange(0, features_2, 2) *
                                -(jnp.log(base) / features_2)))
            pos_embedding.at[:, 0::2].set(jnp.sin(position * div_term))
            pos_embedding.at[:, 1::2].set(jnp.cos(position * div_term))
        else:
            pos_embedding = 0

        if cfg.embed_concat:
            input_embedding = jnp.concatenate([pos_embedding, state_embedding],
                                              axis=-1)
        else:
            input_embedding += pos_embedding

        x = input_embedding
        enc = Encoder(cfg)
        x = enc(input_embedding=x,
                encoder_mask=padding_mask)

        # x = x[:, -1, :]
        x = nn.Dense(features=1, use_bias=cfg.use_bias, dtype=cfg.dtype,
                     kernel_init=cfg.default_kernel_init,
                     bias_init=cfg.default_bias_init)(x)

        x = nn.softmax(x, axis=-2)
        x = jnp.prod(x, axis=-2)

        return x.squeeze(-1)


if __name__ == "__main__":
    n = 11
    tr = Transformer(TransformerConfig(training=True, symm=True, length=n))

    key = jax.random.PRNGKey(42)
    input = jnp.ones(16 * n)
    input = input.at[jax.random.randint(key, (16, n), 0, 16 * n)].set(-1)
    input = input.reshape(16, n)
    init_rngs = {'params': key,
                 'dropout': key}
    params = tr.init(init_rngs, input)
    print(input)
    print(tr.apply(params, input, rngs={'dropout': key}))

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

    # Number of embedding dim
    features: int = 32

    # Number of heads in multihead attention
    num_heads: int = 2

    # Bias
    use_bias: bool = False

    # Dropout rate
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

    symm: bool = False

    embed_concat: bool = False

    pos_embed: str = "rope"

    default_kernel_init: jnn.initializers.Initializer = jnn.initializers.normal(
        1e-1, dtype=jnp.float64)
    default_bias_init: jnn.initializers.Initializer = jnn.initializers.normal(
        1e-4, dtype=jnp.float64)


class MultiheadAttention(nn.Module):
    config: TransformerConfig
    decode: bool = False

    def setup(self):
        cfg = self.config
        self.query_dense = nn.Dense(features=cfg.features,
                                    use_bias=cfg.use_bias,
                                    name='tr_query_dense', dtype=cfg.dtype,
                                    kernel_init=cfg.default_kernel_init,
                                    bias_init=cfg.default_bias_init)
        self.key_dense = nn.Dense(features=cfg.features, use_bias=cfg.use_bias,
                                  name='tr_key_dense', dtype=cfg.dtype,
                                  kernel_init=cfg.default_kernel_init,
                                  bias_init=cfg.default_bias_init)
        self.value_dense = nn.Dense(features=cfg.features,
                                    use_bias=cfg.use_bias,
                                    name='tr_value_dense', dtype=cfg.dtype,
                                    kernel_init=cfg.default_kernel_init,
                                    bias_init=cfg.default_bias_init)
        self.logits_dense = nn.Dense(features=cfg.features,
                                     use_bias=cfg.use_bias,
                                     name='tr_attention_weights',
                                     dtype=cfg.dtype,
                                     kernel_init=cfg.default_kernel_init,
                                     bias_init=cfg.default_bias_init
                                     )
        self.dropout = nn.Dropout(rate=cfg.dropout_rate,
                                  deterministic=not cfg.training)

    @nn.compact
    def __call__(self, kv: jax.Array,
                 q: jax.Array,
                 mask: jax.Array,
                 kv_cache: jax.Array):
        # Initialize config
        cfg = self.config
        # Initialize key, query, value through Dense layer
        query = self.query_dense(q)

        key = self.key_dense(kv[:, -1, :])
        value = self.value_dense(kv[:, -1, :])

        if kv_cache is None:
            kv_cache = jnp.concatenate([jnp.expand_dims(key, axis=-1),
                                        jnp.expand_dims(value, axis=-1)],
                                       axis=-1)
            kv_cache = jnp.expand_dims(kv_cache, axis=1)
        else:
            key = jnp.concatenate(
                [kv_cache[:, :, :, 0], jnp.expand_dims(key, axis=1)], axis=1)
            value = jnp.concatenate(
                [kv_cache[:, :, :, 1], jnp.expand_dims(value, axis=1)], axis=1)

            kv_cache = jnp.concatenate([jnp.expand_dims(key, axis=-1),
                                        jnp.expand_dims(value, axis=-1)],
                                       axis=-1)

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
        logits = self.logits_dense(logits)
        logits = self.dropout(logits)

        return logits, kv_cache

    def scaled_dot_product_attention(self, key: jax.Array, query: jax.Array,
                                     value: jax.Array, mask: jax.Array, dtype,
                                     pos_embed: str):
        """ Matmul
            query: [batch, q_length, num_heads, qk_depth_per_head]
            key: [batch, kv_length, num_heads, qk_depth_per_head]
            -> qk: [batch, num_heads, q_length, kv_length]
        """
        length_q = query.shape[1]
        length_k = key.shape[1]
        hid_head = query.shape[3]

        if pos_embed == "rope":
            query, key = self.rope_embedding(hid_head, length_q, query, key)
        attention_weights = jnp.einsum("bqhd,bkhd->bhqk", query, key)

        if pos_embed == "rel":
            attention_weights_rel_pos = self.rel_embedding(hid_head, length_q,
                                                           length_k, dtype,
                                                           query)
            attention_weights += attention_weights_rel_pos

        # Scale
        d_k = query.shape[-1]
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

    def rope_embedding(self, hid_head: int, length_q: int, query: jax.Array,
                       key: jax.Array) -> (jax.Array, jax.Array):
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

        query = get_rope(query)
        key = get_rope(key)

        return query, key

    def rel_embedding(self, hid_head: int, length_q: int, length_k: int,
                      dtype, query: jax.Array) -> jax.Array:
        range_vec_q = jnp.arange(length_q)
        range_vec_k = jnp.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat += length_q

        embeddings = nn.Embed(num_embeddings=2 * length_q + 1,
                              features=hid_head,
                              name='tr_rel_pos', dtype=dtype)(
            distance_mat.flatten())

        embeddings = nn.Dense(features=hid_head, use_bias=False,
                              name='tr_rel_pos_dense', dtype=dtype,
                              kernel_init=self.config.default_kernel_init,
                              bias_init=self.config.default_bias_init)(
            embeddings)

        # [q_length, kv_length, qk_depth_per_head]
        embeddings = embeddings.reshape(length_q, length_k, hid_head)

        attention_weights_rel_pos = jnp.einsum("bqhd,qkd->bhqk", query,
                                               embeddings)

        return attention_weights_rel_pos

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
        return mask

    @classmethod
    def causal_mask(cls, input_tokens):
        mask = cls.attention_mask(input_tokens)
        mask = jnp.full(mask.shape, 1)
        mask -= jnp.triu(mask, k=1)
        return mask


class MLP(nn.Module):
    config: TransformerConfig

    def setup(self):
        cfg = self.config
        self.dense0 = nn.Dense(features=cfg.mlp_dim_scale * cfg.features,
                               use_bias=cfg.use_bias, dtype=cfg.dtype,
                               name="tr_mlp_dense0",
                               kernel_init=cfg.default_kernel_init,
                               bias_init=cfg.default_bias_init)
        self.dense1 = nn.Dense(features=cfg.features, use_bias=cfg.use_bias,
                               dtype=cfg.dtype,
                               name="tr_mlp_dense1",
                               kernel_init=cfg.default_kernel_init,
                               bias_init=cfg.default_bias_init)

        self.dropout0 = nn.Dropout(rate=cfg.dropout_rate,
                                   deterministic=not cfg.training)
        self.dropout1 = nn.Dropout(rate=cfg.dropout_rate,
                                   deterministic=not cfg.training)

    @nn.compact
    def __call__(self, x):
        x = self.dense0(x)
        x = self.dropout0(x)
        x = netket.nn.elu(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        return x


class Encoder(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, input_embedding: jax.Array,
                 encoder_mask: jax.Array,
                 kv_cache: jax.Array) -> (jax.Array, jax.Array):
        cfg = self.config

        # Encoder layer
        x = input_embedding
        kv_cache_current = []
        for i in range(cfg.layers):
            # Multihead Attention
            if kv_cache is not None:
                kv = kv_cache[:, i, :, :, :]
            else:
                kv = None

            y = nn.RMSNorm()(x)
            y, kv = MultiheadAttention(config=cfg)(
                kv=y,
                q=y,
                mask=encoder_mask,
                kv_cache=kv
            )
            kv_cache_current.append(jnp.expand_dims(kv, axis=1))
            # Add & Norm
            x = y + x
            y = nn.RMSNorm()(x)
            # MLP
            y = MLP(cfg)(y)
            # Add & Norm
            x = y + x

        kv_cache_current = jnp.concatenate(kv_cache_current, axis=1)

        kv_cache = kv_cache_current

        return x, kv_cache


class Transformer(nn.Module):
    config: TransformerConfig
    states: Dict[int, int] = None

    def setup(self) -> None:
        cfg = self.config

        self.features_2 = cfg.features
        if cfg.embed_concat:
            self.features_2 //= 2

        self.embed = nn.Embed(num_embeddings=cfg.n_state,
                              features=self.features_2,
                              name="tr_embed",
                              dtype=cfg.dtype)

        self.dense_out = nn.Dense(features=2,
                                  use_bias=cfg.use_bias,
                                  name="tr_dense_out",
                                  dtype=cfg.dtype,
                                  kernel_init=cfg.default_kernel_init,
                                  bias_init=cfg.default_bias_init)

        self.enc = Encoder(cfg)

    def labs_embedding(self, batch_size: int, features_2: int) -> jax.Array:
        cfg = self.config

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

        return pos_embedding

    def abs_embedding(self, features_2: int) -> jax.Array:
        cfg = self.config

        base = 10000.0
        pos_embedding = jnp.zeros((cfg.length, features_2))
        position = jnp.arange(0, cfg.length)[:, None]
        div_term = jnp.exp((jnp.arange(0, features_2, 2) *
                            -(jnp.log(base) / features_2)))
        pos_embedding.at[:, 0::2].set(jnp.sin(position * div_term))
        pos_embedding.at[:, 1::2].set(jnp.cos(position * div_term))

        return pos_embedding

    def get_embedding(self, batch_size: int,
                      input_embedding: jax.Array) -> jax.Array:
        if self.config.pos_embed == "labs":
            pos_embedding = self.labs_embbeding(batch_size, self.features_2)
        elif self.config.pos_embed == "abs":
            pos_embedding = self.abs_embedding(self.features_2)
        else:
            pos_embedding = jnp.zeros_like(input_embedding)

        if self.config.embed_concat:
            input_embedding = jnp.concatenate([pos_embedding, input_embedding],
                                              axis=-1)
        else:
            input_embedding += pos_embedding

        return input_embedding

    def get_input(self, batch_size: int,
                  encoder_input_tokens: jax.Array) -> jax.Array:
        encoder_input_tokens = (encoder_input_tokens > 0).astype(jnp.int32)
        state_embedding = self.embed(encoder_input_tokens)
        input_embedding = state_embedding

        input_embedding = self.get_embedding(batch_size, input_embedding)

        return input_embedding

    @nn.compact
    def __call__(self, encoder_input_tokens: jax.Array = None,
                 autoregressive: bool = True,
                 generate: bool = False,
                 n_chains: int = 16) -> jax.Array:
        cfg = self.config

        if encoder_input_tokens is None and not generate:
            assert "encoder_input_tokens need for model"

        if not generate:
            batch_size = encoder_input_tokens.shape[0]
            input_embedding = self.get_input(batch_size, encoder_input_tokens)

        if autoregressive:
            if generate:
                def sample_one(p_row, key):
                    return jax.random.choice(key,
                                             a=jnp.array([0, 1]),
                                             p=p_row)

                input_embedding = jnp.zeros((n_chains,
                                             self.config.length + 1,
                                             cfg.features))
                kv_cache = None
                padding_mask = MultiheadAttention.causal_mask(
                    jnp.ones(input_embedding.shape[:-1]))
                samples = jnp.zeros((n_chains, self.config.length))
                key = jax.random.PRNGKey(42)
                for i in range(self.config.length):
                    x = input_embedding[:, :(i + 1), :]
                    mask = padding_mask[:, :, :(i + 1), :(i + 1)]

                    x, kv_cache = self.enc(input_embedding=x,
                                           encoder_mask=mask,
                                           kv_cache=kv_cache)
                    x = x[:, -1, :]
                    x = self.dense_out(x)
                    x = nn.softmax(x, axis=-1)

                    keys = jax.random.split(key, n_chains)
                    sample = jax.vmap(sample_one)(x, keys)
                    samples = samples.at[:, i].set(sample)
                    new_input_embedding = self.get_input(n_chains, sample)
                    input_embedding.at[:, (i + 1), :].set(new_input_embedding)

                x = samples
            else:
                def get_prob(x_row, token):
                    return x_row[token]

                encoder_input_tokens = (encoder_input_tokens > 0).astype(
                    jnp.int32)
                batch_size = encoder_input_tokens.shape[0]

                p = jnp.ones((batch_size,))
                input_embedding = jnp.concatenate(
                    [jnp.zeros((batch_size, 1, cfg.features)),
                     input_embedding],
                    axis=1)
                kv_cache = None
                padding_mask = MultiheadAttention.causal_mask(
                    jnp.ones(input_embedding.shape[:-1]))
                for i in range(self.config.length):
                    x = input_embedding[:, :(i + 1), :]
                    mask = padding_mask[:, :, :(i + 1), :(i + 1)]

                    x, kv_cache = self.enc(input_embedding=x,
                                           encoder_mask=mask,
                                           kv_cache=kv_cache)
                    x = x[:, -1, :]
                    x = self.dense_out(x)
                    x = nn.softmax(x, axis=-1)

                    p *= jax.vmap(get_prob)(x, encoder_input_tokens[:, i])

                x = p
        else:
            x = input_embedding
            padding_mask = MultiheadAttention.attention_mask(
                jnp.ones(x.shape[:-1]))
            x = self.enc(input_embedding=x,
                         encoder_mask=padding_mask)

            x = self.dense_out(x)

            x = nn.softmax(x, axis=-2)
            x = x.max(axis=-1)
            x = jnp.prod(x, axis=-2).squeeze(-1)

        return x


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
    print(tr.apply(params, input, generate=True, rngs={'dropout': key}))

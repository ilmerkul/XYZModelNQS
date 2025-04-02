from typing import Dict

import jax
import netket.nn
from flax import linen as nn
from flax import struct
from jax import lax
from jax import nn as jnn
from jax import numpy as jnp


@struct.dataclass
class EnergyBasedTransformerConfig:
    # Number of encoder layer pairs
    layers: int = 2

    # Number of units in dense layer
    mlp_dim_scale: int = 4

    # Tokens length
    length: int = 10

    # Number of embdedding dim
    features: int = 16

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

    symm: bool = True

    embed_concat: bool = False

    pos_embed: str = "rel"

    default_kernel_init: jnn.initializers.Initializer = jnn.initializers.normal(
        1e-1, dtype=jnp.float64)
    default_bias_init: jnn.initializers.Initializer = jnn.initializers.normal(
        1e-4, dtype=jnp.float64)


class MultiheadAttention(nn.Module):
    config: EnergyBasedTransformerConfig
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
    config: EnergyBasedTransformerConfig

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
        x = netket.nn.cosh(x)
        x = nn.Dense(features=cfg.features, use_bias=cfg.use_bias,
                     dtype=cfg.dtype,
                     kernel_init=cfg.default_kernel_init,
                     bias_init=cfg.default_bias_init)(x)
        x = nn.Dropout(rate=cfg.dropout_rate,
                       deterministic=not cfg.training)(x)
        return x


class Encoder(nn.Module):
    config: EnergyBasedTransformerConfig

    @nn.compact
    def __call__(self, input_embedding, encoder_mask):
        cfg = self.config

        # Encoder layer
        x = input_embedding
        for i in range(cfg.layers):
            # Multihead Attention
            # y = nn.LayerNorm()(x)
            y = nn.RMSNorm()(x)
            # y = nn.GroupNorm()(x)
            # y = nn.BatchNorm(use_running_average=True)(x)
            # y = nn.InstanceNorm()(x)
            # y = x
            y = MultiheadAttention(config=cfg)(
                kv=y,
                q=y,
                mask=encoder_mask
            )
            # Add & Norm
            x = y + x
            # y = nn.LayerNorm()(x)
            y = nn.RMSNorm()(x)
            # y = nn.GroupNorm()(x)
            # y = nn.BatchNorm(use_running_average=True)(x)
            # y = nn.InstanceNorm()(x)
            # y = x
            # MLP
            y = MLP(cfg)(y)
            # Add & Norm
            x = y + x

        return x


class EnergyBasedTransformer(nn.Module):
    config: EnergyBasedTransformerConfig
    states: Dict[int, int] = None
    key = jax.random.PRNGKey(42)

    @nn.compact
    def __call__(self, encoder_input_tokens, energy=None):
        cfg = self.config

        encoder_input_tokens = (encoder_input_tokens > 0).astype(jnp.int32)

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

        energy_dense = nn.Dense(features=features_2,
                                use_bias=cfg.use_bias,
                                dtype=cfg.dtype,
                                kernel_init=cfg.default_kernel_init,
                                bias_init=cfg.default_bias_init)

        enc = Encoder(cfg)
        final_dense = nn.Dense(features=1, use_bias=cfg.use_bias,
                               dtype=cfg.dtype,
                               kernel_init=cfg.default_kernel_init,
                               bias_init=cfg.default_bias_init)

        energy_emb = nn.Embed(num_embeddings=1,
                              features=batch_size,
                              dtype=cfg.dtype)
        params_energy_emb = energy_emb.init({'params': self.key},
                                            (jnp.zeros(shape=(1,),
                                                       dtype=jnp.int32)))

        psi_2_intermediate = []
        if energy is None:

            def computeEnergyBasedModel(param, inp):
                en = energy_emb.apply(param, inp)
                en += 0.001 * jax.random.normal(self.key, shape=en.shape)
                en = en.reshape(batch_size, 1)
                en_emb = energy_dense(en)[:, None, :]
                inp = jax.lax.stop_gradient(input_embedding)
                inp = jnp.concatenate([inp, en_emb], axis=1)

                padding_mask = MultiheadAttention.attention_mask(
                    jnp.ones(inp.shape[:-1]))
                out = enc(input_embedding=inp,
                          encoder_mask=padding_mask)
                out = out[:, -1, :]
                out = final_dense(out)

                out = out ** 2
                psi_2_intermediate.append(jax.lax.stop_gradient(out)[:, None])

                return -1 * jnp.sum(out)

            for i in range(10):
                grad_fun = jax.grad(computeEnergyBasedModel)
                grad = grad_fun(params_energy_emb,
                                (jnp.zeros(shape=(1,), dtype=jnp.int32)))
                params_energy_emb = jax.tree.map(lambda x, y: x - 0.01 * y,
                                                 params_energy_emb,
                                                 grad)

            energy = params_energy_emb["params"]["embedding"].reshape(
                batch_size,
                1, 1)

        energy_embedding = energy_dense(energy)
        x = jnp.concatenate([input_embedding, energy_embedding], axis=1)

        padding_mask = MultiheadAttention.attention_mask(
            jnp.ones(x.shape[:-1]))
        x = enc(input_embedding=x,
                encoder_mask=padding_mask)
        x = x[:, -1, :]
        x = final_dense(x)

        if psi_2_intermediate:
            psi_2_intermediate = jnp.concatenate(psi_2_intermediate,
                                                 axis=1)

        return x.squeeze(-1), energy, psi_2_intermediate

    def inp_2_state(self, inp):
        if self.states is None:
            states = jnp.unique(inp).tolist()
            self.states = {state: i for i, state in enumerate(states)}

        def f(x):
            if isinstance(x, int):
                return self.states[x]
            return lax.map(f, x)

        return lax.map(f, inp)


class EnergyOptimModel(nn.Module):
    m: EnergyBasedTransformer

    @nn.compact
    def __call__(self, parameters, inp):
        energy = nn.Embed(num_embeddings=1,
                          features=1,
                          dtype=cfg.dtype)(jnp.zeros(shape=(inp.shape[0],),
                                                     dtype=jnp.int32))
        energy = energy[:, None]

        return self.m.apply(parameters, inp, energy)


if __name__ == "__main__":
    n = 11
    cfg = EnergyBasedTransformerConfig(training=True, symm=True, length=n)
    tr = EnergyBasedTransformer(cfg)

    key = jax.random.PRNGKey(42)
    inp = jnp.ones(16 * n)
    inp = inp.at[jax.random.randint(key, (16, n), 0, 16 * n)].set(-1)
    inp = inp.reshape(16, n)
    energy = jax.random.normal(key=key,
                               shape=(inp.shape[0], 1, 1),
                               dtype=cfg.dtype)
    init_rngs = {'params': key,
                 'dropout': key}
    params = tr.init(init_rngs, inp)
    print(inp)
    print(tr.apply(params, inp, rngs={'dropout': key}))

    km = EnergyOptimModel(tr)
    params_km = km.init(init_rngs, params, inp, {'dropout': key})


    def computeEnergyBasedModel(params_km, params, inp):
        x = km.apply(params_km, params, inp)
        x = jnp.linalg.norm(x) ** 2

        return x


    for i in range(10):
        grad_fun = jax.grad(computeEnergyBasedModel)
        grad = grad_fun(params_km, params, inp)
        params_km = jax.tree.map(lambda x, y: x - 0.007 * y, params_km, grad)
        print(i, params_km, grad)

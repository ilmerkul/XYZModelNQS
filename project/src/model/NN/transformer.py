from jax import numpy as jnp, nn as jnn
from flax import linen as nn
import jax
from flax import struct
from jax import lax

from typing import Dict

default_kernel_init = jnn.initializers.normal(1e-1, dtype=jnp.complex64)
default_bias_init = jnn.initializers.normal(1e-4, dtype=jnp.complex64)


@struct.dataclass
class Config:
    # Number of encoder layer pairs
    layers: int = 2

    # Number of units in dense layer
    mlp_dim_scale: int = 4

    # Tokens length
    length: int = 10

    # Number of embdedding dim
    features: int = 32

    # Number of heads in multihead attention
    num_heads: int = 2

    # Bias
    use_bias: bool = False

    # Droput rate
    dropout_rate: float = 0.05

    # Dropout or not
    training: bool = False

    # Random seed
    seed: int = 0

    n_state: int = 2

    dtype: jnp.dtype = jnp.complex64

    symm: bool = False


class MultiheadAttention(nn.Module):
    config: Config
    decode: bool = False

    @nn.compact
    def __call__(self, kv, q, mask):
        # Initialize config
        cfg = self.config
        # Initialize key, query, value through Dense layer
        query = nn.Dense(features=cfg.features, use_bias=cfg.use_bias,
                         name='query', dtype=cfg.dtype,
                         kernel_init=default_kernel_init,
                         bias_init=default_bias_init)(q)
        key = nn.Dense(features=cfg.features, use_bias=cfg.use_bias,
                       name='key', dtype=cfg.dtype,
                       kernel_init=default_kernel_init,
                       bias_init=default_bias_init)(kv)
        value = nn.Dense(features=cfg.features, use_bias=cfg.use_bias,
                         name='value', dtype=cfg.dtype,
                         kernel_init=default_kernel_init,
                         bias_init=default_bias_init)(kv)

        batch_size = query.shape[0]

        # Layer norm
        query = nn.LayerNorm()(query)
        key = nn.LayerNorm()(key)
        value = nn.LayerNorm()(value)

        head_dim = cfg.features // cfg.num_heads

        # Split head
        query = query.reshape(batch_size, cfg.length, cfg.num_heads, head_dim)
        key = key.reshape(batch_size, cfg.length, cfg.num_heads, head_dim)
        value = value.reshape(batch_size, cfg.length, cfg.num_heads, head_dim)
        # Scaled dot-product attention
        logits = self.scaled_dot_product_attention(key, query, value, mask,
                                                   cfg.dtype)

        # Concat
        logits = logits.reshape(batch_size, cfg.length, cfg.features)

        # Linear
        logits = nn.Dense(features=cfg.features,
                          use_bias=cfg.use_bias,
                          name='attention_weights',
                          dtype=cfg.dtype,
                          kernel_init=default_kernel_init,
                          bias_init=default_bias_init
                          )(logits)
        logits = nn.Dropout(rate=cfg.dropout_rate,
                            deterministic=not cfg.training)(logits)

        return logits

    def scaled_dot_product_attention(self, key, query, value, mask, dtype):
        """ Matmul
            query: [batch, q_length, num_heads, qk_depth_per_head]
            key: [batch, kv_length, num_heads, qk_depth_per_head]
            -> qk: [batch, num_heads, q_length, kv_length]
        """
        attention_weights = jnp.einsum("bqhd,bkhd->bhqk", query, key)
        # Scale
        d_k = query.shape[-1]
        attention_weights = attention_weights / jnp.sqrt(d_k)
        # Mask
        num_heads = attention_weights.shape[1]
        attention_weights = lax.select(
            jnp.repeat(mask, num_heads, axis=1) > 0,
            attention_weights,
            jnp.full(attention_weights.shape, -jnp.inf).astype(dtype)
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
    def attention_mask(cls, input_tokens, dtype):
        """Mask-making helper for attention weights (mask for padding)
        Args:
            input_tokens: [batch_size, tokens_length]
        return:
            mask: [batch_size, num_heads=1, query_length, key_value_length]
        """
        mask = jnp.multiply(jnp.expand_dims(input_tokens, axis=-1),
                            jnp.expand_dims(input_tokens, axis=-2))
        mask = jnp.expand_dims(mask, axis=-3)
        mask = lax.select(
            mask > 0,
            jnp.full(mask.shape, 1).astype(dtype),
            jnp.full(mask.shape, 0).astype(dtype)
        )
        return mask

    @classmethod
    def causal_mask(cls, input_tokens, dtype):
        mask = cls.attention_mask(input_tokens, dtype)
        mask = jnp.full(mask.shape, -jnp.inf)
        mask = jnp.triu(mask, k=1)
        return mask


class MLP(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, x):
        # Initialize config
        cfg = self.config

        x = nn.Dense(features=cfg.mlp_dim_scale * cfg.features,
                     use_bias=cfg.use_bias, dtype=cfg.dtype,
                     kernel_init=default_kernel_init,
                     bias_init=default_bias_init)(x)
        x = nn.Dropout(rate=cfg.dropout_rate,
                       deterministic=not cfg.training)(x)
        x = nn.elu(x)
        x = nn.Dense(features=cfg.features, use_bias=cfg.use_bias,
                     dtype=cfg.dtype,
                     kernel_init=default_kernel_init,
                     bias_init=default_bias_init)(x)
        x = nn.Dropout(rate=cfg.dropout_rate,
                       deterministic=not cfg.training)(x)
        return x


class Encoder(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, input_embeding, encoder_mask):
        cfg = self.config
        # Encoder layer
        x = input_embeding
        for i in range(cfg.layers):
            # Multihead Attention
            y = MultiheadAttention(config=cfg)(
                kv=x,
                q=x,
                mask=encoder_mask
            )
            # Add & Norm
            y = y + x
            x = nn.LayerNorm()(y)
            # MLP
            y = MLP(cfg)(x)
            # Add & Norm
            y = y + x
            x = nn.LayerNorm()(y)

        return x


class Transformer(nn.Module):
    config: Config
    states: Dict[int, int] = None

    @nn.compact
    def __call__(self, encoder_input_tokens):
        cfg = self.config

        encoder_input_tokens = (encoder_input_tokens > 0).astype(jnp.int32)

        padding_mask = MultiheadAttention.attention_mask(
            jnp.ones(encoder_input_tokens.shape),
            cfg.dtype)

        features_2 = cfg.features // 2
        batch_size = encoder_input_tokens.shape[0]

        if cfg.symm:
            pos_num_embeddings = (cfg.length + 1) // 2
            pos = jnp.concatenate([jnp.arange(pos_num_embeddings),
                                   jnp.arange(pos_num_embeddings -
                                              (1 if cfg.length % 2 == 0
                                               else 0), -1, -1)],
                                  axis=0)
            print(pos)
        else:
            pos_num_embeddings = cfg.length
            pos = jnp.arange(cfg.length)

        pos = jnp.repeat(jnp.expand_dims(pos, axis=0), batch_size, axis=0)
        pos_embeding = nn.Embed(num_embeddings=pos_num_embeddings,
                                features=features_2, dtype=cfg.dtype)(pos)

        state_embeding = nn.Embed(num_embeddings=cfg.n_state,
                                  features=features_2,
                                  dtype=cfg.dtype)(encoder_input_tokens)

        input_embeding = jnp.concatenate([pos_embeding, state_embeding],
                                         axis=-1)
        input_embeding += lax.complex(0., 0.001)

        # Encoder layer
        x = Encoder(cfg)(input_embeding=input_embeding,
                         encoder_mask=padding_mask)

        # Linear
        x = jnp.sum(x, axis=-2)
        x = nn.Dense(features=1, use_bias=cfg.use_bias, dtype=cfg.dtype,
                     kernel_init=default_kernel_init,
                     bias_init=default_bias_init)(x)
        x = nn.Dropout(rate=cfg.dropout_rate,
                       deterministic=not cfg.training)(x)

        return x.squeeze(-1)

    def inp_2_state(self, inp):
        if self.states is None:
            states = jnp.unique(inp).tolist()
            self.states = {state: i for i, state in enumerate(states)}

        def f(x):
            if isinstance(x, int):
                return self.states[x]
            return lax.map(f, x)

        return lax.map(f, inp)


if __name__ == "__main__":
    n = 10
    tr = Transformer(Config(training=True, symm=True))

    key = jax.random.PRNGKey(42)
    input = jax.random.randint(key, (16, n), -1, 2)
    init_rngs = {'params': key,
                 'dropout': key}
    params = tr.init(init_rngs, input)
    print(input)
    print(tr.apply(params, input, rngs={'dropout': key}))

import os
import pathlib
import sys
from dataclasses import dataclass, field
from random import random
from typing import Any, Tuple

import jax
from flax import linen as nn
from jax import lax
from jax import nn as jnn
from jax import numpy as jnp

project_path = pathlib.Path(os.getcwd())

sys.path.append(project_path.as_posix())
from src.model.NN import NNConfig
from src.model.struct import ChainConfig


class KVCache:
    def __init__(self, batch_size: int, n_layers: int, length: int, features: int):
        self.key: jnp.ndarray = jnp.zeros((batch_size, n_layers, length, features))
        self.value: jnp.ndarray = jnp.zeros((batch_size, n_layers, length, features))


@dataclass(frozen=True)
class TransformerConfig(NNConfig):
    chain: ChainConfig
    autoregressive: bool
    use_bias: bool
    use_dropout: bool
    training: bool
    symm: bool
    embed_concat: bool
    state_inverse: bool
    dropout_rate: float
    inverse_iter_rate: float
    seed: int
    pos_embed: str
    eps: float
    dtype: Any = field(default_factory=lambda: jnp.float32)

    default_kernel_init: Any = field(init=False)
    default_bias_init: Any = field(init=False)
    n_state: int = field(init=False)
    layers: int = field(init=False)
    features: int = field(init=False)
    mlp_dim_scale: int = field(init=False)
    num_heads: int = field(init=False)

    def __post_init__(self):
        default_kernel_init = nn.initializers.xavier_uniform(dtype=self.dtype)
        default_bias_init = jnn.initializers.constant(0.0, dtype=self.dtype)
        n_state = int(2 * self.chain.spin + 1)
        layers = int(self.chain.n**0.5)
        features = int(self.chain.n**1.5)
        if features % 2 != 0:
            features += 1
        mlp_dim_scale = int(self.chain.n**0.5)

        for i in range(layers, 0, -1):
            if features % i == 0:
                num_heads = i
                break

        object.__setattr__(self, "default_kernel_init", default_kernel_init)
        object.__setattr__(self, "default_bias_init", default_bias_init)
        object.__setattr__(self, "n_state", n_state)
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "mlp_dim_scale", mlp_dim_scale)
        object.__setattr__(self, "num_heads", num_heads)


@dataclass(frozen=True)
class PosEmbType:
    ROTARY: str = "rope"
    LEARN_ABSOLUTE: str = "labs"
    ABSOLUTE: str = "abs"
    ZERO: str = "zero"
    RELATIVE: str = "rel"


class LearnedPositionalEmbedding(nn.Module):
    num_embeddings: int
    features: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, batch_size: int) -> jax.Array:
        pos = jnp.arange(self.num_embeddings)
        pos = jnp.repeat(pos[None, :], batch_size, axis=0)
        return nn.Embed(
            num_embeddings=self.num_embeddings, features=self.features, dtype=self.dtype
        )(pos)


class MultiheadAttention(nn.Module):
    config: TransformerConfig
    n_layer: int
    decode: bool = False

    def setup(self):
        cfg = self.config
        self.query_dense = nn.Dense(
            features=cfg.features,
            use_bias=cfg.use_bias,
            name="tr_query_dense",
            dtype=cfg.dtype,
            kernel_init=cfg.default_kernel_init,
            bias_init=cfg.default_bias_init,
        )
        self.key_dense = nn.Dense(
            features=cfg.features,
            use_bias=cfg.use_bias,
            name="tr_key_dense",
            dtype=cfg.dtype,
            kernel_init=cfg.default_kernel_init,
            bias_init=cfg.default_bias_init,
        )
        self.value_dense = nn.Dense(
            features=cfg.features,
            use_bias=cfg.use_bias,
            name="tr_value_dense",
            dtype=cfg.dtype,
            kernel_init=cfg.default_kernel_init,
            bias_init=cfg.default_bias_init,
        )
        self.logits_dense = nn.Dense(
            features=cfg.features,
            use_bias=cfg.use_bias,
            name="tr_attention_weights",
            dtype=cfg.dtype,
            kernel_init=cfg.default_kernel_init,
            bias_init=cfg.default_bias_init,
        )
        self.dropout = nn.Dropout(rate=cfg.dropout_rate, deterministic=not cfg.training)

    @nn.compact
    def __call__(
        self,
        kv: jax.Array,
        q: jax.Array,
        mask: jax.Array,
        kv_cache: KVCache,
        n_iter: int,
    ) -> Tuple[jax.Array, KVCache]:
        cfg = self.config
        query = self.query_dense(q)

        if self.config.autoregressive:
            key = self.key_dense(kv[:, n_iter, :])
            value = self.value_dense(kv[:, n_iter, :])
        else:
            key = self.key_dense(kv)
            value = self.value_dense(kv)

        if self.config.autoregressive:
            kv_cache.key = kv_cache.key.at[:, self.n_layer, n_iter, :].set(key)
            kv_cache.value = kv_cache.value.at[:, self.n_layer, n_iter, :].set(value)

        batch_size = query.shape[0]

        if self.config.autoregressive:
            key = kv_cache.key[:, self.n_layer, : (n_iter + 1), :]
            value = kv_cache.value[:, self.n_layer, : (n_iter + 1), :]

        # query = nn.LayerNorm()(query)
        # key = nn.LayerNorm()(key)
        # value = nn.LayerNorm()(value)

        assert cfg.features % cfg.num_heads == 0
        head_dim = cfg.features // cfg.num_heads

        query = query.reshape(batch_size, -1, cfg.num_heads, head_dim)
        key = key.reshape(batch_size, -1, cfg.num_heads, head_dim)
        value = value.reshape(batch_size, -1, cfg.num_heads, head_dim)

        logits = self.scaled_dot_product_attention(
            key, query, value, mask, cfg.dtype, cfg.pos_embed
        )

        logits = logits.reshape(batch_size, -1, cfg.features)

        logits = self.logits_dense(logits)
        if self.config.use_dropout:
            logits = self.dropout(logits)

        return logits, kv_cache

    def scaled_dot_product_attention(
        self,
        key: jax.Array,
        query: jax.Array,
        value: jax.Array,
        mask: jax.Array,
        dtype,
        pos_embed: str,
    ):
        """Matmul
        query: [batch, q_length, num_heads, qk_depth_per_head]
        key: [batch, kv_length, num_heads, qk_depth_per_head]
        -> qk: [batch, num_heads, q_length, kv_length]
        """
        length_q = query.shape[1]
        length_k = key.shape[1]
        hid_head = query.shape[3]

        if pos_embed == PosEmbType.ROTARY:
            pos_embedding = self.rope_embedding(length_q, hid_head)
            query *= pos_embedding
            key *= pos_embedding

        attention_weights = jnp.einsum("bqhd,bkhd->bhqk", query, key)

        if pos_embed == PosEmbType.RELATIVE:
            attention_weights_rel_pos = self.rel_embedding(
                hid_head, length_q, length_k, dtype, query
            )
            attention_weights += attention_weights_rel_pos

        d_k = query.shape[-1]
        attention_weights /= jnp.sqrt(d_k)
        if mask is not None:
            attention_weights = jnp.where(mask, attention_weights, jnp.finfo(dtype).min)

        attention_weights = nn.softmax(attention_weights, axis=-1).astype(dtype)
        """ Matmul
            qk: [batch, num_heads, q_length, kv_length]
            value: [batch, kv_length, num_heads, v_depth_per_head]
            -> Return: [batch, length, num_heads, v_depth_per_head]
        """

        return jnp.einsum("bhqk,bkhd->bqhd", attention_weights, value)

    def rope_embedding(self, length: int, features: int) -> Tuple[jax.Array, jax.Array]:
        base = 10000.0
        inv_freq = 1.0 / (base ** (jnp.arange(0, features, 2) / features))
        sinusoid_inp = jnp.einsum("i,j->ij", jnp.arange(length), inv_freq)

        sin = jnp.sin(sinusoid_inp)
        if features % 2 != 0:
            sinusoid_inp = sinusoid_inp[:, :-1]
        cos = jnp.cos(sinusoid_inp)

        emb = jnp.concat([sin, cos], axis=1)[:, None, :]

        return emb

    def rel_embedding(
        self, hid_head: int, length_q: int, length_k: int, dtype, query: jax.Array
    ) -> jax.Array:
        range_vec_q = jnp.arange(length_q)
        range_vec_k = jnp.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat += length_q

        embeddings = nn.Embed(
            num_embeddings=2 * length_q + 1,
            features=hid_head,
            name="tr_rel_pos",
            dtype=dtype,
        )(distance_mat.flatten())

        embeddings = nn.Dense(
            features=hid_head,
            use_bias=False,
            name="tr_rel_pos_dense",
            dtype=dtype,
            kernel_init=self.config.default_kernel_init,
            bias_init=self.config.default_bias_init,
        )(embeddings)

        # [q_length, kv_length, qk_depth_per_head]
        embeddings = embeddings.reshape(length_q, length_k, hid_head)

        attention_weights_rel_pos = jnp.einsum("bqhd,qkd->bhqk", query, embeddings)

        return attention_weights_rel_pos

    @classmethod
    def attention_mask(cls, input_tokens):
        mask = jnp.multiply(input_tokens[:, :, None], input_tokens[:, None, :])
        mask = mask[:, None, ...]
        mask = lax.select(mask > 0, jnp.full(mask.shape, 1), jnp.full(mask.shape, 0))

        return mask

    @classmethod
    def causal_mask(cls, input_tokens):
        # mask = cls.attention_mask(input_tokens)
        mask = jnp.full(
            (input_tokens.shape[0], 1, input_tokens.shape[1], input_tokens.shape[1]), 1
        )
        # mask -= jnp.triu(mask, k=1)
        return mask


class MLP(nn.Module):
    config: TransformerConfig

    def setup(self):
        cfg = self.config
        self.dense0 = nn.Dense(
            features=cfg.mlp_dim_scale * cfg.features,
            use_bias=cfg.use_bias,
            dtype=cfg.dtype,
            name="tr_mlp_dense0",
            kernel_init=cfg.default_kernel_init,
            bias_init=cfg.default_bias_init,
        )
        self.dense1 = nn.Dense(
            features=cfg.features,
            use_bias=cfg.use_bias,
            dtype=cfg.dtype,
            name="tr_mlp_dense1",
            kernel_init=cfg.default_kernel_init,
            bias_init=cfg.default_bias_init,
        )

        self.dropout0 = nn.Dropout(
            rate=cfg.dropout_rate, deterministic=not cfg.training
        )
        self.dropout1 = nn.Dropout(
            rate=cfg.dropout_rate, deterministic=not cfg.training
        )

    @nn.compact
    def __call__(self, x):
        x = self.dense0(x)
        if self.config.use_dropout:
            x = self.dropout0(x)
        x = nn.tanh(x)
        x = self.dense1(x)
        if self.config.use_dropout:
            x = self.dropout1(x)
        return x


class EncoderBlock(nn.Module):
    config: TransformerConfig
    n_layer: int

    def setup(self) -> None:
        self.norm1 = nn.RMSNorm(dtype=self.config.dtype)
        self.norm2 = nn.RMSNorm(dtype=self.config.dtype)

        self.mha = MultiheadAttention(config=self.config, n_layer=self.n_layer)

        self.mlp = MLP(config=self.config)

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        mask: jax.Array,
        n_iter: int = None,
        kv_cache: KVCache = None,
    ) -> Tuple[jax.Array, KVCache]:
        y = self.norm1(x)
        y, kv_cache = self.mha(
            kv=y,
            q=y,
            mask=mask,
            kv_cache=kv_cache,
            n_iter=n_iter,
        )
        x = y + x
        y = self.norm2(x)
        y = self.mlp(x)
        x = y + x
        return x


class Encoder(nn.Module):
    config: TransformerConfig

    def setup(self) -> None:
        self.layers = [
            EncoderBlock(config=self.config, n_layer=n_layer)
            for n_layer in range(self.config.layers)
        ]

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        encoder_mask: jax.Array,
        n_iter: int = None,
        kv_cache: KVCache = None,
    ) -> Tuple[jax.Array, KVCache]:
        cfg = self.config

        batch_size = x.shape[0]

        if kv_cache is not None and n_iter is None:
            raise ValueError("kv_cache is not None and n_iter is None")

        if n_iter is None:
            n_iter = cfg.chain.n

        if kv_cache is None and cfg.autoregressive:
            kv_cache = KVCache(
                batch_size=batch_size,
                n_layers=self.config.layers,
                length=self.config.chain.n + 1,
                features=self.config.features,
            )
        for layer in self.layers:
            x = layer(x, mask=encoder_mask, n_iter=n_iter, kv_cache=kv_cache)

        return x, kv_cache


class OutputHead(nn.Module):
    config: TransformerConfig

    def setup(self):
        features_out = self.config.n_state
        if not self.config.autoregressive:
            features_out = 1

        self.dense_out = nn.Dense(
            features=features_out,
            use_bias=self.config.use_bias,
            name="tr_dense_out",
            dtype=self.config.dtype,
            kernel_init=self.config.default_kernel_init,
            bias_init=self.config.default_bias_init,
        )

        self.phase_dense_out = nn.Dense(
            features=1,
            use_bias=self.config.use_bias,
            name="tr_phase_dense_out",
            dtype=self.config.dtype,
            kernel_init=self.config.default_kernel_init,
            bias_init=self.config.default_bias_init,
        )

        self.norm = nn.RMSNorm(dtype=self.config.dtype)

    def get_prob(self, x_row, token):
        return x_row[token]

    def sample_one(self, p_row, key):
        return jax.random.choice(key, a=jnp.array(range(self.config.n_state)), p=p_row)

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        encoder_input_tokens: jax.Array = None,
        generate: bool = False,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        batch_size = x.shape[0]

        x = self.norm(x)

        x = x[:, 0, :]
        x = self.dense_out(x)
        phase = jnp.pi * nn.tanh(self.phase_dense_out(x)).squeeze(-1)
        if generate:
            x = nn.softmax(x, axis=-1)
            key = jax.random.PRNGKey(seed=self.config.seed)
            keys = jax.random.split(key, batch_size)
            sample = jax.vmap(self.sample_one)(x, keys)
            prob = jax.vmap(self.get_prob)(x, sample)
        elif self.config.autoregressive:
            x = nn.softmax(x, axis=-1)
            prob = jax.vmap(self.get_prob)(x, encoder_input_tokens)
            sample = None
        else:
            prob = nn.sigmoid(x.squeeze(-1))
            sample = None

        return prob, phase, sample


class Embed(nn.Module):
    config: TransformerConfig

    def setup(self) -> None:
        self.features_2 = self.config.features
        if self.config.embed_concat:
            self.features_2 //= 2

        self.embed = nn.Embed(
            num_embeddings=self.config.n_state + 1,
            features=self.features_2,
            name="tr_embed",
            dtype=self.config.dtype,
        )

    def abs_embedding(self, features_2: int) -> jax.Array:
        cfg = self.config

        base = 10000.0
        pos_embedding = jnp.zeros((cfg.chain.n, features_2))
        position = jnp.arange(0, cfg.chain.n)[:, None]
        div_term = jnp.exp(
            (jnp.arange(0, features_2, 2) * -(jnp.log(base) / features_2))
        )
        pos_embedding.at[:, 0::2].set(jnp.sin(position * div_term))
        pos_embedding.at[:, 1::2].set(jnp.cos(position * div_term))

        return pos_embedding

    def get_embedding(self, batch_size: int, input_embedding: jax.Array) -> jax.Array:
        if self.config.pos_embed == PosEmbType.LEARN_ABSOLUTE:
            pos_embedding = LearnedPositionalEmbedding(
                num_embeddings=self.config.chain.n,
                features=self.features_2,
                dtype=self.config.dtype,
            )(batch_size)
        elif self.config.pos_embed == PosEmbType.ABSOLUTE:
            pos_embedding = self.abs_embedding(self.features_2)
        elif self.config.pos_embed == PosEmbType.ROTARY:
            pos_embedding = jnp.zeros_like(input_embedding, dtype=self.config.dtype)
        elif self.config.pos_embed == PosEmbType.ZERO:
            pos_embedding = jnp.zeros_like(input_embedding, dtype=self.config.dtype)
        else:
            raise ValueError("pos_embed error")

        if self.config.embed_concat:
            input_embedding = jnp.concat([pos_embedding, input_embedding], axis=-1)
        else:
            input_embedding += pos_embedding

        return input_embedding

    def get_input(self, batch_size: int, encoder_input_tokens: jax.Array) -> jax.Array:
        signs = encoder_input_tokens
        encoder_input_tokens_state = encoder_input_tokens
        if self.config.state_inverse:
            encoder_input_tokens_state = jnp.abs(encoder_input_tokens)
        encoder_input_tokens_state = (
            (encoder_input_tokens_state + self.config.chain.spin + 0.5) / 2
        ).astype(jnp.int32)
        encoder_input_tokens = (
            (encoder_input_tokens + self.config.chain.spin + 0.5) / 2
        ).astype(jnp.int32)
        firt_embedding = self.embed(jnp.full((batch_size, 1), self.config.n_state))
        state_embedding = self.embed(encoder_input_tokens_state)
        if self.config.state_inverse:
            state_embedding = jnp.where(
                signs[..., None] > 0, state_embedding, -state_embedding
            )

        state_embedding_direct = jnp.concat([firt_embedding, state_embedding], axis=1)
        state_embedding_inverse = jnp.concat(
            [firt_embedding, jnp.flip(state_embedding, axis=1)], axis=1
        )

        state_embedding_direct = self.get_embedding(batch_size, state_embedding_direct)
        state_embedding_inverse = self.get_embedding(
            batch_size, state_embedding_inverse
        )

        return state_embedding_direct, state_embedding_inverse, encoder_input_tokens

    @nn.compact
    def __call__(self, batch_size, sample) -> jax.Array:
        return self.get_input(batch_size, sample)


class Transformer(nn.Module):
    config: TransformerConfig

    def setup(self) -> None:
        self.embed = Embed(config=self.config)

        self.enc = Encoder(config=self.config)

        self.out_head = OutputHead(config=self.config)

    def enc_iter(
        self,
        x: jax.Array,
        encoder_mask: jax.Array,
        encoder_input_tokens: jax.Array = None,
        kv_cache=None,
        n_iter: int = None,
    ):
        if n_iter is not None:
            x = x[:, : (n_iter + 1), :]
            encoder_mask = encoder_mask[:, :, : (n_iter + 1), : (n_iter + 1)]

        x, kv_cache = self.enc(
            x=x,
            encoder_mask=encoder_mask,
            kv_cache=kv_cache,
            n_iter=n_iter,
        )
        return self.out_head(x=x, encoder_input_tokens=encoder_input_tokens), kv_cache

    @nn.compact
    def __call__(
        self,
        encoder_input_tokens: jax.Array = None,
        generate: bool = False,
        n_chains: int = 16,
    ) -> jax.Array:
        cfg = self.config

        if encoder_input_tokens is None and not generate:
            assert "encoder_input_tokens need for model"

        if encoder_input_tokens is None:
            batch_size = n_chains
        else:
            batch_size = encoder_input_tokens.shape[0]

        if generate:
            input_embedding = jnp.zeros(
                (batch_size, self.config.chain.n + 1, cfg.features)
            )
        else:
            input_embedding, input_embedding_inverse, encoder_input_tokens = self.embed(
                batch_size, encoder_input_tokens
            )

        padding_mask = MultiheadAttention.causal_mask(
            jnp.ones(input_embedding.shape[:-1])
        )

        if generate:

            kv_cache = None

            prob = jnp.ones((batch_size,))
            samples = jnp.zeros((batch_size, self.config.chain.n))
            for n_iter in range(self.config.chain.n):

                (prob_iter, phase, sample), kv_cache = self.enc_iter(
                    x=input_embedding,
                    encoder_mask=padding_mask,
                    kv_cache=kv_cache,
                    n_iter=n_iter,
                )

                samples = samples.at[:, n_iter].set(sample)
                new_input_embedding, _, _ = self.embed(batch_size, sample)
                input_embedding.at[:, (n_iter + 1), :].set(new_input_embedding)
                prob *= prob_iter

            σ = samples
        elif cfg.autoregressive:

            def enc_iter(input_embedding, encoder_input_tokens):
                prob = jnp.ones((batch_size,))
                kv_cache = None
                for n_iter in range(self.config.chain.n + 1):
                    (prob_iter, phase, _), kv_cache = self.enc_iter(
                        x=input_embedding,
                        encoder_mask=padding_mask,
                        encoder_input_tokens=encoder_input_tokens[:, n_iter],
                        kv_cache=kv_cache,
                        n_iter=n_iter,
                    )
                    prob *= prob_iter

                return prob, phase

            prob, phase = enc_iter(
                input_embedding,
                encoder_input_tokens,
            )
            if self.config.symm:
                r = random()
                prob1, phase1 = enc_iter(
                    input_embedding_inverse,
                    encoder_input_tokens,
                )

                if 2 * self.config.inverse_iter_rate < r or not self.config.training:
                    prob = (prob + prob1) / 2.0
                    phase = (phase + phase1) / 2.0
                elif r > self.config.inverse_iter_rate:
                    prob = prob1
                    phase = phase1
        else:

            (prob, phase, _), _ = self.enc_iter(
                x=input_embedding, encoder_mask=padding_mask
            )
            if self.config.symm:
                (prob1, phase1, _), _ = self.enc_iter(
                    x=input_embedding_inverse, encoder_mask=padding_mask
                )
                r = random()

                if 2 * self.config.inverse_iter_rate < r or not self.config.training:
                    prob = (prob + prob1) / 2.0
                    phase = (prob + phase1) / 2.0
                elif r > self.config.inverse_iter_rate:
                    prob = prob1
                    phase = phase1

        log_prob = jnp.log(prob + cfg.eps) + 1j * phase

        if not generate:
            return log_prob

        return σ, log_prob


if __name__ == "__main__":
    chain_cfg = ChainConfig(
        n=20,
        j=-1,
        h=0.0,
        lam=1,
        gamma=0,
        spin=1 / 2,
    )
    transformer_config = TransformerConfig(
        chain=chain_cfg,
        use_bias=True,
        use_dropout=False,
        dropout_rate=0.1,
        inverse_iter_rate=0.3,
        training=True,
        seed=42,
        autoregressive=False,
        dtype=jnp.float64,
        symm=True,
        embed_concat=False,
        state_inverse=True,
        pos_embed=PosEmbType.ROTARY,
        eps=1e-10,
    )
    tr = Transformer(transformer_config)

    key = jax.random.PRNGKey(42)
    input = jnp.ones(16 * chain_cfg.n)
    input = input.at[
        jax.random.randint(key, (16, chain_cfg.n), 0, 16 * chain_cfg.n)
    ].set(-1)
    input = input.reshape(16, chain_cfg.n)
    print(input)
    init_rngs = {"params": key, "dropout": key}
    params = tr.init(init_rngs, input)
    output = tr.apply(params, input, generate=False, rngs={"dropout": key})
    print(output)
    print(output.shape)

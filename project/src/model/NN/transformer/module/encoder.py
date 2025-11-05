from typing import Tuple

import jax
import jax.random as jrnd
from flax import linen as nn
from jax import numpy as jnp
from src.model.NN.transformer.module import (
    KVCache,
    PhysicsInformedAttention,
    TransformerConfig,
)


class MLP(nn.Module):
    config: TransformerConfig
    n_layer: int

    def setup(self):
        cfg = self.config
        self.dense0 = nn.Dense(
            features=cfg.mlp_dim_scale * cfg.features,
            use_bias=cfg.use_bias,
            dtype=cfg.dtype,
            name=f"tr_mlp_dense0_{self.n_layer}",
            kernel_init=cfg.default_kernel_init,
            bias_init=cfg.default_bias_init,
        )
        self.dense1 = nn.Dense(
            features=cfg.features,
            use_bias=cfg.use_bias,
            dtype=cfg.dtype,
            name=f"tr_mlp_dense1_{self.n_layer}",
            kernel_init=cfg.default_kernel_init,
            bias_init=cfg.default_bias_init,
        )

        self.dropout0 = nn.Dropout(
            rate=cfg.dropout_rate,
            deterministic=not cfg.training,
            name=f"tr_mlp_dropout0_{self.n_layer}",
        )
        self.dropout1 = nn.Dropout(
            rate=cfg.dropout_rate,
            deterministic=not cfg.training,
            name=f"tr_mlp_dropout1_{self.n_layer}",
        )

    @nn.compact
    def __call__(self, x):
        x = self.dense0(x)
        if self.config.use_dropout:
            x = self.dropout0(x)
        x = nn.gelu(x)
        x = self.dense1(x)
        if self.config.use_dropout:
            x = self.dropout1(x)
        return x


class EncoderBlock(nn.Module):
    config: TransformerConfig
    n_layer: int

    def setup(self) -> None:
        self.norm1 = nn.LayerNorm(
            dtype=self.config.dtype, name=f"tr_encoder_norm0_{self.n_layer}"
        )
        self.norm2 = nn.LayerNorm(
            dtype=self.config.dtype, name=f"tr_encoder_norm1_{self.n_layer}"
        )

        # self.mha = MultiheadAttention(config=self.config, n_layer=self.n_layer)
        self.mha = PhysicsInformedAttention(config=self.config, n_layer=self.n_layer)

        self.mlp = MLP(config=self.config, n_layer=self.n_layer)

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
                dtype=self.config.dtype,
            )
        for layer in self.layers:
            x = layer(x, mask=encoder_mask, n_iter=n_iter, kv_cache=kv_cache)
            rng_key = self.make_rng("dropout")
            x_first = x[:, 0, :][:, None, :]
            x_flip_z2 = jnp.concat([x_first, jnp.flip(x[:, 1:, :], axis=1)], axis=1)
            use_single = jrnd.uniform(rng_key) > 0.5
            x = jnp.where(use_single, x, x_flip_z2)

        return x, kv_cache

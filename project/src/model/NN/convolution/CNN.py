from typing import List

import jax
import netket.nn
from flax import linen as nn
from flax import struct
from jax import nn as jnn
from jax import numpy as jnp


class CNNConfig:
    lenght: int = 10
    n_state: int = 2
    state_feature: int = 8
    convs: List = [(32, 5), (64, 4), (128, 3)]
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float64
    default_kernel_init: jnn.initializers.Initializer = jnn.initializers.normal(
        1e-1, dtype=jnp.float64
    )
    default_bias_init: jnn.initializers.Initializer = jnn.initializers.normal(
        1e-4, dtype=jnp.float64
    )
    symm: bool = True


class Embed(nn.Module):
    config: CNNConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.config

        x = (x > 0).astype(jnp.int32)

        state_embeddings = nn.Embed(
            num_embeddings=cfg.n_state, features=cfg.state_feature, dtype=cfg.dtype
        )(x)

        return state_embeddings


class CNN(nn.Module):
    config: CNNConfig

    def setup(self):
        self.embed = Embed(self.config)
        self.convs = [
            nn.Conv(
                features=conv_feature,
                padding=0,
                kernel_size=ker,
                use_bias=self.config.use_bias,
                dtype=self.config.dtype,
            )
            for (conv_feature, ker) in self.config.convs
        ]
        self.norms = [nn.LayerNorm() for _ in range(self.config.lenght)]
        self.act = nn.elu
        self.dense = nn.Dense(
            features=1, use_bias=self.config.use_bias, dtype=self.config.dtype
        )

    def __call__(self, x):
        x = self.embed(x)

        for i, conv in enumerate(self.convs):
            x = self.norms[i](x)
            if self.config.symm:
                x = (conv(x) + conv(x[:, :, ::-1])) / 2.0
            else:
                x = conv(x)
            x = self.act(x)

        if self.config.symm:
            x = (self.dense(x) + self.dense(x[:, :, ::-1])) / 2.0
        else:
            x = self.dense(x)
        # x = netket.nn.log_cosh(x)

        return x.squeeze()


if __name__ == "__main__":
    n = 10
    cnn = CNN(CNNConfig())

    key = jax.random.PRNGKey(42)
    input = jnp.ones(16 * n)
    input = input.at[jax.random.randint(key, (16, n), 0, 16 * n)].set(-1)
    input = input.reshape(16, n)
    init_rngs = {"params": key, "dropout": key}
    params = cnn.init(init_rngs, input)
    print(input)
    print(cnn.apply(params, input, rngs={"dropout": key}))

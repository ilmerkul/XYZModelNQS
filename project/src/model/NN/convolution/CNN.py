import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any

import jax
import netket as nk
from flax import linen as nn
from jax import nn as jnn
from jax import numpy as jnp

project_path = pathlib.Path(os.getcwd())

sys.path.append(project_path.as_posix())


from src.model.NN import NNConfig
from src.model.struct import ChainConfig
from src.utils import powers_of_two


@dataclass(frozen=True)
class CNNConfig(NNConfig):
    chain: ChainConfig
    use_bias: bool
    dtype: jnp.dtype
    symm: bool

    default_kernel_init: Any = field(init=False)
    default_bias_init: Any = field(init=False)
    n_state: int = field(init=False)
    state_feature: int = field(init=False)
    convs: Any = field(init=False)

    def __post_init__(self):
        default_kernel_init: jnn.initializers.Initializer = jnn.initializers.normal(
            1e-1, dtype=self.dtype
        )
        default_bias_init: jnn.initializers.Initializer = jnn.initializers.normal(
            1e-4, dtype=self.dtype
        )
        n_state = int(2 * self.chain.spin + 1)
        state_feature = int(self.chain.n)
        powers = powers_of_two(self.chain.n)
        powers = list(map(lambda x: x + 1, powers))
        powers[-1] -= 1
        fs = jnp.linspace(3, self.chain.n**2, len(powers), dtype=jnp.int32).tolist()
        convs = [(f, p) for f, p in zip(fs, powers)]

        object.__setattr__(self, "default_kernel_init", default_kernel_init)
        object.__setattr__(self, "default_bias_init", default_bias_init)
        object.__setattr__(self, "n_state", n_state)
        object.__setattr__(self, "state_feature", state_feature)
        object.__setattr__(self, "convs", convs)


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
        self.norms = [nn.LayerNorm() for _ in range(self.config.chain.n)]
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
        x = nk.nn.activation.log_cosh(x)

        return x.squeeze()


if __name__ == "__main__":
    chain_cfg = ChainConfig(
        n=12,
        j=1,
        h=0.0,
        lam=1,
        gamma=0,
        spin=1 / 2,
    )
    cnn_config = CNNConfig(chain=chain_cfg, dtype=jnp.float64, symm=True, use_bias=True)
    cnn = CNN(cnn_config)

    key = jax.random.PRNGKey(42)
    input = jnp.ones(16 * chain_cfg.n)
    input = input.at[
        jax.random.randint(key, (16, chain_cfg.n), 0, 16 * chain_cfg.n)
    ].set(-1)
    input = input.reshape(16, chain_cfg.n)
    init_rngs = {"params": key, "dropout": key}
    params = cnn.init(init_rngs, input)
    print(input)
    print(cnn.apply(params, input, rngs={"dropout": key}))

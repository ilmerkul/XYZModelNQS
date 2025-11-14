from dataclasses import dataclass, field
from typing import Any

import flax.linen as nn
import netket as nk
from jax import nn as jnn
from jax import numpy as jnp
from src.model.NN import NNConfig
from src.model.struct import ChainConfig


@dataclass(frozen=True)
class FFConfig(NNConfig):
    chain: ChainConfig
    dtype: jnp.dtype
    use_bias: bool

    default_kernel_init: Any = field(init=False)
    default_bias_init: Any = field(init=False)
    alpha: int = field(init=False)

    def __post_init__(self):
        default_kernel_init: jnn.initializers.Initializer = jnn.initializers.normal(
            1e-1, dtype=self.dtype
        )
        default_bias_init: jnn.initializers.Initializer = jnn.initializers.normal(
            1e-4, dtype=self.dtype
        )
        alpha = int(self.chain.n**0.5)
        object.__setattr__(self, "default_kernel_init", default_kernel_init)
        object.__setattr__(self, "default_bias_init", default_bias_init)
        object.__setattr__(self, "alpha", alpha)


class FF(nn.Module):
    config: FFConfig

    def setup(self):
        self.dense = nn.Dense(
            features=self.config.chain.n * self.config.alpha,
            dtype=self.config.dtype,
            use_bias=self.config.use_bias,
            kernel_init=self.config.default_kernel_init,
            bias_init=self.config.default_bias_init,
        )

    @nn.compact
    def __call__(self, x):
        part1 = self.dense(x)
        part1 = nk.nn.activation.log_cosh(part1)
        part1 = jnp.sum(part1, axis=-1)

        part2 = self.dense(jnp.flip(x, axis=-1))
        part2 = nk.nn.activation.log_cosh(part2)
        part2 = jnp.sum(part2, axis=-1)

        output = (part1 + part2) / 2.0

        return output

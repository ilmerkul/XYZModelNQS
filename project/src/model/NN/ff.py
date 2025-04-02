from typing import Any

import flax.linen as nn
import netket
from jax import nn as jnn
from jax import numpy as jnp

default_kernel_init = jnn.initializers.normal(1e-1)
default_bias_init = jnn.initializers.normal(1e-4)


class FF(nn.Module):
    n: int
    alpha: int = 4
    dtype: Any = jnp.complex64
    precision: Any = None
    kernel_init: Any = default_kernel_init
    hidden_bias_init: Any = default_bias_init

    def _get_dense(self, n):
        return nn.Dense(
            features=n,
            dtype=self.dtype,
            use_bias=True,
            precision=self.precision,
            kernel_init=self.kernel_init,
            bias_init=self.hidden_bias_init,
        )

    def setup(self):
        self.dense = self._get_dense(self.n * self.alpha)

    @nn.compact
    def __call__(self, x):
        part1 = self.dense(x)
        part1 = netket.nn.log_cosh(part1)
        part1 = jnp.sum(part1, axis=-1)

        part2 = self.dense(jnp.flip(x, axis=-1))
        part2 = netket.nn.log_cosh(part2)
        part2 = jnp.sum(part2, axis=-1)

        output = (part1 + part2) / 2.0

        return output

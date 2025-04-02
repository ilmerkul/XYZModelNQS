from typing import Any

from jax import nn as jnn
from jax import numpy as jnp
from netket import nn

default_kernel_init = jnn.initializers.normal(1e-1)
default_bias_init = jnn.initializers.normal(1e-4)


class NN(nn.Module):
    n: int
    alpha: 4
    dtype: Any = jnp.complex64
    precision: Any = None
    kernel_init: Any = default_kernel_init
    hidden_bias_init: Any = default_bias_init

    def _get_dense(self, n, name):
        return nn.Dense(
            features=n,
            dtype=self.dtype,
            use_bias=True,
            precision=self.precision,
            kernel_init=self.kernel_init,
            bias_init=self.hidden_bias_init,
            name=name,
        )

    def setup(self):
        self.base = self._get_dense(self.n * self.alpha, name="base")
        self.head = self._get_dense(max([5, int(self.n / self.alpha)]), name="head")

    @nn.compact
    def __call__(self, x):
        part1 = self.base(x)
        part1 = nn.relu(part1)

        part2 = self.base(jnp.flip(x, axis=-1))
        part2 = nn.relu(part2)

        combined = part1 + part2
        result = self.head(combined)
        result = nn.log_cosh(result)
        result = jnp.sum(result, axis=-1)

        return result

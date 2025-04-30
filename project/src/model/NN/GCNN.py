from typing import Any

import flax.linen as nn
import netket
from jax import nn as jnn
from jax import numpy as jnp
from netket.utils.group import PermutationGroup

default_kernel_init = jnn.initializers.normal(1e-1)
default_bias_init = jnn.initializers.normal(1e-4)


class GCNN(nn.Module):
    n: int
    automorphisms: PermutationGroup
    alpha: int = 4
    dtype: Any = jnp.complex64
    precision: Any = None
    kernel_init: Any = default_kernel_init
    hidden_bias_init: Any = default_bias_init

    def _get_dense(self):
        return netket.models.GCNN(
            symmetries=self.automorphisms,
            features=(4, 2, 1),
            name="tr_gcnn",
            layers=3,
            kernel_init=self.kernel_init,
            bias_init=self.hidden_bias_init,
        )

    def setup(self):
        self.dense = self._get_dense()

    @nn.compact
    def __call__(self, x):
        x = self.dense(x)

        return x

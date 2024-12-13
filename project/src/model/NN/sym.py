from typing import Any

from jax import numpy as jnp, nn as jnn
import flax.linen as nn
from netket.utils.group import PermutationGroup
import netket

default_kernel_init = jnn.initializers.normal(1e-1)
default_bias_init = jnn.initializers.normal(1e-4)


class SymmModel(nn.Module):
    automorphisms: PermutationGroup
    alpha: int = 4
    kernel_init: Any = default_kernel_init
    hidden_bias_init: Any = default_bias_init

    def setup(self):
        self.dense1 = netket.nn.DenseSymm(symmetries=self.automorphisms,
                                          features=self.alpha,
                                          kernel_init=self.kernel_init,
                                          bias_init=self.hidden_bias_init)

    @nn.compact
    def __call__(self, x):
        x = x.reshape(-1, 1, x.shape[-1])

        x = self.dense1(x)
        x = jnp.sum(x, axis=(-1, -2))
        x = netket.nn.log_cosh(x)

        return x

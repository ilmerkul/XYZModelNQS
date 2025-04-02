from typing import Any

import flax.linen as nn
import netket
from jax import nn as jnn
from jax import numpy as jnp
from netket.utils.group import PermutationGroup

default_kernel_init = jnn.initializers.normal(1e-1)
default_bias_init = jnn.initializers.normal(1e-4)


class SymmModel(nn.Module):
    automorphisms: PermutationGroup
    alpha: int = 4
    kernel_init: Any = default_kernel_init
    hidden_bias_init: Any = default_bias_init

    def setup(self):
        self.dense = netket.nn.DenseSymm(symmetries=self.automorphisms,
                                         features=self.alpha,
                                         kernel_init=self.kernel_init,
                                         bias_init=self.hidden_bias_init)
        self.dense1 = nn.Dense(features=1,
                               kernel_init=self.kernel_init,
                               bias_init=self.hidden_bias_init)

    @nn.compact
    def __call__(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 1, x.shape[-1])

        x = self.dense(x)
        x = nn.relu(x)
        x = self.dense1(x)
        x = jnp.sum(x, axis=(-1, -2))

        return x

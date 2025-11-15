from dataclasses import dataclass, field
from typing import Any

import flax.linen as nn
import netket as nk
from jax import nn as jnn
from jax import numpy as jnp
from netket.utils.group import PermutationGroup
from ..interface import NNConfig
from src.model.struct import ChainConfig


@dataclass(frozen=True)
class FFSymmConfig(NNConfig):
    chain: ChainConfig
    dtype: jnp.dtype
    use_bias: bool

    default_kernel_init: Any = field(init=False)
    default_bias_init: Any = field(init=False)
    alpha: int = field(init=False)
    automorphisms: Any = field(init=False)

    def __post_init__(self):
        automorphisms: PermutationGroup = nk.graph.Chain(
            length=self.chain.n, pbc=self.chain.pbc
        ).automorphisms()

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
        object.__setattr__(self, "automorphisms", automorphisms)


class SymmModel(nn.Module):
    config: FFSymmConfig

    def setup(self):
        self.dense = nk.nn.DenseSymm(
            symmetries=self.config.automorphisms,
            features=self.config.alpha,
            kernel_init=self.config.default_kernel_init,
            bias_init=self.config.default_bias_init,
            param_dtype=self.config.dtype,
        )
        self.dense1 = nn.Dense(
            features=1,
            use_bias=self.config.use_bias,
            kernel_init=self.config.default_kernel_init,
            bias_init=self.config.default_bias_init,
            dtype=self.config.dtype,
        )

    @nn.compact
    def __call__(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 1, x.shape[-1])

        x = self.dense(x)
        x = nn.relu(x)
        x = self.dense1(x)
        x = jnp.sum(x, axis=(-1, -2))

        return x

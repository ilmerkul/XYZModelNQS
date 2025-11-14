from dataclasses import dataclass, field
from typing import Any

import flax.linen as nn
import netket as nk
from jax import nn as jnn
from jax import numpy as jnp
from netket.utils.group import PermutationGroup
from src.model.NN import NNConfig
from src.model.struct import ChainConfig
from src.utils import powers_of_two


@dataclass(frozen=True)
class GCNNConfig(NNConfig):
    chain: ChainConfig
    dtype: jnp.dtype

    automorphisms: Any = field(init=False)
    convs: Any = field(init=False)

    def __post_init__(self):
        automorphisms: PermutationGroup = nk.graph.Chain(
            length=self.chain.n, pbc=self.chain.pbc
        ).automorphisms()

        powers = powers_of_two(self.chain.n)
        convs = jnp.linspace(
            1, self.chain.n**2, len(powers), dtype=jnp.int32
        ).tolist()[::-1]

        object.__setattr__(self, "automorphisms", automorphisms)
        object.__setattr__(self, "convs", convs)


class GCNN(nn.Module):
    config: GCNNConfig

    def setup(self):
        self.dense = nk.models.GCNN(
            symmetries=self.config.automorphisms,
            features=self.config.convs,
            name="gcnn",
            layers=len(self.config.convs),
            kernel_init=(jnn.initializers.normal(1e-1, dtype=self.config.dtype)),
            bias_init=jnn.initializers.constant(0.0, dtype=self.config.dtype),
            param_dtype=self.config.dtype,
            complex_output=False,
        )

    @nn.compact
    def __call__(self, x):
        return self.dense(x)

import flax.linen as nn
import jax
from jax import lax
from jax import numpy as jnp

from ..feedforward.sym import SymmModel
from ..graph.GCNN import GCNN
from ..quantum.PQC import PQC
from .transformer import Transformer, TransformerConfig


class PhaseTransformer(nn.Module):
    config: TransformerConfig

    def setup(self):
        self.transformer = Transformer(self.config)
        self.gccn = GCNN(self.config.length, self.config.automorphisms)
        self.pqc = PQC(self.config.hilbert, self.config.length, 2)
        self.symm_model = SymmModel(self.config.automorphisms)

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        generate: bool = False,
        n_chains: int = 16,
        key: jax.Array = None,
    ):

        σ, log_prob = self.transformer(x, generate=generate, n_chains=n_chains, key=key)

        if not generate:
            # g = self.gccn(x)
            # g = self.symm_model(x)

            # phi = self.pqc(x)

            return 0.5 * log_prob  # + 1j * jnp.pi * lax.tanh(g) #+ phi

        return σ, log_prob

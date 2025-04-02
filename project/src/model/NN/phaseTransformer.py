import flax.linen as nn
from jax import lax

from .sym import SymmModel
from .transformer import Transformer, TransformerConfig
from .PQC import PQC


class PhaseTransformer(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x):
        p = Transformer(self.config)(x)

        g = SymmModel(self.config.automorphisms)(x)

        phi = PQC(self.config.hilbert, self.config.length, 2)(x)

        return p ** 0.5 * lax.complex(lax.cos(g), lax.sin(g)) * phi

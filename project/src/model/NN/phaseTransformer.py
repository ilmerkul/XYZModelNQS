import flax.linen as nn
import jax
from jax import lax

from .GCNN import GCNN
from .sym import SymmModel
from .transformer import Transformer, TransformerConfig
from .PQC import PQC


class PhaseTransformer(nn.Module):
    config: TransformerConfig

    def setup(self):
        self.transformer = Transformer(self.config)
        self.gccn = GCNN(self.config.length, self.config.automorphisms)
        self.pqc = PQC(self.config.hilbert, self.config.length, 2)
        self.symm_model = SymmModel(self.config.automorphisms)

    @nn.compact
    def __call__(self, x: jax.Array, generate: bool = False,
                 n_chains: int = 16):
        p = self.transformer(x, generate=generate, n_chains=n_chains)

        if not generate:
            print(x.shape)

            #g = self.gccn(x)
            g = self.symm_model(x)

            phi = self.pqc(x)

            return p ** 0.5 * lax.complex(lax.cos(g), lax.sin(g)) * phi

        return p

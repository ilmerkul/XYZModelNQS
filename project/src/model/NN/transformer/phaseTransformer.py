from dataclasses import dataclass, field
from typing import Any

import flax.linen as nn
import jax
import netket as nk
from jax import numpy as jnp
from netket.utils.group import PermutationGroup

from src.model.NN import NNConfig

from ..graph.GCNN import GCNN, GCNNConfig
from ..quantum.PQC import PQC
from .transformer import Transformer, TransformerConfig


@dataclass(frozen=True)
class PhaseTransformerConfig(NNConfig):
    tr_config: TransformerConfig
    pqc: bool
    gcnn: bool
    phase_train: bool = False
    gcnn_config: GCNNConfig = None

    hilbert: Any = field(init=False)
    training: int = field(init=False)

    def __post_init__(self):
        hilbert: nk.hilbert.Spin = nk.hilbert.Spin(
            N=self.tr_config.chain.n, s=self.tr_config.chain.spin
        )
        training = self.tr_config.training

        object.__setattr__(self, "hilbert", hilbert)
        object.__setattr__(self, "training", training)


class PhaseTransformer(nn.Module):
    config: PhaseTransformerConfig

    def setup(self):
        self.transformer = Transformer(self.config.tr_config)
        if self.config.gcnn:
            self.gcnn = GCNN(
                self.config.tr_config.chain.n,
                self.config.automorphisms,
                dtype=self.config.tr_config.dtype,
            )
        if self.config.pqc:
            self.pqc = PQC(
                self.config.hilbert,
                self.config.tr_config.chain.n,
                2,
                dtype=self.config.tr_config.dtype,
            )

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        generate: bool = False,
        n_chains: int = 16,
    ):
        if not generate:
            log_prob = self.transformer(x, generate=generate, n_chains=n_chains)
            res = 0.5 * log_prob
            if self.config.pqc and self.config.phase_train:
                res += self.pqc(x)

            if self.config.gcnn and self.config.phase_train:
                res += 1j * jnp.pi * nn.tanh(self.gcnn(x))

            return res
        else:
            σ, log_prob = self.transformer(x, generate=generate, n_chains=n_chains)

            return σ, log_prob

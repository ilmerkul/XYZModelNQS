from dataclasses import dataclass, field
from typing import Any

import flax.linen as nn
import jax
import netket as nk
from jax import numpy as jnp

from ..graph.GCNN import GCNN, GCNNConfig
from ..quantum.PQC import PQC
from .transformer import Transformer, TransformerConfig


@dataclass(frozen=True)
class PhaseTransformerConfig(TransformerConfig, GCNNConfig):
    pqc: bool
    gcnn: bool
    phase_train: bool
    phase_fine_ratio: float

    hilbert: Any = field(init=False)

    def __post_init__(self):
        super(self, TransformerConfig).__post_init__()
        super(self, GCNNConfig).__post_init__()

        hilbert: nk.hilbert.Spin = nk.hilbert.Spin(
            N=self.chain.n, s=self.chain.spin
        )

        object.__setattr__(self, "hilbert", hilbert)


class PhaseTransformer(nn.Module):
    config: PhaseTransformerConfig

    def setup(self):
        self.transformer = Transformer(self.config)
        if self.config.gcnn:
            self.gcnn = GCNN(
                config=self.config,
            )
        if self.config.pqc:
            self.pqc = PQC(
                self.config.hilbert,
                self.config.chain.n,
                2,
                dtype=self.config.dtype,
            )

    @nn.compact
    def __call__(
        self,
        σ: jax.Array = None,
        generate: bool = None,
        generate_batch_dim: int = None,
    ):
        if not generate:
            log_psi = self.transformer(
                σ, generate=generate, generate_batch_dim=generate_batch_dim
            )
            if self.config.pqc and self.config.phase_train:
                log_psi += self.pqc(σ)

            if self.config.gcnn and self.config.phase_train:
                log_psi += 1j * jnp.pi * nn.tanh(self.gcnn(σ))

            return log_psi
        else:
            σ, log_prob = self.transformer(
                generate=generate, generate_batch_dim=generate_batch_dim
            )

            return σ, log_prob

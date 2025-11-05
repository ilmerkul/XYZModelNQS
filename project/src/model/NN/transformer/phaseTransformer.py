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
    dtype: Any = field(init=False)
    training: int = field(init=False)
    seed: int = field(init=False)

    def __post_init__(self):
        hilbert: nk.hilbert.Spin = nk.hilbert.Spin(
            N=self.tr_config.chain.n, s=self.tr_config.chain.spin
        )
        training = self.tr_config.training

        object.__setattr__(self, "dtype", self.tr_config.dtype)
        object.__setattr__(self, "seed", self.tr_config.seed)
        object.__setattr__(self, "hilbert", hilbert)
        object.__setattr__(self, "training", training)


class PhaseTransformer(nn.Module):
    config: PhaseTransformerConfig

    def setup(self):
        self.transformer = Transformer(self.config.tr_config)
        if self.config.gcnn:
            self.gcnn = GCNN(
                config=self.config.gcnn_config,
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

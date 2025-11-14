from dataclasses import dataclass, field
from typing import Any

from flax import linen as nn
from jax import nn as jnn
from jax import numpy as jnp
from src.model.NN import NNConfig
from src.model.struct import ChainConfig


@dataclass(frozen=True)
class PosEmbType:
    ROTARY: str = "rope"
    LEARN_ABSOLUTE: str = "labs"
    ABSOLUTE: str = "abs"
    ZERO: str = "zero"
    RELATIVE: str = "rel"


@dataclass(frozen=True)
class TransformerConfig(NNConfig):
    chain: ChainConfig
    autoregressive: bool
    use_bias: bool
    use_dropout: bool
    embed_concat: bool
    dropout_rate: float
    inverse_iter_rate: float
    pos_embed: str
    eps: float
    dtype: Any = field(default_factory=lambda: jnp.float32)

    default_kernel_init: Any = field(init=False)
    default_bias_init: Any = field(init=False)
    n_state: int = field(init=False)
    layers: int = field(init=False)
    features: int = field(init=False)
    mlp_dim_scale: int = field(init=False)
    num_heads: int = field(init=False)

    def __post_init__(self):
        default_kernel_init = nn.initializers.xavier_uniform(dtype=self.dtype)
        default_bias_init = jnn.initializers.constant(0.0, dtype=self.dtype)
        n_state = int(2 * self.chain.spin + 1)
        layers = int(self.chain.n**0.5)
        features = int(self.chain.n**0.5)
        if features % 2 != 0:
            features += 1
        mlp_dim_scale = int(self.chain.n**0.5)

        for i in range(layers, 0, -1):
            if features % i == 0:
                num_heads = i
                break

        object.__setattr__(self, "default_kernel_init", default_kernel_init)
        object.__setattr__(self, "default_bias_init", default_bias_init)
        object.__setattr__(self, "n_state", n_state)
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "mlp_dim_scale", mlp_dim_scale)
        object.__setattr__(self, "num_heads", num_heads)

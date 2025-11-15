from dataclasses import dataclass

import jax.numpy as jnp




class SamplerType:
    TRANSFORMER: str = "transformer"
    METROPOLIS: str = "metropolis"


@dataclass
class SamplerConfig:
    type: str
    n_chains: int
    machine_pow: int
    dtype: jnp.dtype

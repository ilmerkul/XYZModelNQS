from jax import numpy as jnp


class KVCache:
    def __init__(
        self,
        batch_size: int,
        n_layers: int,
        length: int,
        features: int,
        dtype: jnp.dtype,
    ):
        self.key: jnp.ndarray = jnp.zeros(
            (batch_size, n_layers, length, features), dtype=dtype
        )
        self.value: jnp.ndarray = jnp.zeros(
            (batch_size, n_layers, length, features), dtype=dtype
        )

from typing import Any

import jax
import netket as nk
from flax import linen as nn
from jax import nn as jnn
from jax import numpy as jnp


class PQC(nn.Module):
    hilbert: nk.hilbert.Spin
    N_q: int
    N_l: int
    dtype: jnp.dtype

    def setup(self):
        self.theta1 = self.param(
            "theta1",
            nn.initializers.uniform(2 * jnp.pi, dtype=self.dtype),
            (self.N_l, self.N_q, 3),
        )
        self.theta2 = self.param(
            "theta2",
            nn.initializers.uniform(2 * jnp.pi, dtype=self.dtype),
            (self.N_l, self.N_q, 3),
        )

        self.w1 = self.param(
            "w1", nn.initializers.normal(1e-4, dtype=self.dtype), (self.N_q,)
        )
        self.w2 = self.param(
            "w2", nn.initializers.normal(1e-4, dtype=self.dtype), (self.N_q,)
        )

    @nn.compact
    def __call__(self, x):
        s = (x > 0).astype(self.dtype)

        f1 = self._jax_quantum_circuit(s, self.theta1, self.w1)
        f2 = self._jax_quantum_circuit(s, self.theta2, self.w2)

        return jnp.log(jnn.sigmoid(f1)) + 1j * 2 * jnp.pi * jnn.sigmoid(f2)

    def _jax_quantum_circuit(self, states, theta, w):
        def process_single_state(state):
            angles = jnp.pi * state

            for layer in range(self.N_l):
                layer_angles = theta[layer]
                angles = angles + layer_angles[:, 0]
                angles = jnp.sin(angles) * layer_angles[:, 1]
                angles = angles + layer_angles[:, 2]

                if layer < self.N_l - 1:
                    angles = angles + 0.1 * jnp.roll(angles, 1)

            expectations = jnp.tanh(angles)
            return jnp.sum(expectations * w)

        return jax.vmap(process_single_state)(states)

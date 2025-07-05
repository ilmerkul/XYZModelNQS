from typing import Any

import jax
import netket as nk
import pennylane as qml
from flax import linen as nn
from jax import nn as jnn
from jax import numpy as jnp

default_kernel_init = jnn.initializers.normal(1e-1)
default_bias_init = jnn.initializers.normal(1e-4)


class PQC(nn.Module):
    hilbert: nk.hilbert.Spin
    N_q: int
    N_l: int
    kernel_init: Any = default_kernel_init
    hidden_bias_init: Any = default_bias_init

    def setup(self):
        self.q_x1 = self.param(
            "pqc_q_x1", nn.initializers.xavier_uniform(), (self.N_l + 1, self.N_q)
        )
        self.q_z1 = self.param(
            "pqc_q_z1", nn.initializers.xavier_uniform(), (self.N_l + 1, self.N_q)
        )
        self.c1 = nn.Embed(num_embeddings=1, name="pqc_embedding1", features=self.N_q)

        self.q_x2 = self.param(
            "pqc_q_x2", nn.initializers.xavier_uniform(), (self.N_l + 1, self.N_q)
        )
        self.q_z2 = self.param(
            "pqc_q_z2", nn.initializers.xavier_uniform(), (self.N_l + 1, self.N_q)
        )
        self.c2 = nn.Embed(num_embeddings=1, name="pqc_embedding2", features=self.N_q)

        self.circuit = self._make_circuit()

    @nn.compact
    def __call__(self, x):
        c1 = self.c1((jnp.zeros(shape=(1,), dtype=jnp.int32)))
        c2 = self.c2((jnp.zeros(shape=(1,), dtype=jnp.int32)))

        f1 = self._quantum_circuit(self.q_x1, self.q_z1, c1, x)
        f2 = self._quantum_circuit(self.q_x2, self.q_z2, c2, x)

        return f1 + 1j * f2

    def _make_circuit(self):
        dev = qml.device("default.qubit", wires=self.N_q)

        @jax.jit
        @qml.qnode(dev, interface="jax", diff_method="parameter-shift")
        def circuit(state, q_x, q_z):
            state = jnp.pi * state
            for wire in range(self.N_q):
                angle = jnp.take(state, wire)
                qml.RY(angle, wires=wire)

            for layer in range(self.N_l):
                for wire in range(self.N_q):
                    qml.RX(q_x[layer, wire], wires=wire)
                    qml.RZ(q_z[layer, wire], wires=wire)

                for wire in range(self.N_q - 1):
                    qml.CNOT(wires=[wire, wire + 1])

            for wire in range(self.N_q):
                qml.RX(q_x[-1, wire], wires=wire)
                qml.RZ(q_z[-1, wire], wires=wire)

            return [qml.expval(qml.PauliZ(wire)) for wire in range(self.N_q)]

        return circuit

    def _quantum_circuit(self, q_x, q_z, c, s):
        s = (s > 0).astype(jnp.int32)

        expectations = jnp.zeros((s.shape[0],))

        for i in range(s.shape[0]):
            r = jnp.array(self.circuit(s[i, :], q_x, q_z))
            r = jnp.sum(c * r)

            expectations.at[i].set(r)

        return expectations

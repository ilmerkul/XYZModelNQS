from netket.operator.spin import sigmax, sigmay, sigmaz, identity
import netket as nk
import jax
from flax import linen as nn
from typing import Any
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
        self.q_x1 = self.param('q_x1', nn.initializers.xavier_uniform(),
                               (self.N_l, self.N_q))
        self.q_z1 = self.param('q_z1', nn.initializers.xavier_uniform(),
                               (self.N_l, self.N_q))
        self.c1 = nn.Embed(num_embeddings=1,
                           features=self.N_q)

        self.q_x2 = self.param('q_x2', nn.initializers.xavier_uniform(),
                               (self.N_l, self.N_q))
        self.q_z2 = self.param('q_z2', nn.initializers.xavier_uniform(),
                               (self.N_l, self.N_q))
        self.c2 = nn.Embed(num_embeddings=1,
                           features=self.N_q)

        self.H = 1
        for _ in range(self.N_q):
            self.H = jnp.kron(self.H, self.hadamar())

    def pauli_z(self):
        return jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)

    def pauli_x(self):
        return jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)

    def pauli_y(self):
        return jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)

    def identity_matrix(self, n=1):
        return jnp.eye(2 ** n, dtype=jnp.complex128)

    def hadamar(self):
        return jnp.array([[1, 1], [1, -1]], dtype=jnp.complex128) / 2 ** 0.5

    def rx(self, t):
        return self.identity_matrix() * jnp.cos(t / 2.) + 1j * jnp.sin(
            t / 2.) * self.pauli_x()

    def rz(self, t):
        return self.identity_matrix() * jnp.cos(t / 2.) + 1j * jnp.sin(
            t / 2.) * self.pauli_z()

    def cnot(self, n, m):
        s = self.identity_matrix(n)

        s = jnp.kron(s, (self.identity_matrix() - self.pauli_z()))

        s = jnp.kron(s, self.identity_matrix(m - n - 1))

        s = jnp.kron(s, (self.identity_matrix() - self.pauli_x()))

        s = jnp.kron(s, self.identity_matrix(self.N_q - m - 1))

        cnot = (self.identity_matrix(self.N_q) + 1j * s) / 2 ** 0.2

        return cnot

    def get_operator_U(self, q_x, q_z):
        U = self.identity_matrix(self.N_q)

        for i in range(self.N_l):
            d = self.identity_matrix(self.N_q)
            r = 1
            for j in range(self.N_q):
                r = jnp.kron(r,
                             self.rz(q_z[i, j]) @ self.rx(q_x[i, j]))

            for n in range(self.N_q - 1):
                for m in range(n + 1, n + 2):
                    d = d @ self.cnot(n, m)
                    d = d @ r

            U = U @ d

        U = U @ self.H

        return U

    def f(self, q_x, q_z, c, s):

        U = self.get_operator_U(q_x, q_z)

        U_t = U.conjugate().transpose()

        out = 0
        batch_size = s.shape[0]
        ind = self.identity_matrix()
        s = (s > 0).astype(jnp.int32)
        s = ind[s]
        r = []
        for i in range(batch_size):
            c = 1
            for j in range(self.N_q):
                c = jnp.kron(c, s[i, j, :])
            r.append(c[None, :])
        s = jnp.concatenate(r, axis=0)
        for i in range(self.N_q):
            Z = self.identity_matrix(i)
            Z = jnp.kron(Z, self.pauli_z())
            Z = jnp.kron(Z, self.identity_matrix(self.N_q - i - 1))
            A = U_t @ (Z @ U)

            out += c[i] * jnp.einsum('nb,bn->b', s.transpose(),
                                     jnp.einsum('nn,bn->bn', A, s))

        return out

    @nn.compact
    def __call__(self, x):
        c1 = self.c1((jnp.zeros(shape=(1,), dtype=jnp.int32)))
        c2 = self.c2((jnp.zeros(shape=(1,), dtype=jnp.int32)))

        f1 = self.f(self.q_x1, self.q_z1, c1, x)
        f2 = self.f(self.q_x2, self.q_z2, c2, x)

        return jax.lax.exp(f1 + 1j * f2)

from typing import Dict

import jax.numpy
import netket as nk
from netket.operator.spin import sigmax, sigmay, sigmaz


def get_spin_operators(
    n: int, hilbert: nk.hilbert.Spin, dtype: jax.numpy.dtype = jax.numpy.complex128
) -> Dict[str, nk.operator.LocalOperator]:
    """
    Generate measurable operators
    """
    ops = {}

    for i in range(n):
        ops[f"s_{i}"] = sigmaz(hilbert, i, dtype=dtype)

    ops["xx"] = sigmax(hilbert, 0, dtype=dtype) * sigmax(hilbert, n - 1)
    ops["yy"] = sigmay(hilbert, 0, dtype=dtype) * sigmay(hilbert, n - 1)
    ops["zz"] = sigmaz(hilbert, 0, dtype=dtype) * sigmaz(hilbert, n - 1)
    ops["zz_mid"] = sigmaz(hilbert, 0, dtype=dtype) * sigmaz(hilbert, n // 2)

    return ops


def get_model_netket_op(
    n: int,
    j: float,
    h: float,
    lam: float,
    gamma: float,
    hilbert: nk.hilbert.Spin,
    dtype: jax.numpy.dtype = jax.numpy.complex128,
) -> nk.operator.LocalOperator:
    """
    Generate netket model Hamiltonian
    """

    ham = nk.operator.LocalOperator(hilbert, dtype=dtype)

    if j != 0:
        ham += sum(
            [j * sigmax(hilbert, i) * sigmax(hilbert, i + 1) for i in range(n - 1)]
        )

    if lam != 0:
        ham += sum(
            [lam * sigmay(hilbert, i) * sigmay(hilbert, i + 1) for i in range(n - 1)]
        )

    if gamma != 0:
        ham += sum(
            [gamma * sigmaz(hilbert, i) * sigmaz(hilbert, i + 1) for i in range(n - 1)]
        )

    if h != 0:
        ham += sum([h * sigmax(hilbert, i) for i in range(n)])

    return ham

from typing import Dict

import netket as nk
import numpy as np
from src.scripts.ops import sx, sy, sz


sx *= 0.5
sy *= 0.5
sz *= 0.5


def get_spin_operators(
    n: int, hilbert: nk.hilbert.Spin
) -> Dict[str, nk.operator.LocalOperator]:
    """Generate measurable operators
    """
    ops = {}

    for i in range(n):
        ops[f"s_{i}"] = nk.operator.LocalOperator(hilbert, sz, [i])

    ops["xx"] = nk.operator.LocalOperator(hilbert, np.kron(sx, sx).tolist(), [0, n - 1])
    ops["yy"] = nk.operator.LocalOperator(hilbert, np.kron(sy, sy).tolist(), [0, n - 1])
    ops["zz"] = nk.operator.LocalOperator(hilbert, np.kron(sz, sz).tolist(), [0, n - 1])
    ops["zz_mid"] = nk.operator.LocalOperator(
        hilbert, np.kron(sz, sz).tolist(), [0, n // 2]
    )

    return ops


def get_xx_netket_op(
    n: int, j: float, h: float, hilbert: nk.hilbert.Spin
) -> nk.operator.LocalOperator:
    """Generate netket XX-model Hamiltonian
    """
    ham = nk.operator.LocalOperator(hilbert, dtype=complex)

    for i in range(n - 1):
        k = i + 1
        ham += h * nk.operator.LocalOperator(hilbert, sz, [i])
        ham += -j * nk.operator.LocalOperator(hilbert, np.kron(sx, sx).tolist(), [i, k])
        ham += -j * nk.operator.LocalOperator(hilbert, np.kron(sy, sy).tolist(), [i, k])

    ham += h * nk.operator.LocalOperator(hilbert, sz, [n - 1])

    return ham


def get_xy_netket_op(
        n: int, lam: float, h: float, hilbert: nk.hilbert.Spin
) -> nk.operator.LocalOperator:
    """Generate netket XY-model Hamiltonian
    """
    ham = nk.operator.LocalOperator(hilbert, dtype=complex)

    for i in range(n - 1):
        k = i + 1
        ham += h * nk.operator.LocalOperator(hilbert, sz, [i])
        ham -= (1 + lam) * nk.operator.LocalOperator(hilbert, np.kron(sx, sx).tolist(), [i, k])
        ham -= (1 - lam) * nk.operator.LocalOperator(hilbert, np.kron(sy, sy).tolist(), [i, k])

    ham += h * nk.operator.LocalOperator(hilbert, sz, [n - 1])

    return ham

import jax.numpy as np
import netket as nk
from netket.operator.spin import sigmax, sigmay, sigmaz

from src.model.struct import ChainConfig


def get_model_netket_op(
    cfg: ChainConfig,
    hilbert: nk.hilbert.Spin,
    dtype: np.dtype = np.complex128,
) -> nk.operator.LocalOperator:
    """
    Generate netket model Hamiltonian
    """

    ham = nk.operator.LocalOperator(hilbert, dtype=dtype)

    if cfg.j != 0:
        if cfg.lam != -1:
            ham += sum(
                [
                    cfg.j * (1 + cfg.lam) * sigmax(hilbert, i) * sigmax(hilbert, i + 1)
                    for i in range(cfg.n - 1)
                ]
            )

        if cfg.lam != 1:
            ham += sum(
                [
                    cfg.j * (1 - cfg.lam) * sigmay(hilbert, i) * sigmay(hilbert, i + 1)
                    for i in range(cfg.n - 1)
                ]
            )

        if cfg.gamma != 0:
            ham += sum(
                [
                    cfg.j * cfg.gamma * sigmaz(hilbert, i) * sigmaz(hilbert, i + 1)
                    for i in range(cfg.n - 1)
                ]
            )

    if cfg.h != 0:
        ham += sum([cfg.h * sigmaz(hilbert, i) for i in range(cfg.n)])

    return ham

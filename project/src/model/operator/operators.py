import jax.numpy as jnp
import netket as nk
from netket.operator.spin import sigmax, sigmay, sigmaz
from src.model.struct import ChainConfig


def get_model_netket_op(
    cfg: ChainConfig,
    hilbert: nk.hilbert.Spin,
    dtype: jnp.dtype = jnp.complex64,
) -> nk.operator.LocalOperator:
    """
    Generate netket model Hamiltonian
    """

    ham = nk.operator.LocalOperator(hilbert, dtype=dtype)

    if cfg.j != 0:
        if cfg.lam != -1:
            ham += sum(
                [
                    cfg.j * (1 + cfg.lam) * sigmax(hilbert, i) @ sigmax(hilbert, i + 1)
                    for i in range(cfg.n - 1)
                ]
            )

        if cfg.lam != 1:
            ham += sum(
                [
                    cfg.j * (1 - cfg.lam) * sigmay(hilbert, i) @ sigmay(hilbert, i + 1)
                    for i in range(cfg.n - 1)
                ]
            )

        if cfg.gamma != 0:
            ham += sum(
                [
                    cfg.j * cfg.gamma * sigmaz(hilbert, i) @ sigmaz(hilbert, i + 1)
                    for i in range(cfg.n - 1)
                ]
            )

    if cfg.h != 0:
        ham += sum([cfg.h * sigmaz(hilbert, i) for i in range(cfg.n)])

    return ham


def create_shift_invert_operator(ham, sigma, alpha=1.0):
    """
    Создает оператор (H - σI)⁻¹ используя разложение в подпространстве Крылова
    """

    class ShiftInvertOperator(nk.operator.AbstractOperator):
        def __init__(self, ham: nk.operator.LocalOperator, sigma: float, alpha):
            self._ham = ham
            self._sigma = sigma
            self._alpha = alpha
            super().__init__(ham.hilbert)

        @property
        def dtype(self):
            return self._ham.dtype

        def _apply(self, v):
            # Применяем (H - σI)⁻¹ к вектору v используя итерационный решатель
            # Для больших систем используем CG или GMRES
            from scipy.sparse.linalg import gmres

            def matvec(x):
                return self._ham @ x - self._sigma * x

            # Решаем (H - σI)x = v
            x, info = gmres(matvec, v, tol=1e-6, maxiter=100)
            if info != 0:
                raise RuntimeError(f"GMRES не сошелся: {info}")
            return self._alpha * x

    return ShiftInvertOperator(ham, sigma, alpha)


def create_polynomial_operator(
    ham: nk.operator.LocalOperator, sigma: float, power: int = 2
):
    identity = nk.operator.spin.identity(ham.hilbert)
    shifted_ham = ham - sigma * identity

    if power == 2:
        return shifted_ham @ shifted_ham
    else:
        result = shifted_ham
        for _ in range(power - 1):
            result = result @ shifted_ham
        return result


def create_shifted_operator(ham: nk.operator.LocalOperator, sigma: float):
    identity = nk.operator.spin.identity(ham.hilbert)
    shifted_ham = ham - sigma * identity

    return shifted_ham

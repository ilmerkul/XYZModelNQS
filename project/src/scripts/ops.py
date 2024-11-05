import numpy as np
from scipy import sparse

sx = np.array([[0, 1], [1, 0]], dtype=np.complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex)
sz = np.array([[1, 0], [0, -1]], dtype=np.complex)
identity = sparse.eye(2, dtype=np.complex)


def sigmaz_k(k, N) -> sparse.csr_matrix:
    """SZ op

    :param k:
    :param N:
    :rtype: sparse.csr_matrix
    """
    res = 1
    for i in range(N):
        if i == k:
            res = sparse.kron(res, sparse.csr_matrix(sz))
        else:
            res = sparse.kron(res, identity)

    return 0.5 * res


def sigmax_k(k, N) -> sparse.csr_matrix:
    """SX op

    :param k:
    :param N:
    :rtype: sparse.csr_matrix
    """
    res = 1
    for i in range(N):
        if i == k:
            res = sparse.kron(res, sparse.csr_matrix(sx))
        else:
            res = sparse.kron(res, identity)

    return 0.5 * res


def sigmay_k(k, N) -> sparse.csr_matrix:
    """SY op

    :param k:
    :param N:
    :rtype: sparse.csr_matrix
    """
    res = 1
    for i in range(N):
        if i == k:
            res = sparse.kron(res, sparse.csr_matrix(sy))
        else:
            res = sparse.kron(res, identity)

    return 0.5 * res


def ham_xx(n, j, h) -> sparse.csr_matrix:
    """XX model Hamiltonian

    :param n:
    :param j:
    :param h:
    :rtype: sparse.csr_matrix
    """
    res = sparse.csr_matrix((2 ** N, 2 ** N), dtype=np.complex)

    for i in range(N - 1):
        res += -j * (
            sigmax_k(i, N) * sigmax_k(i + 1, N) + sigmay_k(i, N) * sigmay_k(i + 1, N)
        )
        res += h * sigmaz_k(i, N)

    res += h * sigmaz_k(N - 1, N)

    return res

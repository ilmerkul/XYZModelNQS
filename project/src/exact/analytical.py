import numpy as np


def get_ellist(length: int, h: float, j: float) -> np.ndarray:
    res = []
    for l in range(length):
        res.append(-j * np.cos(np.pi * (l + 1) / (length + 1)) + h)

    return np.array(res)


def analytical_energy(length: int, h: float, j: float, l0: int) -> float:
    return -j * (
        (
            np.cos(np.pi * l0 / (2 * length + 2))
            * np.sin(np.pi * (l0 + 1) / (2 * length + 2))
        )
        / np.sin(np.pi / (2 * length + 2))
        - 1
    ) + h * (l0 - length / 2)


def analytical_s1z(length: int, l0: int) -> float:
    n_1 = length + 1

    res = -(
        np.cos(np.pi * l0 / n_1) * np.sin(np.pi * (l0 + 1) / n_1) / np.sin(np.pi / n_1)
    )
    res += l0 + 1
    res /= n_1
    res -= 0.5

    return res


def analytical_s1sn_z(length: int, l0: int) -> float:
    res = 0.0
    for l in range(l0):
        res += (-1) ** l * np.sin(np.pi * (l + 1) / (length + 1)) ** 2

    res *= 2 / (length + 1)

    return -(res * res) + analytical_s1z(length, l0) ** 2


def analytical_s1sn_xy(length: int, l0: int) -> float:
    res = 0.0
    for l in range(l0):
        res += (-1) ** l * np.sin(np.pi * (l + 1) / (length + 1)) ** 2

    res *= (-1) ** (l0 + 1) / (length + 1)

    return res


def analytical_m(length: int, l0: int) -> float:
    return (l0 - length / 2) / length

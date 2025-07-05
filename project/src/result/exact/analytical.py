import numpy as np


def get_ellist(length: int, h: float, j: float) -> np.ndarray:
    return np.array(
        [-j * np.cos(np.pi * (l + 1) / (length + 1)) + h for l in range(length)]
    )


def analytical_energy(length: int, h: float, j: float, l0: int) -> float:
    l2_2 = 2 * length + 2
    return -j * (
        np.cos(np.pi * l0 / l2_2)
        * np.sin(np.pi * (l0 + 1) / l2_2)
        / np.sin(np.pi / l2_2)
        - 1
    ) + h * (l0 - 0.5 * length)


def analytical_s1z(length: int, l0: int) -> float:
    l_1 = length + 1

    res = (
        -1
        * np.cos(np.pi * l0 / l_1)
        * np.sin(np.pi * (l0 + 1) / l_1)
        / np.sin(np.pi / l_1)
    )
    res += l0 + 1
    res /= l_1
    res -= 0.5

    return res


def analytical_s1sn_z(length: int, l0: int) -> float:
    res = sum(
        [(-1) ** l * np.sin(np.pi * (l + 1) / (length + 1)) ** 2 for l in range(l0)]
    )
    res *= 2 / (length + 1)

    return -1 * res**2 + analytical_s1z(length, l0) ** 2


def analytical_s1sn_xy(length: int, l0: int) -> float:
    res = sum(
        [(-1) ** l * np.sin(np.pi * (l + 1) / (length + 1)) ** 2 for l in range(l0)]
    )
    res *= (-1) ** (l0 + 1) / (length + 1)

    return res


def analytical_m(length: int, l0: int) -> float:
    return (l0 - 0.5 * length) / length

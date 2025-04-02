import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

base_prj_path = Path(__file__).parent.parent
sys.path.append(str(base_prj_path.absolute()))

from python.ExactAnalytics import get_ellist

from main import check_bad_params

plt.style.use("ggplot")
plt.rc("text", usetex=True)


def get_m(n: int, l0: int) -> float:
    return (l0 - n / 2) / n


def get_m_for_h(n: int, h: float) -> float:
    ellist = get_ellist(n, h, 1.0)
    l0 = np.sum(ellist < 0)

    return get_m(n, l0)


if __name__ == "__main__":
    n = int(sys.argv[1])

    h_list = np.linspace(0.1, 0.99, 1000)
    mask = []
    for h_ in h_list:
        mask.append(check_bad_params(n, 1.0, h_))

    h_list = h_list[np.array(mask)]

    m_list = np.array([get_m_for_h(n, h) for h in h_list])

    plots_prefix = base_prj_path.joinpath("plots")
    plots_prefix.mkdir(exist_ok=True, parents=True)

    f: plt.Figure = plt.figure(figsize=(8, 6))
    ax = f.subplots(2)
    ax[0].plot(h_list, m_list, ".", label=r"$f(h) = \frac{l_0}{n} - \frac{1}{2}$")
    ax[0].plot(h_list, 1 / np.pi * np.arccos(h_list) - 0.5, ".-", label=r"$f(h) = \frac{1}{\pi}\arccos(\frac{h}{J}) - \frac{1}{2}$")
    ax[0].legend()
    ax[0].set_xlabel("h")
    ax[0].set_ylabel("m")

    dx = np.diff(h_list)
    dy = np.diff(1 / np.pi * np.arccos(h_list) - 0.5)
    chi = -dy / dx
    ax[1].set_title(r"$\chi$")
    ax[1].plot(h_list[1:], chi, ".-", label=r"$\chi(h)$")
    ax[1].set_xlabel("h")
    ax[1].legend()
    ax[1].set_ylabel(r"$\chi$")

    fname = str(plots_prefix.joinpath(f"m_{n}.png").absolute())
    f.tight_layout()
    f.savefig(fname)

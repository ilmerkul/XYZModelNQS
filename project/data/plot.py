import os
import sys

import matplotlib.pyplot as plt
import netket as nk
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())

from src.model.nqs.operators import get_model_netket_op

if __name__ == "__main__":
    n = 10
    hilbert = nk.hilbert.Spin(N=n, s=1 / 2)

    params = {"n": n, "j": 1.0, "lam": 1.0, "gamma": 1.0, "hilbert": hilbert}
    h = -1 * np.linspace(0.0, 1.5, 100)
    e = []
    for i in range(h.shape[0]):
        H = get_model_netket_op(**params, h=h[i])
        E_gs = nk.exact.lanczos_ed(H, compute_eigenvectors=False)
        en = E_gs[0]
        print(h[i], en)
        e.append(en)

    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=False)
    fig.suptitle("Model j=-1, lam=0.5, gamma=1.0")

    data_path = f'report_n{params["n"]}_j{params["j"]}_h{0.001}_lam{params["lam"]}_gamma{params["gamma"]}.csv'
    data_path = os.path.join(os.getcwd(), "data", data_path)
    data = pd.read_csv(data_path)

    ax[0][0].scatter(h, e)
    ax[0][0].set(xlabel=r"$h$", ylabel=r"$\frac{E}{n}$")
    ax[0][0].scatter(data["h"], data["estimated_e"])

    ax[1][0].set(xlabel=r"$h$", ylabel=r"$\langle \hat{S}_i^x \hat{S}_{i+1}^x \rangle$")
    ax[1][0].scatter(data["h"], data["estimated_xx"])

    ax[1][1].set(xlabel=r"$h$", ylabel=r"$\langle \hat{S}_i^y \hat{S}_{i+1}^y \rangle$")
    ax[1][1].scatter(data["h"], data["estimated_yy"])

    ax[1][2].set(xlabel=r"$h$", ylabel=r"$\langle \hat{S}_i^z \hat{S}_{i+1}^z \rangle$")
    ax[1][2].scatter(data["h"], data["estimated_zz"])

    plt.show()

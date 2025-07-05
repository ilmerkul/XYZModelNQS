import sys
from pathlib import Path

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

base_prj_path = Path(__file__).parent.parent
sys.path.append(str(base_prj_path.absolute()))

from magnetisation_analytical import get_m_for_h
from main import check_bad_params
from python.XXModelImpl import exact_solution

plt.style.use("ggplot")


if __name__ == "__main__":
    j = 1.0
    N = int(sys.argv[1])

    exact = []
    analytical = []
    xx = []
    yy = []
    zz = []

    xx_var = []
    yy_var = []
    zz_var = []

    xx_a = []
    yy_a = []
    zz_a = []

    nqs = []
    hh = []

    z1_a = []
    zn_a = []
    z1 = []
    zn = []

    mu = []
    mu_a = []

    start_point = Path(__file__).parent.parent.joinpath(f"len_{N}")

    bad_points = np.linspace(0.5, 1.0, 10000)
    mask = np.zeros(bad_points.shape[0])

    for i, x in enumerate(bad_points):
        if not check_bad_params(N, j, x):
            mask[i] = 1

    bad_points = bad_points[mask == 1]
    bad_yy = np.zeros(bad_points.shape[0])
    bad_yy_zz = np.zeros(bad_points.shape[0])
    bad_yy_xx = np.zeros(bad_points.shape[0])
    bad_yy_yy = np.zeros(bad_points.shape[0])
    bad_yy_z1 = np.zeros(bad_points.shape[0])
    bad_yy_zn = np.zeros(bad_points.shape[0])

    for i, bp in enumerate(bad_points):
        ex = exact_solution(N, bp, j)
        bad_yy[i] = ex.report_df["energy"][0]
        bad_yy_xx[i] = ex.xx[0]
        bad_yy_yy[i] = ex.yy[0]
        bad_yy_zz[i] = ex.zz[0]
        bad_yy_z1[i] = ex.s1z[0]
        bad_yy_zn[i] = ex.snz[0]

    for subfolder in start_point.glob("*"):
        if not subfolder.joinpath("analytical").exists():
            continue
        mu_ = 0.0
        h = float(subfolder.name[2:])
        exact_sol = exact_solution(N, h, j)
        analytical.append(exact_sol.report_df["energy"][0])
        xx_a.append(exact_sol.xx[0])
        yy_a.append(exact_sol.yy[0])
        zz_a.append(exact_sol.zz[0])

        z1_a.append(exact_sol.s1z[0])
        zn_a.append(exact_sol.snz[0])

        str_path_csv = str(
            subfolder.joinpath("nqs_vmc").joinpath("main_report.csv").absolute()
        )
        df = pd.read_csv(str_path_csv)
        nqs.append(df.tail(1)["energy"].iloc[0])
        hh.append(h)

        pref = subfolder.joinpath("nqs_vmc")

        for i in range(N):
            mu_ += float(
                pref.joinpath(f"z_{i}.tsv").read_text().split("\n")[-1].split("\t")[1]
            )

        mu.append(mu_ / N)
        mu_a.append(get_m_for_h(N, h))

        xx.append(
            float(pref.joinpath("xx.tsv").read_text().split("\n")[-1].split("\t")[1])
        )
        xx_var.append(
            float(pref.joinpath("xxvar.tsv").read_text().split("\n")[-1].split("\t")[1])
        )
        yy_var.append(
            float(pref.joinpath("yyvar.tsv").read_text().split("\n")[-1].split("\t")[1])
        )
        zz_var.append(
            float(pref.joinpath("zzvar.tsv").read_text().split("\n")[-1].split("\t")[1])
        )
        yy.append(
            float(pref.joinpath("yy.tsv").read_text().split("\n")[-1].split("\t")[1])
        )
        zz.append(
            float(pref.joinpath("zz.tsv").read_text().split("\n")[-1].split("\t")[1])
        )
        z1.append(
            float(pref.joinpath("s1z.tsv").read_text().split("\n")[-1].split("\t")[1])
        )
        zn.append(
            float(pref.joinpath("snz.tsv").read_text().split("\n")[-1].split("\t")[1])
        )

    root = Path(__file__).parent.parent.joinpath("plots").joinpath(f"plots_{N}")
    root.mkdir(exist_ok=True)

    prefix = str(root.absolute()) + "/"

    f = plt.figure(figsize=(10, 8))
    ax = f.add_subplot()
    ax.plot(hh, analytical, "x", label="Analytical solution")
    ax.plot(hh, nqs, "^", label="NQS VMC approximation")
    ax.legend()
    ax.set_xlabel("h")
    ax.set_ylabel("E")
    ax.set_title(f"{N} spins, j=1.0")
    f.savefig(prefix + f"Energy_{N}.png", dpi=150)

    f = plt.figure(figsize=(10, 8))
    ax = f.add_subplot()
    ax.plot(hh, xx_a, "x", label="Analytical solution")
    ax.errorbar(x=hh, y=xx, yerr=xx_var, fmt="^", label="NQS VMC approximation")
    ax.legend()
    ax.set_xlabel("h")
    ax.set_ylabel("XX")
    ax.set_title(f"{N} spins, j=1.0")
    f.savefig(prefix + f"XX_{N}.png", dpi=150)

    f = plt.figure(figsize=(10, 8))
    ax = f.add_subplot()
    ax.plot(hh, yy_a, "x", label="Analytical solution")
    ax.errorbar(x=hh, y=yy, yerr=yy_var, fmt="^", label="NQS VMC approximation")
    ax.legend()
    ax.set_xlabel("h")
    ax.set_ylabel("YY")
    ax.set_title(f"{N} spins, j=1.0")
    f.savefig(prefix + f"YY_{N}.png", dpi=150)

    f = plt.figure(figsize=(10, 8))
    ax = f.add_subplot()
    ax.plot(hh, zz_a, "x", label="Analytical solution")
    ax.errorbar(x=hh, y=zz, yerr=zz_var, fmt="^", label="NQS VMC approximation")
    ax.legend()
    ax.set_xlabel("h")
    ax.set_ylabel("ZZ")
    ax.set_title(f"{N} spins, j=1.0")
    f.savefig(prefix + f"ZZ_{N}.png", dpi=150)

    f = plt.figure(figsize=(10, 8))
    ax = f.add_subplot()
    ax.plot(hh, zz_a, "x", label="Analytical solution")
    ax.plot(hh, zz, "^", label="NQS VMC approximation")
    ax.legend()
    ax.set_xlabel("h")
    ax.set_ylabel("ZZ")
    ax.set_title(f"{N} spins, j=1.0")
    f.savefig(prefix + f"ZZ_{N}_no_errbar.png", dpi=150)

    f = plt.figure(figsize=(10, 8))
    ax = f.add_subplot()
    ax.plot(hh, z1_a, "x", label="Analytical solution")
    ax.plot(hh, z1, "^", label="NQS VMC approximation")
    ax.legend()
    ax.set_xlabel("h")
    ax.set_ylabel("Z1")
    ax.set_title(f"{N} spins, j=1.0")
    f.savefig(prefix + f"Z1_{N}.png", dpi=150)

    f = plt.figure(figsize=(10, 8))
    ax = f.add_subplot()
    ax.plot(hh, zn_a, "x", label="Analytical solution")
    ax.plot(hh, zn, "^", label="NQS VMC approximation")
    ax.legend()
    ax.set_xlabel("h")
    ax.set_ylabel("ZN")
    ax.set_title(f"{N} spins, j=1.0")
    f.savefig(prefix + f"ZN_{N}.png", dpi=150)

    f = plt.figure(figsize=(10, 8))
    ax = f.add_subplot()
    ax.plot(hh, mu_a, "x", label="Analytical solution")
    ax.plot(hh, mu, "^", label="NQS VMC approximation")
    ax.legend()
    ax.set_xlabel("h")
    ax.set_ylabel("Mu")
    ax.set_title(f"{N} spins, j=1.0")
    f.savefig(prefix + f"Mu_{N}.png", dpi=150)

    f = plt.figure(figsize=(10, 8))
    ax = f.add_subplot()
    ax.plot(hh, mu_a, "x", label="Exact solution")
    ax.plot(hh, mu, "^", label="NQS VMC approximation")
    ax.legend()
    ax.set_xlabel("h")
    ax.set_ylabel("Mu")
    ax.set_title(f"{N} spins, j=1.0")
    f.savefig(prefix + f"Mu_{N}_excat.png", dpi=150)

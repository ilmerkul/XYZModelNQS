import sys
from pathlib import Path

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

base_prj_path = Path(__file__).parent.parent
sys.path.append(str(base_prj_path.absolute()))

from python.XXModelImpl import exact_solution

from main import check_bad_params

plt.style.use("ggplot")
plt.rc("text", usetex=True)


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
        xx.append(
            float(pref.joinpath("xx.tsv").read_text().split("\n")[-1].split("\t")[1])
        )
        xx_var.append(
            float(pref.joinpath("xxvar.tsc").read_text().split("\n")[-1].split("\t")[1])
        )
        yy_var.append(
            float(pref.joinpath("yyvar.tsc").read_text().split("\n")[-1].split("\t")[1])
        )
        zz_var.append(
            float(pref.joinpath("zzvar.tsc").read_text().split("\n")[-1].split("\t")[1])
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
    ax.plot(hh, analytical, "x", label="Exact solution")
    ax.plot(hh, nqs, "^", label="NQS VMC approximation")
    ax.legend()
    ax.set_xlabel("h")
    ax.set_ylabel("E")
    ax.set_title(f"{N} spins, j=1.0")
    f.savefig(prefix + f"Energy_{N}_l.png", dpi=150)

    f = plt.figure(figsize=(10, 8))
    ax = f.add_subplot()
    ax.plot(hh, analytical, "x", label="Exact solution")
    ax.legend()
    ax.set_xlabel("h")
    ax.set_ylabel("E")
    ax.set_title(f"{N} spins, j=1.0")
    f.savefig(prefix + f"Energy_{N}_exact_only.png", dpi=150)

    f = plt.figure(figsize=(10, 8))
    ax = f.add_subplot()
    ax.plot(hh, zz_a, "x", label="Exact solution")
    ax.plot(hh, zz, "^", label="NQS VMC approximation")
    ax.legend()
    ax.set_xlabel("h")
    ax.set_ylabel(r"$\sigma^z_1\sigma^z_N$")
    ax.set_title(f"{N} spins, j=1.0")
    f.savefig(prefix + f"ZZ_{N}_l.png", dpi=150)

    f = plt.figure(figsize=(10, 8))
    ax = f.add_subplot()
    ax.plot(hh, zz, "^", label="NQS VMC approximation")
    ax.legend()
    ax.set_xlabel("h")
    ax.set_ylabel(r"$\sigma^z_1\sigma^z_N$")
    ax.set_title(f"{N} spins, j=1.0")
    f.savefig(prefix + f"ZZ_{N}_nqs_only.png", dpi=150)

    f = plt.figure(figsize=(10, 8))
    ax = f.add_subplot()
    ax.plot(hh, zz_a, "x", label="Exact solution")
    ax.legend()
    ax.set_xlabel("h")
    ax.set_ylabel(r"$\sigma^z_1\sigma^z_N$")
    ax.set_title(f"{N} spins, j=1.0")
    f.savefig(prefix + f"ZZ_{N}_exact_only.png", dpi=150)

    f = plt.figure(figsize=(10, 8))
    ax = f.add_subplot()
    ax.plot(hh, z1_a, "x", label="Exact solution")
    ax.plot(hh, z1, "^", label="NQS VMC approximation")
    ax.legend()
    ax.set_xlabel("h")
    ax.set_ylabel("$\sigma^z_1$")
    ax.set_title(f"{N} spins, j=1.0")
    f.savefig(prefix + f"Z1_{N}_l.png", dpi=150)

    f = plt.figure(figsize=(10, 8))
    ax = f.add_subplot()
    ax.plot(hh, zn_a, "x", label="Exact solution")
    ax.plot(hh, zn, "^", label="NQS VMC approximation")
    ax.legend()
    ax.set_xlabel("h")
    ax.set_ylabel("$\sigma^z_N$")
    ax.set_title(f"{N} spins, j=1.0")
    f.savefig(prefix + f"ZN_{N}_l.png", dpi=150)

    f = plt.figure(figsize=(10, 8))
    ax = f.add_subplot()
    ax.plot(hh, z1_a, "x", label="Exact solution")
    ax.legend()
    ax.set_xlabel("h")
    ax.set_ylabel("$\sigma^z_1$")
    ax.set_title(f"{N} spins, j=1.0")
    f.savefig(prefix + f"Z1_{N}_exact_only.png", dpi=150)

    f = plt.figure(figsize=(10, 8))
    ax = f.add_subplot()
    ax.plot(hh, zn_a, "x", label="Exact solution")
    ax.legend()
    ax.set_xlabel("h")
    ax.set_ylabel("$\sigma^z_N$")
    ax.set_title(f"{N} spins, j=1.0")
    f.savefig(prefix + f"ZN_{N}_exact_only.png", dpi=150)

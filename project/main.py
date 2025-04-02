import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

import netket as nk
import numpy as np

from src.model.nqs import Model, get_spin_operators

prj_root = Path(__file__).parent.absolute()
if str(prj_root) not in sys.path:
    sys.path.append(str(prj_root))


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--len", required=True, type=int)
    parser.add_argument("--h", required=False, type=float)
    parser.add_argument("--j", required=False, type=float, default=1.0)
    parser.add_argument("--lam", required=False, type=float, default=0)
    parser.add_argument("--gamma", required=False, type=float, default=0)
    parser.add_argument("--path_data", required=False, type=str,
                        default="data")
    parser.add_argument("--train", required=False, type=str, default="default")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    logging.info(f"NetKet version: {nk.__version__}")

    args = get_args()

    n = args.len
    j = args.j
    h = args.h
    lam = args.lam
    gamma = args.gamma
    path_data = args.path_data
    train = args.train

    logging.info(f"Model with n={n}, j={j}, h={h}, lam={lam}, gamma={gamma}")

    report_file_name = f"report_n{n}_j{j}_h{h}_lam{lam}_gamma{gamma}.csv"
    report_file = prj_root.joinpath(path_data).joinpath(report_file_name)
    report_file.touch()

    if h is not None:
        model = Model(n=n, h=h, j=j, lam=lam, gamma=gamma)
        model.set_machine()

        model.set_ops(get_spin_operators(n, model.get_hilbert()))
        model.set_optimizer()
        model.set_sampler()

        exact = model.get_analytical()
        logging.info(f"Run for h={h:.4f}.\t"
                     f"Analytical energy: {exact.exact():.6f}")

        if train == "default":
            model.set_vmc()
            model.train()
        else:
            model.custom_train()

        res = model.get_results()

        with report_file.open("a") as file:
            file.writelines([res.row(), ])
        sys.exit(0)

    h = np.linspace(0.0, 1.5, 100)

    # First run
    for i in range(h.shape[0] - 1):
        last_h = -1
        with report_file.open("r") as file:
            lines = file.readlines()

            if len(lines) > 1:
                last = lines[-1]
                last_h = float(last.split(",")[1])

        if h[i] <= last_h:
            logging.info(f"skip {h[i]}")
            continue

        model = Model(n=n, h=h[i], j=j, lam=lam, gamma=gamma)
        model.set_machine()

        model.set_ops(get_spin_operators(n, model.get_hilbert()))
        model.set_optimizer()
        model.set_sampler()
        model.set_vmc()

        exact = model.get_analytical()
        logging.info(f"Run for h={h[i]:.4f}.\t"
                     f"Analytical energy: {exact.exact():.6f}")

        model.train()
        res = model.get_results()

        msg = f"Results estimated (analytical):\n\t" \
              f"Energy:\t{res.res['e']:.6f}({res.ares['e']:.6f})\n\t"
        msg += f"ZZ:\t{res.res['zz']:.6f}({res.ares['zz']:.6f})\n\t"
        msg += f"Magnetization:\t{res.res['m']:.6f}({res.ares['m']:.6f})\n"
        logging.info(msg)

        if i == 0:
            with report_file.open("a") as file:
                file.writelines([res.header(), ])

        with report_file.open("a") as file:
            file.writelines([res.row(), ])

    sys.exit(0)

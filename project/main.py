import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import netket as nk

prj_root = Path(__file__).parent.absolute()
if str(prj_root) not in sys.path:
    sys.path.append(str(prj_root))

from src.model import Model, get_spin_operators, get_xx_netket_op, get_xy_netket_op

if __name__ == "__main__":
    print(f"NetKet version: {nk.__version__}")
    parser = ArgumentParser()
    parser.add_argument("--length", required=True, type=int)
    parser.add_argument("--point", required=False, type=float)
    parser.add_argument("--model", required=False, type=str)
    parser.add_argument("--lam", required=False, type=float, default=1.0)
    args = parser.parse_args()
    n = args.length
    report_file = prj_root.joinpath("data").joinpath(f"report_{n}_{args.model or 'xx'}_lam_{args.lam}.csv")
    report_file.touch()

    j = 1.0

    if args.point is not None:
        h = args.point
        model = Model(n=n, type="xy" if args.model == "xy" else "xx")
        model.set_machine()
        model.set_meta_params(h, j)

        if args.model == "xy":
            print("Use XY op")
            lam = args.lam or 1.0
            op = get_xy_netket_op(n, lam, h, model.get_hilbert())
        else:
            op = get_xx_netket_op(n, j, h, model.get_hilbert())
            print("Use XX op")

        model.set_ham(op)
        model.set_ops(get_spin_operators(n, model.get_hilbert()))
        model.set_optimizer()
        model.set_sampler()
        model.set_vmc()

        exact = model.get_analytical()
        print(f"Run for h={h:.4f}.\tAnalytical energy: {exact.exact():.6f}")

        model.train()
        res = model.get_results()

        with report_file.open("a") as file:
            file.writelines(
                [
                    res.row(),
                ]
            )
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
            print(f"skip {h[i]}")
            continue

        if args.model == "xy":
            model = Model(n=n, type="xy")
        else:
            model = Model(n=n)

        model.set_machine()
        model.set_meta_params(h[i], j)

        if args.model == "xy":
            print("Use XY op")
            lam = args.lam or 1.0
            op = get_xy_netket_op(n, lam, h[i], model.get_hilbert())
        else:
            op = get_xx_netket_op(n, j, h[i], model.get_hilbert())
            print("Use XX op")

        model.set_ham(op)
        model.set_ops(get_spin_operators(n, model.get_hilbert()))
        model.set_optimizer()
        model.set_sampler()
        model.set_vmc()

        exact = model.get_analytical()
        print(f"Run for h={h[i]:.4f}.\tAnalytical energy: {exact.exact():.6f}")

        model.train()
        res = model.get_results()

        msg = f"Results estimated (analytical):\n\tEnergy:\t{res.res['e']:.6f}({res.ares['e']:.6f})\n\t"
        msg += f"ZZ:\t{res.res['zz']:.6f}({res.ares['zz']:.6f})\n\t"
        msg += f"Magnetization:\t{res.res['m']:.6f}({res.ares['m']:.6f})\n"
        print(msg)

        if i == 0:
            with report_file.open("a") as file:
                file.writelines(
                    [
                        res.header(),
                    ]
                )

        with report_file.open("a") as file:
            file.writelines(
                [
                    res.row(),
                ]
            )

    sys.exit(0)

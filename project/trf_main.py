import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import netket as nk
import numpy as np

prj_root = Path(__file__).parent.absolute()
if str(prj_root) not in sys.path:
    sys.path.append(str(prj_root))

from src.model.tl_experiment import Model

from src.model import get_spin_operators, get_xx_netket_op, get_xy_netket_op

if __name__ == "__main__":
    print(f"NetKet version: {nk.__version__}")
    parser = ArgumentParser()
    parser.add_argument("--length", required=True, type=int)
    parser.add_argument("--h_cons", required=True, type=float)
    parser.add_argument("--train_base", required=False, type=int, default=1)
    parser.add_argument("--from_scratch", required=False, type=int, default=0)
    parser.add_argument("--gamma", required=False, type=float, default=0.25)
    args = parser.parse_args()

    n = args.length
    h = args.h_cons
    j = 1.0

    prefix_path = prj_root.joinpath("data").joinpath(f"transfer_learning_{n}_{h}")
    prefix_path.mkdir(exist_ok=True, parents=True)

    report_file = prefix_path.joinpath(f"results_fs_{args.from_scratch}.csv")
    report_file.touch()

    if args.train_base == 1:
        xx = Model(n=n)
        xx.set_machine()
        xx.set_meta_params(h, j)

        op = get_xx_netket_op(n, j, h, xx.get_hilbert())
        xx.set_ham(op)
        xx.set_ops(get_spin_operators(n, xx.get_hilbert()))
        xx.set_optimizer()
        xx.set_sampler()
        xx.set_vmc()

        exact = nk.exact.lanczos_ed(op)[0]
        print(f"Run for h={h:.4f}.\tAnalytical energy: {exact:.6f}")
        print(f"Initial weights: {xx.print_base_mean()}")
        xx.train(4000)
        print(f"Final weights: {xx.print_base_mean()}")
        xx.save_base_weights(prefix_path)

    model = Model(n=n)

    if args.from_scratch:
        model.set_machine(base_weights=None)
    else:
        model.set_machine(base_weights=prefix_path)

    model.set_meta_params(h, j)
    model.set_ham(get_xy_netket_op(n, args.gamma, h, model.get_hilbert()))
    model.set_ops(get_spin_operators(n, model.get_hilbert()))
    model.set_optimizer(train_base=args.from_scratch)
    model.set_sampler()
    model.set_vmc()

    exact = nk.exact.lanczos_ed(model.ops["e"], compute_eigenvectors=False)
    print(f"Run for h={h:.4f}.\tAnalytical energy: {exact[0]:.6f}")

    print(f"Initial base weights: {model.print_base_mean()}")
    print(f"Initial head weights: {model.print_head_mean()}")
    model.train(n_iter=1000)
    print(f"Final base weights: {model.print_base_mean()}")
    print(f"Final head weights: {model.print_head_mean()}")
    res = model.get_results(is_xx=False)

    if not args.train_base:
        training_file = prefix_path.joinpath(
            f"training_logs_scratch_is_{args.from_scratch}.json"
        )
        exact_file = prefix_path.joinpath(
            "exact_sol.txt"
        )

        with exact_file.open("w") as file:
            file.write(f"{exact[0]:.6f}")

        with training_file.open(mode="w") as file:
            history = model.logger.data["Energy"].to_dict()
            data = {
                "iters": history["iters"].tolist(),
                "Mean": np.real(history["Mean"]).tolist(),
                "Variance": np.real(history["Variance"]).tolist(),
                "Sigma": np.real(history["Sigma"]).tolist(),
                "R_hat": np.real(history["R_hat"]).tolist(),
                "TauCorr": np.real(history["TauCorr"]).tolist(),
            }
            file.write(
                json.dumps(
                    data,
                    allow_nan=True,
                    indent=1,
                    ensure_ascii=False,
                )
            )

    with report_file.open(mode="w") as file:
        file.writelines(
            [
                res.row(),
            ]
        )
    sys.exit(0)

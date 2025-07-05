import logging
from pathlib import Path

import numpy as np
from src.model.nqs import ModelNQS, ModelNQSConfig

prj_root = Path(__file__).parent.parent.parent.absolute()


class App:
    def __init__(self, args):
        self.model_config = ModelNQSConfig(
            n=args.len,
            j=args.j,
            h=0.0 if args.h is None else args.h,
            lam=args.lam,
            gamma=args.gamma,
            sampler="transformer",
        )
        self.model = ModelNQS(cfg=self.model_config)

        self.path_data = args.path_data
        self.train = args.train

    def run(self):
        report_file_name = f"report_n{self.model_config.n}_j{self.model_config.j}_h{self.model_config.h}_lam{self.model_config.lam}_gamma{self.model_config.gamma}.csv"
        report_file = prj_root.joinpath(self.path_data).joinpath(report_file_name)
        report_file.touch()

        if self.model_config.h is None:
            h_rng = np.linspace(0.0, 1.5, 100)
        else:
            h_rng = np.array([self.model_config.h])

        for i, h in enumerate(h_rng):
            last_h = -1
            with report_file.open("r") as file:
                lines = file.readlines()

                if len(lines) > 1:
                    last = lines[-1]
                    last_h = float(last.split(",")[1])

            if h <= last_h:
                logging.info(f"skip {h}")
                continue

            self.model.set_h(h)

            exact = self.model.get_analytical()
            logging.info(
                f"Run for h={h:.4f}.\t" f"Analytical energy: {exact.exact():.6f}"
            )

            self.model.train()
            res = self.model.get_results()

            msg = (
                f"Results estimated (analytical):\n\t"
                f"Energy:\t{res.res['e']:.6f}({res.ares['e']:.6f})\n\t"
            )
            msg += f"ZZ:\t{res.res['zz']:.6f}({res.ares['zz']:.6f})\n\t"
            msg += f"Magnetization:\t{res.res['m']:.6f}({res.ares['m']:.6f})\n"
            logging.info(msg)

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

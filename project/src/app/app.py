import logging
from pathlib import Path

import numpy as np

from src.model.NN import NNType
from src.model.nqs import ModelNQS, ModelNQSConfig
from src.model.optimizer import NQSOptimizer
from src.model.sampler import SamplerType
from src.result.struct import Result
from src.model.struct import ChainConfig

prj_root = Path(__file__).parent.parent.parent.absolute()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class App:
    def __init__(self, args):
        self.args = args
        chain_cfg = ChainConfig(n=10 if args.len is None else args.len,
            j=-1 if args.j is None else args.j,
            h=0.0 if args.h is None else args.h,
            lam=0.5 if args.lam is None else args.lam,
            gamma=1 if args.gamma is None else args.gamma,)
        self.model_config = ModelNQSConfig(
            chain = chain_cfg,
            nn=NNType.PHASE_TRANSFORMER,
            optimizer=NQSOptimizer.ADAM_ZERO_PQC,
            n_iter=100,
            lr=5e-4,
            sampler=SamplerType.METROPOLIS,
            preconditioner=True,
        )
        self.model = ModelNQS(cfg=self.model_config)

        self.path_data = "data"
        if args.path_data is not None:
            self.path_data = args.path_data
        self.train = "default"
        if args.train is not None:
            self.train = args.train

    def run(self):
        report_file_name = f"report_n{self.model_config.chain.n}_j{self.model_config.chain.j}_h{self.model_config.chain.h}_lam{self.model_config.chain.lam}_gamma{self.model_config.chain.gamma}.csv"
        report_file = prj_root.joinpath(self.path_data).joinpath(report_file_name)
        report_file.touch()

        if self.args.h is None:
            h_min = 0.0 if self.args.h_min is None else self.args.h_min
            h_max = 1.5 if self.args.h_max is None else self.args.h_max
            h_n = 100 if self.args.h_n is None else self.args.h_n
            h_rng = np.linspace(h_min, h_max, h_n)
        else:
            h_rng = np.array([self.model_config.h])

        with report_file.open("r+") as file:
            lines = file.readlines()

            if len(lines) == 0:
                file.writelines([Result.header()])

        file_h = set()
        with report_file.open("r") as file:
            lines = file.readlines()

            if len(lines) > 1:
                file_h = set(map(lambda line: float(line.split(",")[1]), lines[1:]))

        for h in h_rng:
            if h in file_h:
                logging.info(f"skip {h}")
                continue

            self.model.set_h(h)

            # exact = self.model.get_analytical()
            logging.info(f"Run for h={h:.4f}.\t")

            self.model.train()
            res: Result = self.model.get_result()

            msg = "Results:\n"
            for res_k, res_v in res.res.items():
                msg += f"{res_k}: {res_v}\n"
            logging.info(msg)

            with report_file.open("a") as file:
                file.writelines([res.row()])

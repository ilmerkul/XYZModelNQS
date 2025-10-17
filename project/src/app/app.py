import logging
import pprint

import jax
import numpy as np

from src.app.config import get_ff_config
from src.model.nqs import ModelNQS, ModelNQSConfig
from src.result.struct import Result
from src.utils import report_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)

jax.config.update("jax_platform_name", "gpu")

print("JAX devices:", jax.devices())
print("Default device:", jax.default_device)


class App:
    path_data: str = "data/report"
    save_model_path: str = "data/model"
    train: str = "default"

    def __init__(self, args):
        self.args = args
        self.model_config: ModelNQSConfig = get_ff_config(
            args=args, save_model_path=self.save_model_path
        )
        pprint.pprint(self.model_config)

        self.model = ModelNQS(cfg=self.model_config)

        if args.path_data is not None:
            self.path_data = args.path_data

        if args.train is not None:
            self.train = args.train

    def run(self):
        report_file = report_path(self.model_config.chain, self.path_data)
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

            logging.info(f"Run for h={h:.4f}.\t")

            self.model.train()
            res: Result = self.model.get_result()

            msg = "Results:\n"
            for res_k, res_v in res.res.items():
                msg += f"{res_k}: {res_v}\n"
            logging.info(msg)

            with report_file.open("a") as file:
                file.writelines([res.row()])

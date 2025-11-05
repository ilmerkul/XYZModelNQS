import logging
import os

import jax
import numpy as np
from src.app.config import dict2class_config
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
    def __init__(self, cfg):
        self.path_data = cfg["path_data"]
        self.train = cfg["train"]
        self.h = cfg["h"]
        self.model_config = dict2class_config(cfg=cfg)
        self.logger = logging.getLogger(__name__)
        self.logger.info(self.model_config)

        self.model = ModelNQS(cfg=self.model_config)

    def run(self):
        report_file = report_path(self.model_config.chain, self.path_data)
        os.makedirs(self.path_data, exist_ok=True)
        report_file.touch()

        h_rng = np.linspace(*self.h)

        with report_file.open("r+") as file:
            lines = file.readlines()

            if len(lines) == 0:
                file.writelines([Result.header()])

        file_h = set()
        with report_file.open("r") as file:
            lines = file.readlines()

            if len(lines) > 1:
                file_h = set(map(lambda line: float(line.split(",")[5]), lines[1:]))

        for h in h_rng:
            if h in file_h:
                self.logger.info(f"skip {h}")
                continue

            self.model.set_h(h)

            self.logger.info(f"Run for h={h:.4f}.\t")

            self.model.train()
            res: Result = self.model.get_result()

            msg = "Results:\n"
            for res_k, res_v in res.res.items():
                msg += f"{res_k}: {res_v}\n"
            self.logger.info(msg)

            with report_file.open("a") as file:
                file.writelines([res.row()])

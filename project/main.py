import logging
import os
import sys
from pathlib import Path

import hydra
import netket as nk
from omegaconf import DictConfig
from src.app import App

# from src.monkey_patch import patch_mc_state

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

prj_root = Path(__file__).parent.absolute()
if str(prj_root) not in sys.path:
    sys.path.append(str(prj_root))

# patch_mc_state()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    logging.info(f"NetKet version: {nk.__version__}")

    app = App(cfg=cfg)

    app.run()


if __name__ == "__main__":
    main()

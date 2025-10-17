import logging

# from src.monkey_patch import patch_mc_state
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import netket as nk

from src.app import App

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

prj_root = Path(__file__).parent.absolute()
if str(prj_root) not in sys.path:
    sys.path.append(str(prj_root))

# patch_mc_state()


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--len", required=False, type=int)
    parser.add_argument("--h", required=False, type=float)
    parser.add_argument("--h_min", required=False, type=float)
    parser.add_argument("--h_max", required=False, type=float)
    parser.add_argument("--h_n", required=False, type=int)
    parser.add_argument("--j", required=False, type=float)
    parser.add_argument("--lam", required=False, type=float)
    parser.add_argument("--gamma", required=False, type=float)
    parser.add_argument("--spin", required=False, type=float)
    parser.add_argument("--path_data", required=False, type=str)
    parser.add_argument("--train", required=False, type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    logging.info(f"NetKet version: {nk.__version__}")

    args = get_args()

    app = App(args=args)

    app.run()

import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

import netket as nk
from src.app import App
from src.monkey_patch import patch_mc_state

prj_root = Path(__file__).parent.absolute()
if str(prj_root) not in sys.path:
    sys.path.append(str(prj_root))

# patch_mc_state()


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--len", required=True, type=int)
    parser.add_argument("--h", required=False, type=float)
    parser.add_argument("--j", required=False, type=float, default=1.0)
    parser.add_argument("--lam", required=False, type=float, default=0)
    parser.add_argument("--gamma", required=False, type=float, default=0)
    parser.add_argument("--path_data", required=False, type=str, default="data")
    parser.add_argument("--train", required=False, type=str, default="default")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    logging.info(f"NetKet version: {nk.__version__}")

    args = get_args()

    app = App(args=args)

    app.run()

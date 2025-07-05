import sys
from pathlib import Path
from typing import List

import matplotlib.pylab as plt
import numpy as np

base_prj_path = Path(__file__).parent.parent
sys.path.append(str(base_prj_path.absolute()))

from mpi4py import MPI, futures
from scipy import sparse
from scipy.sparse import linalg
from scripts.exact_test import ham, sigmaz_k


def zz_n_to_other(n: int, length: int) -> List[sparse.csc_matrix]:
    ops = []
    for i in range(length):
        ops.append(sigmaz_k(n, length) * sigmaz_k(i, length))

    return ops


def estimate(ops: List[sparse.csc_matrix], n: int, h: float, j: float) -> List[float]:
    rank = MPI.COMM_WORLD.Get_rank()
    sys.stdout.write(f"I'm {rank}. I'm working with h={h:.4f}.\n")
    sys.stdout.flush()
    ham_ = ham(n, j, h)
    psi = linalg.eigs(ham_, k=1, which="SR", return_eigenvectors=True)[1][:, 0]

    sys.stdout.write(f"I'm {rank}. I've estimated Psi.\n")
    sys.stdout.flush()
    results = np.zeros(shape=len(ops))
    for i, op in enumerate(ops):
        results[i] = np.real(np.vdot(psi, op.dot(psi)))

    sys.stdout.write(f"I'm {rank}. I've finished. Give me a next task!\n")
    sys.stdout.flush()

    return results


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        n = int(sys.argv[1])
        j = 1.0

        h_list = np.hstack((np.linspace(0.5, 1.0, 50), np.linspace(1.1, 1.5, 10)))
        pool = futures.MPIPoolExecutor(max_workers=size)

        dict_of_ops = {h_: zz_n_to_other(n // 2, n) for h_ in h_list}
        dict_of_futures = {
            h_: pool.submit(estimate, ops=ops, n=n, h=h_, j=j)
            for h_, ops in dict_of_ops.items()
        }

        dict_of_results = {h_: f.result() for h_, f in dict_of_futures.items()}

        prefix = base_prj_path.joinpath("estimations").joinpath(f"len_{n}")
        prefix.mkdir(parents=True, exist_ok=True)

        f: plt.Figure = plt.figure(figsize=(18, 12))
        ax = f.subplots(10, 6, sharex=True, sharey=True)
        idx = np.arange(1, n + 1)
        i = j = 0
        for h_, res in dict_of_results.items():
            ax[i, j].plot(idx, res, ".-")
            ax[i, j].set_title(f"Sz * Sz, h={h_:.4f}")
            ax[i, j].set_xlabel("Spin number")
            ax[i, j].set_ylabel("sz * sz")
            fname = str(prefix.joinpath(f"h_{h_:.5e}.txt").absolute())
            np.savetxt(fname, res)
            j += 1
            if j == 6:
                j = 0
                i += 1
        f.tight_layout()
        f.savefig(str(prefix.joinpath("all_plot.png").absolute()))

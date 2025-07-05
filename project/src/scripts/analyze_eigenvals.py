import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI, futures
from scipy.sparse import linalg
from tqdm import tqdm

base_prj_path = Path(__file__).parent.parent
sys.path.append(str(base_prj_path.absolute()))

from main import check_bad_params
from scripts.exact_test import ham

plt.style.use("ggplot")
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    }
)


def estimate_diff(N: int, j: float, h: float) -> float:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    sys.stdout.write(f"I'm {rank}: start the task, h={h}\n")
    sys.stdout.flush()
    op = ham(N, j, h)
    eigvals = linalg.eigs(op, k=2, return_eigenvectors=False, which="SR")

    sys.stdout.write(f"I'm {rank}: eigvals have been computeed. Give me a next one!\n")
    sys.stdout.flush()

    return (np.abs(eigvals[1] - eigvals[0])) / (np.abs(eigvals[0]))


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        with futures.MPIPoolExecutor(max_workers=size) as pool:
            N = int(sys.argv[1])
            j = 1.00

            bad_points = np.linspace(0.5, 0.99, 10000)
            mask = np.zeros(bad_points.shape[0])

            for i, x in enumerate(bad_points):
                if not check_bad_params(N, j, x):
                    mask[i] = 1

            bad_points = bad_points[mask == 1]
            real_bad_points = []

            for bp in bad_points:
                if len(real_bad_points) == 0:
                    real_bad_points.append(bp)
                else:
                    res = True
                    for cbp in real_bad_points:
                        res &= np.abs(bp - cbp) >= 0.05

                    if res:
                        real_bad_points.append(bp)

            bad_points = np.array(real_bad_points)
            points = []

            for bp in bad_points:
                points.append(bp)

                for dif in np.linspace(0.005, 0.1, 10):
                    points.append(bp + dif)
                    points.append(bp - dif)

            sys.stdout.write(
                f"There are {len(points)} candiadtes. Start the computataions!\n"
            )
            sys.stdout.flush()
            futures = {p: pool.submit(estimate_diff, N=N, h=p, j=1.0) for p in points}

            results = {k: v.result() for k, v in futures.items()}

            f = plt.figure(figsize=(10, 8))
            ax = f.add_subplot()
            ax.set_title(r"$\frac{| E_0 - E_1 |}{| E_0 |}$ in the degeneracy area.")
            for bp in bad_points:
                ax.axvline(bp, linestyle="--", label=f"Degeneracy point {bp}")

            ax.plot(
                results.keys(),
                results.values(),
                "o",
                label=r"$\frac{| E_0 - E_1 |}{| E_0 |}$",
            )
            ax.set_xlabel("h")
            ax.set_ylabel("")
            ax.legend()

            folder = base_prj_path.joinpath("plots")
            folder.mkdir(exist_ok=True)

            f.savefig("plots/Energies analysis.png")

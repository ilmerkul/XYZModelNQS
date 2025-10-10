import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

base_prj_path = Path(__file__).parent.parent
sys.path.append(str(base_prj_path.absolute()))

from python.XXModelImpl import exact_solution

if __name__ == "__main__":
    plt.style.use("ggplot")

    n = int(sys.argv[1])
    j = 1.0

    h = np.linspace(0.0, 1.5, 5000)
    yy = []

    for h_ in h:
        exact = exact_solution(n, h_, j)
        yy.append(exact.zz[0])

    f = plt.figure(figsize=(10, 8))
    ax = f.add_subplot()
    ax.plot(h, yy, ".", label=f"ZZ for {n} spins")
    ax.legend()
    ax.set_xlabel("h")
    ax.set_ylabel("ZZ")
    f.savefig(f"plots/ZZ_analytical_{n}.png")

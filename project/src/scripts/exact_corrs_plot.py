import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")
plt.rc("text", usetex=True)


if __name__ == "__main__":
    n = int(sys.argv[1])
    ss = int(sys.argv[2])
    base_dir = Path(__file__).parent.parent.joinpath("estimations").joinpath(f"len_{n}")

    h_list = []
    corr_list = []

    for sub in base_dir.glob("*"):
        if sub.suffix == ".png":
            continue

        h = float(sub.name[2:-4])
        corrs = np.loadtxt(str(sub.absolute()))
        corr_list.append(corrs[ss])
        h_list.append(h)

    f = plt.figure(figsize=(8, 6))
    ax = f.add_subplot()
    ax.plot(h_list, corr_list, "o")
    ax.set_title(f"$\sigma^z_{n // 2 + 1}\cdot\sigma^z_{ss + 1}$")
    ax.set_xlabel("h")
    ax.set_ylabel("$\sigma^z\cdot\sigma^z$")
    f.savefig(str(base_dir.joinpath(f"ZZ_{n // 2 + 1}-{ss + 1}.png")))

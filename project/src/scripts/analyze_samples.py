import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")


def slow_read_arr(file: Path, dim: int):
    rows = file.read_text().split("\n")
    cnts = {}
    num = 0

    for r in rows:
        if len(r.split(" ")) != dim:
            continue
        else:
            num += 1
            if r not in cnts:
                cnts[r] = 1
            else:
                cnts[r] += 1

    return max(cnts.values()) / num


if __name__ == "__main__":
    n = int(sys.argv[1])

    data_root_dir = Path(__file__).parent.parent.joinpath(f"len_{n}")

    h_vals = []
    part_vals = []

    for subfolder in data_root_dir.glob("*"):
        h = float(subfolder.name[2:])
        part = slow_read_arr(subfolder.joinpath("samples.txt"), n)

        h_vals.append(h)
        part_vals.append(part)

    f = plt.figure(figsize=(8, 6))
    ax = f.add_subplot()
    ax.plot(h_vals, part_vals, "o")
    ax.set_title(f"Relative part of most common sample for len={n}")
    ax.set_xlabel("h")
    ax.set_ylabel("part")

    plot_path = Path(__file__).parent.parent.joinpath("plots").joinpath(f"plots_{n}")
    plot_path.mkdir(parents=True, exist_ok=True)
    fname = plot_path.joinpath("Samples_part.png")
    f.savefig(str(fname.absolute()))

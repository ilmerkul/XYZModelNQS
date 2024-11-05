import json
from typing import Dict

import numpy as np
from src import exact


class Results:
    def __init__(self, h: float, j: float, n: int, type: str):
        self.h = h
        self.j = j
        self.n = n
        self.type = type

    def analytical(self):
        """Analytical solution"""
        ares = {}
        els = exact.get_ellist(self.n, self.h, self.j)

        if np.abs(self.h) > np.abs(self.j):
            ares["e"] = -self.h * self.n / 2.0
            ares["xx"] = ares["yy"] = 0.0
            ares["zz"] = 0.25
            ares["1z"] = ares["nz"] = -0.5
            ares["m"] = -0.5
        else:
            l0 = np.sum(els < 0)

            ares["e"] = exact.analytical_energy(self.n, self.h, self.j, l0)
            ares["xx"] = ares["yy"] = exact.analytical_s1sn_xy(self.n, l0)
            ares["zz"] = exact.analytical_s1sn_z(self.n, l0)
            ares["1z"] = ares["nz"] = exact.analytical_s1z(self.n, l0)
            ares["m"] = exact.analytical_m(self.n, l0)

        self.ares = ares

    def exact(self) -> float:
        return self.ares["e"]

    def update(
        self,
        energy: float,
        var: float,
        spins: np.ndarray,
        xx: float,
        yy: float,
        zz: float,
        zz_mid: float,
    ):
        self.var = var
        self.spins = spins

        res = {}
        res["e"] = energy
        res["1z"] = spins[0]
        res["nz"] = spins[self.n - 1]
        res["xx"] = xx
        res["yy"] = yy
        res["zz"] = zz
        res["m"] = spins.sum() / self.n
        res["zz_mid"] = zz_mid

        self.res = res

    def row(self) -> str:
        if self.type == "xy":
            return (
                f"{self.n},"
                + ",".join(
                    map(
                        lambda x: f"{x:.5f}",
                        [self.h, self.j] + list(self.res.values()),
                    )
                )
                + "\n"
            )
        return (
            f"{self.n},"
            + ",".join(
                map(
                    lambda x: f"{x:.5f}",
                    [self.h, self.j]
                    + list(self.ares.values())
                    + list(self.res.values()),
                )
            )
            + "\n"
        )

    def header(self) -> str:
        if self.type == "xy":
            return (
                "n_spins,h,j,"
                + ",".join([f"estimated_{v}" for v in self.res.keys()])
                + "\n"
            )
        return (
            "n_spins,h,j,"
            + ",".join(
                [f"analytical_{v}" for v in self.ares.keys()]
                + [f"estimated_{v}" for v in self.res.keys()]
            )
            + "\n"
        )

    def get_spins(self) -> str:
        return ",".join([f"{v}:.5f" for v in self.spins])

    def get_entropies(self) -> str:
        return json.dumps(self.entropies)

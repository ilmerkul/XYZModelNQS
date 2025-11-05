from dataclasses import asdict, dataclass, fields
from typing import Callable, Dict

import jax
import jax.numpy as jnp
import netket as nk
from netket.operator.spin import sigmax, sigmay, sigmaz
from src.model.nqs.operator import get_model_netket_op
from src.model.struct import ChainConfig, NameChain, get_type

from ..exact import (
    analytical_energy,
    analytical_m,
    analytical_s1sn_xy,
    analytical_s1sn_z,
    analytical_s1z,
    get_ellist,
)


@dataclass
class ResultData:
    energy: float = 0.0
    energy_var: float = 0.0
    spins: jnp.ndarray = None
    xx: float = 0.0
    yy: float = 0.0
    zz: float = 0.0
    zz_mid: float = 0.0


class Result:
    ENERGY: str = "e"
    ENERGY_VAR: str = "e_var"
    SIGMA_Z_ONE: str = "1z"
    SIGMA_Z_LAST: str = "nz"
    SIGMA_XX_LEN: str = "xx"
    SIGMA_YY_LEN: str = "yy"
    SIGMA_ZZ_LEN: str = "zz"
    SIGMA_ZZ_MID_LEN: str = "zz_mid"

    TREE_RES: Dict[str, Callable] = {
        ENERGY: lambda res_data: res_data.energy,
        ENERGY_VAR: lambda res_data: res_data.energy_var,
        SIGMA_Z_ONE: lambda res_data: res_data.spins[0],
        SIGMA_Z_LAST: lambda res_data: res_data.spins[-1],
        SIGMA_XX_LEN: lambda res_data: res_data.xx,
        SIGMA_YY_LEN: lambda res_data: res_data.yy,
        SIGMA_ZZ_LEN: lambda res_data: res_data.zz,
        SIGMA_ZZ_MID_LEN: lambda res_data: res_data.spins.sum() / len(res_data.spins),
    }

    def __init__(self, cfg: ChainConfig):
        self.cfg = cfg
        self.data = None
        self.res = None
        self.ares = None

    @staticmethod
    def analytical_xy(cfg: ChainConfig) -> Dict[str, float]:
        """Analytical solution"""
        if get_type(cfg) != NameChain.XY:
            raise ValueError("type is not xy")

        ares = {}
        els = get_ellist(cfg.n, cfg.h, cfg.j)

        if jnp.abs(cfg.h) > jnp.abs(cfg.j):
            ares[Result.ENERGY] = -cfg.h * cfg.n / 2.0
            ares[Result.SIGMA_XX_LEN] = ares[Result.SIGMA_YY_LEN] = 0.0
            ares[Result.SIGMA_ZZ_LEN] = 0.25
            ares[Result.SIGMA_Z_ONE] = ares[Result.SIGMA_Z_LAST] = -0.5
            ares[Result.SIGMA_ZZ_MID_LEN] = -0.5
        else:
            l0 = jnp.sum(els < 0)

            ares[Result.ENERGY] = analytical_energy(cfg.n, cfg.h, cfg.j, l0)
            ares[Result.SIGMA_XX_LEN] = ares[Result.SIGMA_YY_LEN] = analytical_s1sn_xy(
                cfg.n, l0
            )
            ares[Result.SIGMA_ZZ_LEN] = analytical_s1sn_z(cfg.n, l0)
            ares[Result.SIGMA_Z_ONE] = ares[Result.SIGMA_Z_LAST] = analytical_s1z(
                cfg.n, l0
            )
            ares[Result.SIGMA_ZZ_MID_LEN] = analytical_m(cfg.n, l0)

        return ares

    @staticmethod
    def analytical_energy(
        cfg: ChainConfig, hilbert: nk.hilbert.Spin, order: int = 0
    ) -> float:
        H = get_model_netket_op(cfg, hilbert)
        E_gs = nk.exact.lanczos_ed(H, k=order + 1, compute_eigenvectors=False)
        en = E_gs[order]
        return en

    def _update(self, res_data: ResultData) -> Dict[str, float]:
        res = dict()
        for res_type, res_func in self.TREE_RES.items():
            res[res_type] = res_func(res_data)
        return res

    def update(self, res_data: ResultData):
        self.data = res_data

        self.res = self._update(res_data)

    def row(self) -> str:
        row = ",".join(map(lambda x: str(x), asdict(self.cfg).values())) + ","
        row += ",".join(map(lambda x: f"{x:.5f}", list(self.res.values())))
        row += "\n"
        return row

    @staticmethod
    def header() -> str:
        header = ",".join(map(lambda field: field.name, fields(ChainConfig))) + ","
        header += ",".join([f"estimated_{v}" for v in Result.TREE_RES.keys()])
        header += "\n"
        return header

    @staticmethod
    def get_spin_operators(
        cfg: ChainConfig, hilbert: nk.hilbert.Spin, dtype: jnp.dtype = jnp.complex128
    ) -> Dict[str, nk.operator.LocalOperator]:
        """
        Generate measurable operators
        """
        ops = {}

        for i in range(cfg.n):
            ops[f"s_{i}"] = sigmaz(hilbert, i, dtype=dtype)

        ops[Result.SIGMA_XX_LEN] = sigmax(hilbert, 0, dtype=dtype) * sigmax(
            hilbert, cfg.n - 1
        )
        ops[Result.SIGMA_YY_LEN] = sigmay(hilbert, 0, dtype=dtype) * sigmay(
            hilbert, cfg.n - 1
        )
        ops[Result.SIGMA_ZZ_LEN] = sigmaz(hilbert, 0, dtype=dtype) * sigmaz(
            hilbert, cfg.n - 1
        )
        ops[Result.SIGMA_ZZ_MID_LEN] = sigmaz(hilbert, 0, dtype=dtype) * sigmaz(
            hilbert, cfg.n // 2
        )
        ops[Result.ENERGY] = get_model_netket_op(cfg, hilbert)

        return ops

    @staticmethod
    def ops_vals_to_res_data(ops_vals: jax.tree) -> ResultData:
        return ResultData(
            energy=jnp.real(ops_vals["e"].mean),
            energy_var=jnp.real(ops_vals["e"].variance),
            spins=jnp.array(
                [
                    jnp.real(val.mean)
                    for op, val in ops_vals.items()
                    if op.startswith("s_")
                ]
            ),
            xx=jnp.real(ops_vals["xx"].mean),
            yy=jnp.real(ops_vals["yy"].mean),
            zz=jnp.real(ops_vals["zz"].mean),
            zz_mid=jnp.real(ops_vals["zz_mid"].mean),
        )

from dataclasses import dataclass


@dataclass(frozen=True)
class ChainConfig:
    spin: float
    gamma: float
    lam: float
    j: float
    n: int
    h: float
    pbc: bool

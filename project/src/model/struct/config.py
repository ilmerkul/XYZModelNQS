from dataclasses import dataclass

@dataclass
class ChainConfig:
    spin: float = 1 / 2
    gamma: float = 1.0
    lam: float = 1.0
    j: float = 1.0
    n: int = 10
    h: float = 1.0

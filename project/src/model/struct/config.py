from dataclasses import dataclass, field
from typing import List


@dataclass
class SymmetryChain:
    Z2: str = "z2"
    Z2_EMB: str = "z2_emb"
    TRANSLATION: str = "translation"
    MIRROR: str = "mirror"


@dataclass
class NameChain:
    XY: str = "xy"
    XX: str = "xx"
    XYZ: str = "xyz"
    XXX: str = "xxx"
    XXZ: str = "xxz"
    TFI: str = "tfi"
    OTHER: str = "other"


@dataclass(frozen=True)
class ChainConfig:
    spin: float
    gamma: float
    lam: float
    j: float
    n: int
    h: float
    pbc: bool

    symmetries: List[str] = field(init=False)
    name: str = field(init=False)

    def __post_init__(self):
        name = get_type(cfg=self)
        symmetries = [SymmetryChain.MIRROR]

        if name == NameChain.XX:
            symmetries.append(SymmetryChain.Z2_EMB)

        if self.h == 0 and name != NameChain.XX:
            symmetries.append(SymmetryChain.Z2)

        if self.pbc:
            symmetries.append(SymmetryChain.TRANSLATION)

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "symmetries", symmetries)


def get_type(cfg: ChainConfig):
    if cfg.gamma == 0.0 and cfg.lam == 0.0:
        return NameChain.XX
    elif cfg.gamma == 0.0 and (cfg.lam == 1.0 or cfg.lam == -1.0):
        return NameChain.TFI
    elif cfg.gamma == 0.0:
        return NameChain.XY
    elif cfg.gamma == 1.0 and cfg.lam == 0.0:
        return NameChain.XXX
    elif cfg.gamma != 1.0 and cfg.lam == 0.0:
        return NameChain.XXZ
    elif cfg.gamma not in (0.0, 1, 0) and cfg.lam != 0.0:
        return NameChain.XXZ
    return NameChain.OTHER

from dataclasses import dataclass


class NNType:
    CNN: str = "cnn"
    FFN: str = "ffn"
    FFN_SYMM: str = "ffn_symm"
    GCNN: str = "gcnn"
    TUTORIAL_VIT: str = "tutorial_vit"
    TRANSFORMER: str = "transformer"
    PHASE_TRANSFORMER: str = "phase_transformer"


@dataclass(frozen=True)
class NNConfig:
    nntype: NNType
    seed: int
    training: bool

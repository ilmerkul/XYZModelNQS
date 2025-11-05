from dataclasses import dataclass


class NNType:
    CNN: str = "cnn"
    FFN: str = "ffn"
    GCNN: str = "gcnn"
    TRANSFORMER: str = "transformer"
    PHASE_TRANSFORMER: str = "phase_transformer"


@dataclass(frozen=True)
class NNConfig:
    nntype: NNType

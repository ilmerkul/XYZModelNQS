from dataclasses import dataclass


class OptimizerType:
    ADAM: str = "adam"
    ADAM_COS: str = "adam_cos"
    ADAM_EXP: str = "adam_exp"
    SGD_LIN_EXP: str = "sgd_lin_exp"
    SGD_EXP: str = "sgd_exp"
    ADAM_ZERO_PQC: str = "adam_zero_pqc"
    ADAM_ZERO_TR: str = "adam_zero_tr"


@dataclass
class OptimizerConfig:
    type: str
    lr: float
    n_iter: int
    global_norm: float

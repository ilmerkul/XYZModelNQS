from ..nqs import (
    adam,
    adam_cos,
    adam_exp,
    adam_zero_pqc,
    adam_zero_tr,
    sgd_exp,
    sgd_lin_exp,
    AdamConfig,
    AdamCosConfig,
    SGDExpConfig,
    AdamExpConfig,SGDLinExpConfig,AdamZeroPqcTrConfig
)
from ..interface import OptimizerType


type2optimizer = {
            OptimizerType.ADAM: adam,
            OptimizerType.ADAM_COS: adam_cos,
            OptimizerType.ADAM_EXP: adam_exp,
            OptimizerType.SGD_LIN_EXP: sgd_lin_exp,
            OptimizerType.SGD_EXP: sgd_exp,
            OptimizerType.ADAM_ZERO_PQC: adam_zero_pqc,
            OptimizerType.ADAM_ZERO_TR: adam_zero_tr,
        }
type2config = {
            OptimizerType.ADAM: AdamConfig,
            OptimizerType.ADAM_COS: AdamCosConfig,
            OptimizerType.ADAM_EXP: AdamExpConfig,
            OptimizerType.SGD_LIN_EXP: SGDLinExpConfig,
            OptimizerType.SGD_EXP: SGDExpConfig,
            OptimizerType.ADAM_ZERO_PQC: AdamZeroPqcTrConfig,
            OptimizerType.ADAM_ZERO_TR: AdamZeroPqcTrConfig,
        }

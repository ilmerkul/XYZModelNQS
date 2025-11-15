from ..callbacks import (
    AdaptiveMomentumCallback,
    ChainReset,
    ParametersPrint,
    RHatStop,
    SigmaCallback,
    VarianceCallback,
)
from ..interface import CallbackType

type2callback = {
            CallbackType.Variance: VarianceCallback,
            CallbackType.Sigma: SigmaCallback,
            CallbackType.ChainReset: ChainReset,
            CallbackType.RHatStop: RHatStop,
            CallbackType.ParametersPrint: ParametersPrint,
            CallbackType.AdaptiveMomentum: AdaptiveMomentumCallback,
        }
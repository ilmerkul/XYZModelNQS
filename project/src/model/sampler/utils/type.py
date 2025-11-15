from ..sampler import MetropolisSampler, TransformerSampler, MetropolisSamplerConfig, TransformerSamplerConfig
from ..interface import SamplerType

type2sampler = {
            SamplerType.TRANSFORMER: TransformerSampler,
            SamplerType.METROPOLIS: MetropolisSampler,
        }
type2config = {
            SamplerType.TRANSFORMER: TransformerSamplerConfig,
            SamplerType.METROPOLIS: MetropolisSamplerConfig,
        }
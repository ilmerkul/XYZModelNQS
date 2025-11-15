from ..interface import NNType
from ..convolution import CNNConfig
from ..feedforward import FFConfig, FFSymmConfig
from ..graph import GCNNConfig
from ..transformer.phaseTransformer import PhaseTransformerConfig
from ..transformer.transformer import TransformerConfig
from ..transformer.tutorial import TutorialViTConfig
from ..convolution import CNN
from ..feedforward import FF, SymmModel
from ..graph import GCNN
from ..transformer import PhaseTransformer, Transformer, TutorialViT

type2config = {
        NNType.TRANSFORMER: TransformerConfig,
        NNType.CNN: CNNConfig,
        NNType.FFN: FFConfig,
        NNType.GCNN: GCNNConfig,
        NNType.TUTORIAL_VIT: TutorialViTConfig,
        NNType.FFN_SYMM: FFSymmConfig,
        NNType.PHASE_TRANSFORMER: PhaseTransformerConfig,
    }

type2nn = {
        NNType.TRANSFORMER: Transformer,
        NNType.CNN: CNN,
        NNType.FFN: FF,
        NNType.GCNN: GCNN,
        NNType.TUTORIAL_VIT: TutorialViT,
        NNType.FFN_SYMM: SymmModel,
        NNType.PHASE_TRANSFORMER: PhaseTransformer,
    }

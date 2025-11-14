import os
import pathlib

import jax.numpy as jnp
from hydra.utils import instantiate
from omegaconf import OmegaConf
from src.model.NN import NNConfig, NNType
from src.model.NN.convolution import CNNConfig
from src.model.NN.feedforward import FFConfig, FFSymmConfig
from src.model.NN.graph import GCNNConfig
from src.model.NN.transformer.phaseTransformer import PhaseTransformerConfig
from src.model.NN.transformer.transformer import TransformerConfig
from src.model.NN.transformer.tutorial import TutorialViTConfig
from src.model.nqs import ModelNQSConfig

project_path = pathlib.Path(os.getcwd())


def get_phase_transformer_config(chain_cfg) -> NNConfig:
    return PhaseTransformerConfig(
        tr_config=None,
        pqc=False,
        gcnn=True,
        phase_train=False,
        gcnn_config=None,
    )


def dict2class_config(cfg) -> ModelNQSConfig:
    config_dict = {
        NNType.TRANSFORMER: TransformerConfig,
        NNType.CNN: CNNConfig,
        NNType.FFN: FFConfig,
        NNType.GCNN: GCNNConfig,
        NNType.TUTORIAL_VIT: TutorialViTConfig,
        NNType.FFN_SYMM: FFSymmConfig,
        # NNType.PHASE_TRANSFORMER: PhaseTransformerConfig,
    }

    dtype_dict = {
        "float32": jnp.float32,
        "float64": jnp.float64,
        "float16": jnp.float16,
    }

    chain_cfg = instantiate(cfg["chain"])
    nn_params = OmegaConf.to_container(cfg["model"])
    nn_params["dtype"] = dtype_dict[nn_params["dtype"]]
    nnconfig = config_dict[nn_params["nntype"]](**nn_params, chain=chain_cfg)
    nqs_params = OmegaConf.to_container(cfg["nqs"])
    model_config = ModelNQSConfig(**nqs_params, chain=chain_cfg, model_config=nnconfig)

    return model_config

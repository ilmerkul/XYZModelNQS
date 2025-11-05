import os
import pathlib

import jax.numpy as jnp
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from src.model.NN import NNConfig, NNType
from src.model.NN.convolution import CNNConfig
from src.model.NN.feedforward import FFConfig
from src.model.NN.graph import GCNNConfig
from src.model.NN.transformer.phaseTransformer import PhaseTransformerConfig
from src.model.NN.transformer.transformer import PosEmbType, TransformerConfig
from src.model.nqs import ModelNQSConfig
from src.model.optimizer import NQSOptimizer
from src.model.sampler import SamplerType
from src.model.struct import ChainConfig

project_path = pathlib.Path(os.getcwd())


def get_chain_config(args) -> ChainConfig:
    return ChainConfig(
        n=16 if args.len is None else args.len,
        j=-1 if args.j is None else args.j,
        h=0.0 if args.h is None else args.h,
        lam=0 if args.lam is None else args.lam,
        gamma=0 if args.gamma is None else args.gamma,
        spin=1 / 2 if args.spin is None else args.spin,
        pbc=False,
    )


def get_phase_transformer_config(chain_cfg) -> NNConfig:
    return PhaseTransformerConfig(
        tr_config=get_transformer_config(chain_cfg),
        pqc=False,
        gcnn=True,
        phase_train=False,
        gcnn_config=get_gcnn_config(chain_cfg),
    )


def get_transformer_config(chain_cfg) -> NNConfig:
    return TransformerConfig(
        nntype=NNType.TRANSFORMER,
        chain=chain_cfg,
        use_bias=True,
        use_dropout=True,
        dropout_rate=0.2,
        inverse_iter_rate=0.5,
        training=True,
        seed=42,
        autoregressive=False,
        dtype=jnp.float32,
        embed_concat=False,
        pos_embed=PosEmbType.ROTARY,
        eps=1e-10,
    )


def get_cnn_config(chain_cfg) -> NNConfig:
    return CNNConfig(chain=chain_cfg, dtype=jnp.float64, symm=True, use_bias=True)


def get_gcnn_config(chain_cfg) -> NNConfig:
    return GCNNConfig(
        chain=chain_cfg,
        dtype=jnp.float64,
    )


def get_ff_config(chain_cfg) -> NNConfig:
    return FFConfig(chain=chain_cfg, dtype=jnp.float64, precision=None, use_bias=True)


def get_nn_config(nntype: str, chain_cfg: ChainConfig) -> NNConfig:
    config_dict = {
        NNType.TRANSFORMER: get_transformer_config,
        NNType.CNN: get_cnn_config,
        NNType.FFN: get_ff_config,
        NNType.GCNN: get_gcnn_config,
        NNType.PHASE_TRANSFORMER: get_phase_transformer_config,
    }

    return config_dict[nntype](chain_cfg)


def get_model_nqs_config(args, save_model_path) -> ModelNQSConfig:
    chain_cfg = get_chain_config(args=args)
    nntype = NNType.TRANSFORMER
    nnconfig = get_nn_config(nntype, chain_cfg)
    model_config = ModelNQSConfig(
        chain=chain_cfg,
        optimizer=NQSOptimizer.SGD_EXP,
        sampler=SamplerType.METROPOLIS,
        n_iter=2000,
        n_chains=500,
        lr=1e-4,
        min_n_samples=1000,
        scale_n_samples=100,
        preconditioner=True,
        sr_diag_shift=1e-2,
        model_config=nnconfig,
        tr_learning=False,
        save_model_path=project_path / save_model_path,
    )

    return model_config


def dict2class_config(cfg) -> ModelNQSConfig:
    config_dict = {
        NNType.TRANSFORMER: TransformerConfig,
        NNType.CNN: CNNConfig,
        NNType.FFN: FFConfig,
        NNType.GCNN: GCNNConfig,
        NNType.PHASE_TRANSFORMER: PhaseTransformerConfig,
    }

    dtype_dict = {"float32": jnp.float32}

    chain_cfg = instantiate(cfg["chain"])
    nn_params = OmegaConf.to_container(cfg["model"])
    nn_params["dtype"] = dtype_dict[nn_params["dtype"]]
    nnconfig = config_dict[nn_params["nntype"]](**nn_params, chain=chain_cfg)
    nqs_params = OmegaConf.to_container(cfg["nqs"])
    model_config = ModelNQSConfig(**nqs_params, chain=chain_cfg, model_config=nnconfig)

    return model_config

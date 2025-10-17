import os
import pathlib

import jax.numpy as jnp

from src.model.NN import NNType
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
        n=12 if args.len is None else args.len,
        j=1 if args.j is None else args.j,
        h=0.0 if args.h is None else args.h,
        lam=1 if args.lam is None else args.lam,
        gamma=0 if args.gamma is None else args.gamma,
        spin=1 / 2 if args.spin is None else args.spin,
        pbc=False,
    )


def get_phase_transformer_config(args, save_model_path) -> ModelNQSConfig:
    chain_cfg = get_chain_config(args=args)
    transformer_config = TransformerConfig(
        chain=chain_cfg,
        use_bias=True,
        use_dropout=False,
        dropout_rate=0.1,
        inverse_iter_rate=0.3,
        training=True,
        seed=42,
        autoregressive=False,
        dtype=jnp.float64,
        symm=False,
        embed_concat=False,
        state_inverse=False,
        pos_embed=PosEmbType.ROTARY,
        eps=1e-10,
    )
    gcnn_config = GCNNConfig(
        chain=chain_cfg,
        dtype=transformer_config.dtype,
    )
    phase_transformer = PhaseTransformerConfig(
        tr_config=transformer_config,
        pqc=False,
        gcnn=True,
        phase_train=False,
        gcnn_config=gcnn_config,
    )
    model_config = ModelNQSConfig(
        chain=chain_cfg,
        nn=NNType.PHASE_TRANSFORMER,
        optimizer=NQSOptimizer.ADAM_ZERO_PQC,
        sampler=SamplerType.METROPOLIS,
        n_iter=1000,
        n_chains=500,
        lr=5e-4,
        min_n_samples=500,
        scale_n_samples=50,
        preconditioner=True,
        dtype=phase_transformer.tr_config.dtype,
        rnd_seed=phase_transformer.tr_config.seed,
        sr_diag_shift=1e-2,
        model_config=phase_transformer,
        tr_learning=False,
        save_model_path=project_path / save_model_path,
    )

    return model_config


def get_transformer_config(args, save_model_path) -> ModelNQSConfig:
    chain_cfg = get_chain_config(args=args)
    transformer_config = TransformerConfig(
        chain=chain_cfg,
        use_bias=True,
        use_dropout=False,
        dropout_rate=0.1,
        inverse_iter_rate=0.3,
        training=True,
        seed=42,
        autoregressive=False,
        dtype=jnp.float64,
        symm=False,
        embed_concat=False,
        state_inverse=False,
        pos_embed=PosEmbType.ROTARY,
        eps=1e-10,
    )
    model_config = ModelNQSConfig(
        chain=chain_cfg,
        nn=NNType.TRANSFORMER,
        optimizer=NQSOptimizer.ADAM_ZERO_PQC,
        sampler=SamplerType.METROPOLIS,
        n_iter=1000,
        n_chains=500,
        lr=5e-4,
        min_n_samples=500,
        scale_n_samples=50,
        preconditioner=True,
        dtype=transformer_config.dtype,
        rnd_seed=transformer_config.seed,
        sr_diag_shift=1e-2,
        model_config=transformer_config,
        tr_learning=False,
        save_model_path=project_path / save_model_path,
    )

    return model_config


def get_cnn_config(args, save_model_path) -> ModelNQSConfig:
    chain_cfg = get_chain_config(args=args)
    cnn_config = CNNConfig(chain=chain_cfg, dtype=jnp.float64, symm=True, use_bias=True)
    model_config = ModelNQSConfig(
        chain=chain_cfg,
        nn=NNType.CNN,
        optimizer=NQSOptimizer.ADAM,
        sampler=SamplerType.METROPOLIS,
        n_iter=1000,
        n_chains=500,
        lr=5e-4,
        min_n_samples=500,
        scale_n_samples=50,
        preconditioner=True,
        dtype=cnn_config.dtype,
        rnd_seed=42,
        sr_diag_shift=1e-2,
        model_config=cnn_config,
        tr_learning=False,
        save_model_path=project_path / save_model_path,
    )

    return model_config


def get_gcnn_config(args, save_model_path) -> ModelNQSConfig:
    chain_cfg = get_chain_config(args=args)
    gcnn_config = GCNNConfig(
        chain=chain_cfg,
        dtype=jnp.float64,
    )

    model_config = ModelNQSConfig(
        chain=chain_cfg,
        nn=NNType.GCNN,
        optimizer=NQSOptimizer.ADAM,
        sampler=SamplerType.METROPOLIS,
        n_iter=1000,
        n_chains=500,
        lr=5e-4,
        min_n_samples=500,
        scale_n_samples=50,
        preconditioner=True,
        dtype=gcnn_config.dtype,
        rnd_seed=42,
        sr_diag_shift=1e-2,
        model_config=gcnn_config,
        tr_learning=False,
        save_model_path=project_path / save_model_path,
    )

    return model_config


def get_ff_config(args, save_model_path) -> ModelNQSConfig:
    chain_cfg = get_chain_config(args=args)
    ff_config = FFConfig(
        chain=chain_cfg, dtype=jnp.float64, precision=None, use_bias=True
    )

    model_config = ModelNQSConfig(
        chain=chain_cfg,
        nn=NNType.FFN,
        optimizer=NQSOptimizer.ADAM,
        sampler=SamplerType.METROPOLIS,
        n_iter=1000,
        n_chains=500,
        lr=5e-4,
        min_n_samples=500,
        scale_n_samples=50,
        preconditioner=True,
        dtype=ff_config.dtype,
        rnd_seed=42,
        sr_diag_shift=1e-2,
        model_config=ff_config,
        tr_learning=False,
        save_model_path=project_path / save_model_path,
    )

    return model_config

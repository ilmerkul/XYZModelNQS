import os
import pathlib

import jax.numpy as jnp
from hydra.utils import instantiate
from omegaconf import OmegaConf
from src.model.NN.utils import type2config as nn_type2config
from src.model.nqs import ModelNQSConfig
from src.model.sampler.utils import type2config as sampler_type2config
from src.model.optimizer.utils import type2config as optimizer_type2config

project_path = pathlib.Path(os.getcwd())


def dict2class_config(cfg) -> ModelNQSConfig:
    dtype_dict = {
        "float32": jnp.float32,
        "float64": jnp.float64,
        "float16": jnp.float16,
    }

    chain_cfg = instantiate(cfg["chain"])

    nn_config = OmegaConf.to_container(cfg["model"])
    dtype = dtype_dict[nn_config["dtype"]]
    nn_config["dtype"] = dtype
    nn_config = nn_type2config[nn_config["nntype"]](**nn_config, chain=chain_cfg)

    sampler_config = OmegaConf.to_container(cfg["sampler"])
    sampler_config["dtype"] = dtype
    sampler_config = sampler_type2config[sampler_config["type"]](**sampler_config)

    optimizer_config = OmegaConf.to_container(cfg["optimizer"])
    optimizer_config = optimizer_type2config[optimizer_config["type"]](**optimizer_config)

    nqs_params = OmegaConf.to_container(cfg["nqs"])
    model_config = ModelNQSConfig(**nqs_params, chain=chain_cfg, model_config=nn_config, sampler=sampler_config, optimizer=optimizer_config)

    return model_config

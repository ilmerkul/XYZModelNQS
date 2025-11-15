import jax
import jax.numpy as jnp
import jax.random as jrnd
from ..interface import NNType

from .load import load_model


def apply_fun(
    params,
    variables,
    *,
    model,
    nntype: NNType,
    rnd_seed: int,
    **kwargs,
):
    x_hash = jnp.abs(jnp.sum(variables)).astype(jnp.int32)
    base_key = jrnd.PRNGKey(rnd_seed)
    key = jrnd.fold_in(base_key, x_hash)
    dropout_key, symmetry_key, output_key = jrnd.split(key, num=3)

    if nntype == NNType.PHASE_TRANSFORMER:
        output = model.apply(
            params,
            variables,
            rngs={
                "dropout": dropout_key,
                "symmetry": symmetry_key,
                "output": output_key,
            },
            **kwargs,
        )
    elif nntype == NNType.TRANSFORMER:
        output = model.apply(
            params,
            variables,
            rngs={
                "dropout": dropout_key,
                "symmetry": symmetry_key,
                "output": output_key,
            },
            **kwargs,
        )
    elif nntype in [
        NNType.CNN,
        NNType.GCNN,
        NNType.FFN,
        NNType.TUTORIAL_VIT,
        NNType.FFN_SYMM,
    ]:
        output = model.apply(
            params,
            variables,
            **kwargs,
        )
    else:
        raise ValueError("error apply function")

    return output


def init_fun(params_rng_key, inp, *, model):
    rng_key = params_rng_key["params"]
    init_rng = {
        "params": rng_key,
        "dropout": rng_key,
        "symmetry": rng_key,
        "output": rng_key,
    }

    variables = model.init(init_rng, inp)

    return variables


def init_fun_tr_learning( params_rng_key, inp, *, model, tr_learning: bool, model_file:str, noise_scale: float
):
    rng_key = params_rng_key["params"]
    variables = init_fun(params_rng_key, inp, model=model)

    if tr_learning and model_file is not None:
        loaded_params = load_model(model_file=model_file)
        if loaded_params is not None:

            def add_adaptive_noise(params, key, noise_scale):
                def add_noise_to_leaf(param, leaf_key):
                    param_std = jnp.std(param)
                    noise = jrnd.normal(leaf_key, param.shape) * param_std * noise_scale
                    return param + noise

                keys_tree = jax.tree_util.tree_map(
                    lambda param: jrnd.fold_in(key, jnp.sum(param).astype(jnp.int32)),
                    params,
                )

                return jax.tree_util.tree_map(add_noise_to_leaf, params, keys_tree)

            noisy_params = add_adaptive_noise(
                loaded_params, rng_key, noise_scale=noise_scale
            )

            variables = {"params": noisy_params}

    return variables

import os
import pickle

import jax


def load_model(model_file: str):
    if os.path.exists(model_file):
        with open(model_file, "rb") as f:
            loaded_params = pickle.load(f)
        return loaded_params
    return None


def save_model(parameters, save_model_path: str, name_model: str, postfix: str):
    concrete_params = jax.tree_map(
        lambda x: x.copy() if hasattr(x, "copy") else x, parameters
    )

    os.makedirs(save_model_path, exist_ok=True)

    model_file = os.path.join(
        save_model_path,
        f"{name_model}_{postfix}.pkl",
    )

    with open(model_file, "wb") as f:
        pickle.dump(concrete_params, f)

    return model_file

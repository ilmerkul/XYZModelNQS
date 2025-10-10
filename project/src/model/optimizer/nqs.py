import jax
import netket as nk
import optax
from jax import numpy as jnp


class NQSOptimizer:
    ADAM: str = "adam"
    ADAM_COS: str = "adam_cos"
    ADAM_EXP: str = "adam_exp"
    SGD_LIN_EXP: str = "sgd_lin_exp"
    SGD_EXP: str = "sgd_exp"
    ADAM_ZERO_PQC: str = "adam_zero_pqc"
    ADAM_PQC: str = "adam_pqc"

    @staticmethod
    def get_optimizer(type: str, lr: float):
        if type == NQSOptimizer.ADAM:
            optimizer = nk.optimizer.Adam(lr)
        elif type == NQSOptimizer.ADAM_COS:
            optimizer = optax.adam(
                learning_rate=optax.cosine_decay_schedule(init_value=lr, decay_steps=10)
            )
        elif type == NQSOptimizer.ADAM_EXP:
            optimizer = optax.adam(
                learning_rate=optax.join_schedules(
                    [
                        optax.constant_schedule(lr * 10),
                        optax.exponential_decay(
                            init_value=lr * 10,
                            transition_steps=40,
                            decay_rate=0.8,
                            end_value=lr,
                        ),
                    ],
                    boundaries=[25],
                )
            )
        elif type == NQSOptimizer.SGD_LIN_EXP:
            optimizer = optax.sgd(
                learning_rate=optax.join_schedules(
                    [
                        optax.linear_schedule(
                            init_value=lr, end_value=lr * 10, transition_steps=10
                        ),
                        optax.constant_schedule(lr * 10),
                        optax.exponential_decay(
                            init_value=lr * 10,
                            transition_steps=35,
                            decay_rate=0.8,
                            end_value=lr / 10,
                        ),
                    ],
                    boundaries=[10, 15],
                )
            )
        elif type == NQSOptimizer.SGD_EXP:
            optimizer = optax.sgd(
                learning_rate=optax.exponential_decay(
                    init_value=lr * 10,
                    transition_steps=10,
                    decay_rate=0.9,
                    transition_begin=50,
                    end_value=lr,
                ),
                momentum=0.9,
                nesterov=True,
            )
        elif type == NQSOptimizer.ADAM_ZERO_PQC:

            def zero_grads():
                def init_fn(_):
                    return ()

                def update_fn(updates, state, params=None):
                    return jax.tree_map(jnp.zeros_like, updates), ()

                return optax.GradientTransformation(init_fn, update_fn)

            def map_nested_fn(fn):
                def map_fn(nested_dict):
                    p = {
                        k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
                        for k, v in nested_dict.items()
                    }
                    return p

                return map_fn

            label_fn = map_nested_fn(lambda k, _: "tr" if k.startswith("tr") else "pqc")
            optimizer = optax.multi_transform(
                {
                    "tr": optax.adam(
                        learning_rate=optax.join_schedules(
                            [
                                optax.constant_schedule(lr * 10),
                                optax.exponential_decay(
                                    init_value=lr * 10,
                                    transition_steps=10,
                                    decay_rate=0.8,
                                    end_value=lr,
                                ),
                            ],
                            boundaries=[25],
                        )
                    ),
                    "pqc": zero_grads(),
                },
                label_fn,
            )
        elif type == NQSOptimizer.ADAM_PQC:
            optimizer = nk.optimizer.Adam(lr)

        return optimizer

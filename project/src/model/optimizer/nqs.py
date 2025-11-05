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
    def get_optimizer(type: str, lr: float, n_iter: int):
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
                            transition_steps=(n_iter - n_iter // 4),
                            decay_rate=0.8,
                            end_value=lr,
                        ),
                    ],
                    boundaries=[n_iter // 4],
                )
            )
        elif type == NQSOptimizer.SGD_LIN_EXP:
            optimizer = optax.sgd(
                learning_rate=optax.join_schedules(
                    [
                        optax.linear_schedule(
                            init_value=lr,
                            end_value=lr * 10,
                            transition_steps=n_iter // 4,
                        ),
                        optax.constant_schedule(lr * 10),
                        optax.exponential_decay(
                            init_value=lr * 10,
                            transition_steps=(n_iter - n_iter // (4 / 3)) // 2,
                            decay_rate=0.8,
                            end_value=lr,
                        ),
                    ],
                    boundaries=[n_iter // 4, n_iter // (4 / 3)],
                ),
                momentum=0.999,
                nesterov=True,
            )
        elif type == NQSOptimizer.SGD_EXP:
            optimizer = optax.sgd(
                learning_rate=optax.exponential_decay(
                    init_value=lr * 10,
                    transition_steps=(n_iter - n_iter // 4) // (4 / 3),
                    decay_rate=0.9,
                    transition_begin=n_iter // 4,
                    end_value=lr,
                ),
                momentum=0.999,
                nesterov=True,
            )
        elif type == NQSOptimizer.ADAM_ZERO_PQC or type == NQSOptimizer.ADAM_PQC:

            def zero_grads():
                def init_fn(_):
                    return ()

                def update_fn(updates, state, params=None):
                    return jax.tree_map(jnp.zeros_like, updates), ()

                return optax.GradientTransformation(init_fn, update_fn)

            def map_nested_fn(fn):
                def map_fn(nested_dict, carry=False):
                    result = {}
                    current_carry = carry
                    for k, v in nested_dict.items():
                        # Определяем метку
                        label = fn(k, current_carry)
                        # Обновляем carry
                        if label == "tr":
                            current_carry = True

                        if isinstance(v, dict):
                            result[k] = map_fn(v, current_carry)
                        else:
                            result[k] = label
                    return result

                return map_fn

            label_fn = map_nested_fn(
                lambda k, carry: "tr" if (k.startswith("tr") or carry) else "pqc"
            )
            if type == NQSOptimizer.ADAM_ZERO_PQC:
                optimizer = optax.multi_transform(
                    {
                        "tr": optax.sgd(
                            learning_rate=optax.join_schedules(
                                [
                                    optax.linear_schedule(
                                        init_value=lr,
                                        end_value=lr * 10,
                                        transition_steps=n_iter // 4,
                                    ),
                                    optax.constant_schedule(lr * 10),
                                    optax.exponential_decay(
                                        init_value=lr * 10,
                                        transition_steps=(n_iter - n_iter // (4 / 3))
                                        // 2,
                                        decay_rate=0.8,
                                        end_value=lr,
                                    ),
                                ],
                                boundaries=[n_iter // 4, n_iter // (4 / 3)],
                            ),
                            momentum=0.999,
                            nesterov=True,
                        ),
                        "pqc": zero_grads(),
                    },
                    label_fn,
                )
            else:
                optimizer = optax.multi_transform(
                    {
                        "pqc": optax.sgd(
                            learning_rate=optax.join_schedules(
                                [
                                    optax.linear_schedule(
                                        init_value=lr,
                                        end_value=lr * 10,
                                        transition_steps=n_iter // 4,
                                    ),
                                    optax.constant_schedule(lr * 10),
                                    optax.exponential_decay(
                                        init_value=lr * 10,
                                        transition_steps=(n_iter - n_iter // (4 / 3))
                                        // 2,
                                        decay_rate=0.8,
                                        end_value=lr,
                                    ),
                                ],
                                boundaries=[n_iter // 4, n_iter // (4 / 3)],
                            ),
                            momentum=0.999,
                            nesterov=True,
                        ),
                        "tr": zero_grads(),
                    },
                    label_fn,
                )
        else:
            raise ValueError()

        optimizer = optax.chain(optax.clip_by_global_norm(max_norm=1.0), optimizer)

        return optimizer

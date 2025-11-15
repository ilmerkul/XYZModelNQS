from dataclasses import dataclass

import jax
import optax
from jax import numpy as jnp

from .interface.type import OptimizerConfig


def global_norm_optimizer(optimizer_func):
    def wrapper(cfg: OptimizerConfig):
        optimizer = optimizer_func(cfg)
        return optax.chain(
            optax.clip_by_global_norm(max_norm=cfg.global_norm), optimizer
        )

    return wrapper


@dataclass
class AdamConfig(OptimizerConfig):
    b1: float
    b2: float
    eps: float


@global_norm_optimizer
def adam(cfg: AdamConfig):
    return optax.adam(learning_rate=cfg.lr, b1=cfg.b1, b2=cfg.b2, eps=cfg.eps)


@dataclass
class AdamExpConfig(OptimizerConfig):
    b1: float
    b2: float
    eps: float


@global_norm_optimizer
def adam_exp(cfg: AdamExpConfig):
    return optax.adam(
        learning_rate=optax.join_schedules(
            [
                optax.constant_schedule(cfg.lr * 10),
                optax.exponential_decay(
                    init_value=cfg.lr * 10,
                    transition_steps=(cfg.n_iter - cfg.n_iter // 4),
                    decay_rate=0.8,
                    end_value=cfg.lr,
                ),
            ],
            boundaries=[cfg.n_iter // 4],
        )
    )


@dataclass
class AdamCosConfig(OptimizerConfig):
    b1: float
    b2: float
    eps: float


@global_norm_optimizer
def adam_cos(cfg: AdamCosConfig):
    return optax.adam(
        learning_rate=optax.cosine_decay_schedule(
            init_value=cfg.lr, decay_steps=cfg.n_iter
        )
    )


@dataclass
class SGDLinExpConfig(OptimizerConfig):
    decay_rate: float
    nesterov: bool
    momentum: float


@global_norm_optimizer
def sgd_lin_exp(cfg: SGDLinExpConfig):
    return optax.sgd(
        learning_rate=optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=cfg.lr,
                    end_value=cfg.lr * 10,
                    transition_steps=cfg.n_iter // 4,
                ),
                optax.constant_schedule(cfg.lr * 10),
                optax.exponential_decay(
                    init_value=cfg.lr * 10,
                    transition_steps=(cfg.n_iter - cfg.n_iter // (4 / 3)) // 2,
                    decay_rate=0.8,
                    end_value=cfg.lr,
                ),
            ],
            boundaries=[cfg.n_iter // 4, cfg.n_iter // (4 / 3)],
        ),
        momentum=cfg.momentum,
        nesterov=cfg.nesterov,
    )


@dataclass
class SGDExpConfig(OptimizerConfig):
    decay_rate: float
    nesterov: bool
    momentum: float


@global_norm_optimizer
def sgd_exp(cfg: SGDExpConfig):
    return optax.sgd(
        learning_rate=optax.exponential_decay(
            init_value=cfg.lr * 10,
            transition_steps=(cfg.n_iter - cfg.n_iter // 4) // (4 / 3),
            decay_rate=cfg.decay_rate,
            transition_begin=cfg.n_iter // 4,
            end_value=cfg.lr,
        ),
        momentum=cfg.momentum,
        nesterov=cfg.nesterov,
    )


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


@dataclass
class AdamZeroPqcTrConfig(OptimizerConfig):
    decay_rate: float
    nesterov: bool
    momentum: float


@global_norm_optimizer
def adam_zero_pqc(cfg: AdamZeroPqcTrConfig):
    return optax.multi_transform(
        {
            "tr": optax.sgd(
                learning_rate=optax.join_schedules(
                    [
                        optax.linear_schedule(
                            init_value=cfg.lr,
                            end_value=cfg.lr * 10,
                            transition_steps=cfg.n_iter // 4,
                        ),
                        optax.constant_schedule(cfg.lr * 10),
                        optax.exponential_decay(
                            init_value=cfg.lr * 10,
                            transition_steps=(cfg.n_iter - cfg.n_iter // (4 / 3)) // 2,
                            decay_rate=cfg.decay_rate,
                            end_value=cfg.lr,
                        ),
                    ],
                    boundaries=[cfg.n_iter // 4, cfg.n_iter // (4 / 3)],
                ),
                momentum=cfg.momentum,
                nesterov=cfg.nesterov,
            ),
            "pqc": zero_grads(),
        },
        label_fn,
    )


@global_norm_optimizer
def adam_zero_tr(cfg: AdamZeroPqcTrConfig):
    return optax.multi_transform(
        {
            "pqc": optax.sgd(
                learning_rate=optax.join_schedules(
                    [
                        optax.linear_schedule(
                            init_value=cfg.lr,
                            end_value=cfg.lr * 10,
                            transition_steps=cfg.n_iter // 4,
                        ),
                        optax.constant_schedule(cfg.lr * 10),
                        optax.exponential_decay(
                            init_value=cfg.lr * 10,
                            transition_steps=(cfg.n_iter - cfg.n_iter // (4 / 3)) // 2,
                            decay_rate=cfg.decay_rate,
                            end_value=cfg.lr,
                        ),
                    ],
                    boundaries=[cfg.n_iter // 4, cfg.n_iter // (4 / 3)],
                ),
                momentum=cfg.momentum,
                nesterov=cfg.nesterov,
            ),
            "tr": zero_grads(),
        },
        label_fn,
    )

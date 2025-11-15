from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import netket as nk
from flax import linen as nn
from netket.jax import dtype_real
from netket.jax.sharding import shard_along_axis
from netket.sampler.base import Sampler, SamplerState
from netket.utils.types import PyTree

from .interface.type import SamplerConfig


@dataclass
class TransformerSamplerConfig(SamplerConfig):
    sample_ratio: float


class TransformerSamplerState(SamplerState):
    rng: jnp.ndarray
    σ: jnp.ndarray
    kv_cache: jnp.ndarray
    log_prob: jnp.ndarray

    def __init__(
        self,
        rng: jnp.ndarray,
        σ: jnp.ndarray,
        kv_cache: jnp.ndarray,
        log_prob: jnp.ndarray,
    ):
        self.rng = rng
        self.σ = σ
        self.kv_cache = kv_cache
        self.log_prob = log_prob
        super().__init__()


class TransformerSampler(Sampler):
    def __init__(self, hilbert, graph, cfg: TransformerSamplerConfig):
        super().__init__(hilbert, machine_pow=cfg.machine_pow, dtype=cfg.dtype)

        self.sample_ratio = cfg.sample_ratio

    @partial(jax.jit, static_argnames=("machine"))
    def _init_state(self, machine, parameters, key):
        rng, key = jax.random.split(key)
        σ = jnp.zeros((self.n_batches, self.hilbert.size), dtype=self.dtype)
        σ = shard_along_axis(σ, axis=0)

        output_dtype = jax.eval_shape(machine.apply, parameters, σ).dtype
        log_prob = jnp.full((self.n_batches,), -jnp.inf, dtype=dtype_real(output_dtype))
        log_prob = shard_along_axis(log_prob, axis=0)

        state = TransformerSamplerState(rng=rng, σ=σ, log_prob=log_prob, kv_cache=None)
        return state

    @partial(jax.jit, static_argnames=("machine"))
    def _reset(self, machine, parameters, state):
        rng, key = jax.random.split(state.rng)
        return self._init_state(machine, parameters, rng)

    @partial(
        jax.jit, static_argnames=("machine", "chain_length", "return_log_probabilities")
    )
    def _sample_chain(
        self,
        machine: nn.Module,
        parameters: PyTree,
        state: TransformerSamplerState,
        chain_length: int,
        return_log_probabilities: bool = False,
    ):
        def loop_body(s):
            s_key, key = jax.random.split(s["key"])

            batch_dim = int(chain_length * 0.7)

            σ, log_prob = machine.apply(
                parameters, None, generate=True, batch_dim=batch_dim  # , key=key
            )

            s["key"] = s_key
            s["σ"] = jnp.concat([s["σ"][: chain_length - batch_dim, :], σ], axis=0)
            s["log_prob"] = jnp.concat(
                [s["log_prob"][: chain_length - batch_dim], log_prob], axis=0
            )

            return s

        s = {
            "key": state.rng,
            "σ": state.σ,
            "log_prob": state.log_prob,
            "kv_cache": state.kv_cache,
        }
        s = loop_body(s)

        new_state = state.replace(
            rng=s["key"], σ=s["σ"], log_prob=s["log_prob"], kv_cache=s["kv_cache"]
        )

        σ = jnp.expand_dims(new_state.σ, axis=0)
        log_prob = jnp.expand_dims(new_state.log_prob, axis=0)

        if return_log_probabilities:
            return (σ, log_prob), new_state
        else:
            return σ, new_state

    def __repr__(self):
        return f"TransformerSampler(hilbert={self.hilbert})"


@dataclass
class MetropolisSamplerConfig(SamplerConfig):
    d_max: int
    reset_chains: bool


class MetropolisSampler(nk.sampler.MetropolisSampler):
    def __init__(self, hilbert, graph, cfg: MetropolisSamplerConfig):
        rule = nk.sampler.rules.ExchangeRule(graph=graph, d_max=cfg.d_max)
        super().__init__(
            hilbert,
            rule,
            sweep_size=None,
            reset_chains=cfg.reset_chains,
            n_chains=cfg.n_chains,
            n_chains_per_rank=None,
            chunk_size=None,
            machine_pow=cfg.machine_pow,
            dtype=cfg.dtype,
        )
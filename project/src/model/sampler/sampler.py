from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import netket as nk
from flax import linen as nn
from netket.jax import apply_chunked, dtype_real
from netket.jax.sharding import shard_along_axis
from netket.sampler.base import SamplerState, Sampler
from netket.utils.struct import dataclass
from netket.utils.types import PyTree
from src.model.NN import Transformer, TransformerConfig


class TransformerSamplerState(SamplerState):
    rng: jnp.ndarray
    σ: jnp.ndarray
    kv_cache: jnp.ndarray
    log_prob: jnp.ndarray


class TransformerSampler(Sampler):
    def __init__(self, hilbert, machine_pow=2, dtype=float):
        super().__init__(hilbert, machine_pow=machine_pow, dtype=dtype)

    @partial(jax.jit, static_argnames=("machine"))
    def _init_state(self, machine, parameters, key):
        rng, key = jax.random.split(key)
        σ = jnp.zeros((self.n_batches, self.hilbert.size), dtype=self.dtype)
        σ = shard_along_axis(σ, axis=0)

        output_dtype = jax.eval_shape(machine.apply, parameters, σ).dtype
        log_prob = jnp.full((self.n_batches,), -jnp.inf, dtype=dtype_real(output_dtype))
        log_prob = shard_along_axis(log_prob, axis=0)

        return TransformerSamplerState(rng=rng, σ=σ, log_prob=log_prob, kv_cache=None)

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

            σ, log_prob = machine.apply(
                parameters, None, generate=True, n_chains=2000, key=key
            )

            s["key"] = s_key
            s["σ"] = σ
            s["log_prob"] = log_prob

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

        if return_log_probabilities:
            return (new_state.σ, new_state.log_prob), new_state
        else:
            return new_state.σ, new_state

    def __repr__(self):
        return f"TransformerSampler(hilbert={self.hilbert})"


if __name__ == "__main__":
    n = 11
    tr = Transformer(TransformerConfig(training=True, symm=True, length=n))

    key = jax.random.PRNGKey(42)
    init_rngs = {"params": key, "dropout": key}

    # Initialize with dummy input
    dummy_input = jnp.zeros((16, n), dtype=jnp.int32)
    params = tr.init(init_rngs, dummy_input)

    hilbert = nk.hilbert.Spin(N=n, s=1 / 2)
    sampler = TransformerSampler(hilbert)

    # Initialize sampler state
    state = sampler.init_state(tr, params, seed=42)

    # Generate samples
    samples, state = sampler.sample(tr, params, state=state, chain_length=100)
    print("Generated samples shape:", samples.shape)  # (100, 16, 11)

import netket as nk
from netket.sampler import SamplerState
import jax.numpy as jnp
from netket.utils.struct import dataclass
import jax
from netket.utils.types import PyTree
from flax import linen as nn
from typing import Any
from .transformer import Transformer, TransformerConfig


@dataclass
class TransformerSamplerState(SamplerState):
    key: jax.random.PRNGKey
    σ: jnp.ndarray
    kv_cache: Any = None  # For storing attention cache during generation


class TransformerSampler(nk.sampler.Sampler):
    def __init__(self, hilbert, machine_pow=2, dtype=float):
        super().__init__(hilbert, machine_pow=machine_pow, dtype=dtype)

    def _init_state(self, machine, parameters, seed):
        key = jax.random.PRNGKey(seed[0])
        # Initialize with all zeros (start tokens)
        σ = jnp.zeros((self.n_chains, self.hilbert.size), dtype=jnp.int32)
        return TransformerSamplerState(key=key, σ=σ)

    def _reset(self, machine, parameters, state):
        # Reset to initial state
        return self._init_state(machine, parameters,
                                jax.random.split(state.key)[0])

    def _sample_chain(self, machine, parameters, state, chain_length):
        def step(carry, _):
            state = carry
            key, subkey = jax.random.split(state.key)

            # Generate next tokens using the transformer
            logits = machine.apply(
                parameters,
                None,
                generate=True,
                n_chains=2000
            )

            # Sample from logits (assuming binary spins: -1 or 1)
            # new_spins = jax.random.bernoulli(subkey, logits) * 2 - 1
            # new_σ = state.σ.at[:, -1].set((logits > 0).astype(jnp.int32))

            #new_state = TransformerSamplerState(key=key, σ=new_σ)
            return logits, state

        # Run the chain
        #final_state, samples = jax.lax.scan(
        #    step, state, None, length=chain_length)
        samples = step(state, None)

        return samples

    @property
    def n_chains(self) -> int:
        return 16  # Match your transformer's batch size

    def __repr__(self):
        return f"TransformerSampler(hilbert={self.hilbert})"


if __name__ == "__main__":
    n = 11
    tr = Transformer(TransformerConfig(training=True, symm=True, length=n))

    key = jax.random.PRNGKey(42)
    init_rngs = {'params': key, 'dropout': key}

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
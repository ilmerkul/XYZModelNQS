import netket as nk
from jax import numpy as jnp
from netket.utils import timing
from netket.vqs.mc.mc_state.state import check_chunk_size, compute_chain_length


@timing.timed
def sample(
    self,
    *,
    chain_length: int | None = None,
    n_samples: int | None = None,
    n_discard_per_chain: int | None = None,
) -> jnp.ndarray:
    """
    Sample a certain number of configurations.

    If one among chain_length or n_samples is defined, that number of samples
    are generated. Otherwise the value set internally is used.

    Args:
        chain_length: The length of the markov chains.
        n_samples: The total number of samples across all MPI ranks.
        n_discard_per_chain: Number of discarded samples at the beginning of the markov chain.
    """

    if n_samples is None and chain_length is None:
        chain_length = self.chain_length
    else:
        if chain_length is not None and n_samples is not None:
            raise ValueError("Cannot specify both `chain_length` and `n_samples`.")
        elif chain_length is None:
            chain_length = compute_chain_length(self.sampler.n_chains, n_samples)

        if self.chunk_size is not None:
            check_chunk_size(chain_length * self.sampler.n_chains, self.chunk_size)

    if n_discard_per_chain is None:
        n_discard_per_chain = self.n_discard_per_chain

    # Store the previous sampler state, for serialization purposes
    self._sampler_state_previous = self.sampler_state

    self.sampler_state = self.sampler.reset(
        self._sampler_model, self._sampler_variables, self.sampler_state
    )

    if self.n_discard_per_chain > 0:
        with timing.timed_scope("sampling n_discarded samples") as timer:
            _, self.sampler_state = self.sampler.sample(
                self._sampler_model,
                self.variables,
                state=self.sampler_state,
                chain_length=n_discard_per_chain,
            )
            # This won't actually block unless we are really timing
            timer.block_until_ready(_)

    self._samples, self.sampler_state = self.sampler.sample(
        self._sampler_model,
        self._sampler_variables,
        state=self.sampler_state,
        chain_length=chain_length,
    )
    return self._samples


def patch_mc_state():
    nk.vqs._mc.mc_state.state.MCState.sample = sample

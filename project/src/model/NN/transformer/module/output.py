from typing import Tuple

import jax
import jax.random as jrnd
from flax import linen as nn
from jax import numpy as jnp

from .config import TransformerConfig


class OutputHead(nn.Module):
    config: TransformerConfig

    def setup(self):
        features_out = self.config.n_state
        if not self.config.autoregressive:
            features_out = 1

        self.dense_out = nn.Dense(
            features=features_out,
            use_bias=self.config.use_bias,
            name="tr_dense_out",
            dtype=self.config.dtype,
            kernel_init=self.config.default_kernel_init,
            bias_init=self.config.default_bias_init,
        )

        self.phase_dense_out = nn.Dense(
            features=1,
            use_bias=self.config.use_bias,
            name="tr_phase_dense_out",
            dtype=self.config.dtype,
            kernel_init=self.config.default_kernel_init,
            bias_init=self.config.default_bias_init,
        )

        self.norm = nn.LayerNorm(dtype=self.config.dtype, name="tr_output_norm")

    def get_prob(self, x_row, token):
        return x_row[token]

    def sample_one(self, p_row, key):
        return jrnd.choice(key, a=jnp.array(range(self.config.n_state)), p=p_row)

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        encoder_input_tokens: jax.Array = None,
        generate: bool = False,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        batch_size = x.shape[0]
        rng_key = self.make_rng("output")

        x = self.norm(x)

        if generate:
            x = x[:, -1, :]
            phase = jnp.pi * nn.tanh(self.phase_dense_out(x)).squeeze(-1)
            prob_x = self.dense_out(x)

            prob_x = nn.softmax(prob_x, axis=-1)
            keys = jrnd.split(rng_key, batch_size)
            sample = jax.vmap(self.sample_one)(prob_x, keys)
            prob = jax.vmap(self.get_prob)(prob_x, sample)
            prob = jnp.log(prob + self.config.eps)
        elif self.config.autoregressive:
            x = x[:, -1, :]
            phase = jnp.pi * nn.tanh(self.phase_dense_out(x)).squeeze(-1)
            prob_x = self.dense_out(x)

            prob_x = nn.softmax(prob_x, axis=-1)
            prob = jax.vmap(self.get_prob)(prob_x, encoder_input_tokens)
            prob = jnp.log(prob + self.config.eps)
            sample = None
        else:
            x = jnp.mean(x, axis=1)
            phase = jnp.pi * nn.tanh(self.phase_dense_out(x)).squeeze(-1)
            prob_x = self.dense_out(x)

            prob = jnp.log(nn.sigmoid(prob_x.squeeze(-1)) + self.config.eps)
            sample = None

        return prob, phase, sample

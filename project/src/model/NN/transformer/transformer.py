import os
import pathlib
import sys
from typing import List, Tuple

import jax
import jax.random as jrnd
from flax import linen as nn
from jax import numpy as jnp

project_path = pathlib.Path(os.getcwd())

sys.path.append(project_path.as_posix())
from src.model.NN.transformer.module import (
    Embed,
    Encoder,
    KVCache,
    MultiheadAttention,
    OutputHead,
    PosEmbType,
    TransformerConfig,
)
from src.model.struct import ChainConfig


class Transformer(nn.Module):
    config: TransformerConfig

    def setup(self) -> None:
        self.embed = Embed(config=self.config)

        self.enc = Encoder(config=self.config)

        self.out_head = OutputHead(config=self.config)

        self.padding_mask = MultiheadAttention.causal_mask(
            jnp.ones((1, self.config.chain.n + 1))
        )

    def enc_iter(
        self,
        x: jax.Array,
        encoder_mask: jax.Array,
        encoder_input_tokens: jax.Array = None,
        kv_cache: KVCache = None,
        n_iter: int = None,
    ) -> Tuple[jax.Array, KVCache]:
        if n_iter is not None:
            x = x[:, : (n_iter + 1), :]
            encoder_mask = encoder_mask[:, :, : (n_iter + 1), : (n_iter + 1)]

        x, kv_cache = self.enc(
            x=x,
            encoder_mask=encoder_mask,
            kv_cache=kv_cache,
            n_iter=n_iter,
        )
        return self.out_head(x=x, encoder_input_tokens=encoder_input_tokens), kv_cache

    def _generate_forward(
        self,
        generate_batch_dim: int,
    ) -> Tuple[jax.Array, jax.Array]:
        cfg = self.config

        state_embeddings = [
            jnp.zeros((generate_batch_dim, self.config.chain.n + 1, cfg.features))
        ]

        kv_cache = None
        input_embedding = state_embeddings[0]
        prob = jnp.ones((generate_batch_dim,))
        samples = jnp.zeros((generate_batch_dim, cfg.chain.n))
        for n_iter in range(cfg.chain.n):

            (prob_iter, phase, sample), kv_cache = self.enc_iter(
                x=input_embedding,
                encoder_mask=self.padding_mask,
                kv_cache=kv_cache,
                n_iter=n_iter,
            )

            samples = samples.at[:, n_iter].set(sample)
            new_input_embedding, _, _ = self.embed(generate_batch_dim, sample)
            input_embedding.at[:, (n_iter + 1), :].set(new_input_embedding)
            prob *= prob_iter

        return samples, prob + 1j * phase

    def _autoregressive_forward(
        self,
        state_embeddings: List[jax.Array],
        state_σ: jax.Array,
    ) -> jax.Array:
        cfg = self.config
        rng_key = self.make_rng("symmetry")

        batch_dim = state_σ.shape[0]

        def enc_iter(input_embedding, state_σ):
            prob = jnp.ones((batch_dim,))
            kv_cache = None
            for n_iter in range(cfg.chain.n + 1):
                (prob_iter, phase, _), kv_cache = self.enc_iter(
                    x=input_embedding,
                    encoder_mask=self.padding_mask,
                    encoder_input_tokens=state_σ[:, n_iter],
                    kv_cache=kv_cache,
                    n_iter=n_iter,
                )
                prob *= prob_iter
            return prob, phase

        key_iter, key_choice = jrnd.split(rng_key)
        use_single = jrnd.uniform(key_iter) > cfg.inverse_iter_rate

        single_input_embedding = state_embeddings[
            jrnd.choice(key_choice, jnp.arange(len(state_embeddings)))
        ]
        single_prob, single_phase = enc_iter(
            single_input_embedding,
            state_σ,
        )

        avg_prob = jnp.zeros((batch_dim,))
        avg_phase = jnp.zeros((batch_dim,))
        for input_embedding in state_embeddings:
            prob_iter, phase_iter = enc_iter(
                input_embedding,
                state_σ,
            )
            avg_prob += prob_iter
            avg_phase += phase_iter

        avg_prob /= len(state_embeddings)
        avg_phase /= len(state_embeddings)

        if not cfg.training:
            prob = avg_prob
            phase = avg_phase
        else:
            prob = jnp.where(use_single, single_prob, avg_prob)
            phase = jnp.where(use_single, single_phase, avg_phase)

        return prob + 1j * phase

    def _notautoregressive_forward(self, state_embeddings: jax.Array) -> jax.Array:
        cfg = self.config
        rng_key = self.make_rng("symmetry")

        def single_forward(x, mask):
            (prob, phase, _), _ = self.enc_iter(x=x, encoder_mask=mask)
            return prob, phase

        batched_forward = jax.vmap(single_forward, in_axes=(0, None))
        prob_iters, phase_iters = batched_forward(state_embeddings, self.padding_mask)

        if not cfg.training:
            prob = jnp.mean(prob_iters, axis=0)
            phase = jnp.mean(phase_iters, axis=0)
        else:
            key_iter, key_choice = jrnd.split(rng_key)
            use_single = jrnd.uniform(key_iter) > cfg.inverse_iter_rate

            idx = jrnd.choice(key_choice, jnp.arange(len(state_embeddings)))
            prob = jnp.where(use_single, prob_iters[idx], jnp.mean(prob_iters, axis=0))
            phase = jnp.where(
                use_single, phase_iters[idx], jnp.mean(phase_iters, axis=0)
            )

        return prob + 1j * phase

    @nn.compact
    def __call__(
        self,
        σ: jax.Array = None,
        generate: bool = None,
        generate_batch_dim: int = None,
    ) -> jax.Array:
        cfg = self.config

        if σ is None and (generate is None or generate_batch_dim is None):
            raise ValueError("σ is needed for model")

        if generate:
            return self._generate_forward(
                generate_batch_dim=generate_batch_dim,
            )

        state_embeddings, state_σ = self.embed(σ)
        state_embeddings = jnp.stack(state_embeddings, axis=0)

        if cfg.autoregressive:
            log_psi = self._autoregressive_forward(
                state_embeddings=state_embeddings,
                state_σ=state_σ,
            )
        else:
            log_psi = self._notautoregressive_forward(
                state_embeddings=state_embeddings,
            )

        return log_psi


if __name__ == "__main__":
    chain_cfg = ChainConfig(
        n=20,
        j=-1,
        h=0.0,
        lam=1,
        gamma=0,
        spin=1 / 2,
    )
    transformer_config = TransformerConfig(
        chain=chain_cfg,
        use_bias=True,
        use_dropout=False,
        dropout_rate=0.1,
        inverse_iter_rate=0.3,
        training=True,
        seed=42,
        autoregressive=False,
        dtype=jnp.float64,
        embed_concat=False,
        pos_embed=PosEmbType.ROTARY,
        eps=1e-10,
    )
    tr = Transformer(transformer_config)

    key = jrnd.PRNGKey(42)
    input = jnp.ones(16 * chain_cfg.n)
    input = input.at[jrnd.randint(key, (16, chain_cfg.n), 0, 16 * chain_cfg.n)].set(-1)
    input = input.reshape(16, chain_cfg.n)
    print(input)
    init_rngs = {"params": key, "dropout": key}
    params = tr.init(init_rngs, input)
    output = tr.apply(params, input, generate=False, rngs={"dropout": key})
    print(output)
    print(output.shape)

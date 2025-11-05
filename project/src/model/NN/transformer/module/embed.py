from typing import List, Tuple

import jax
from flax import linen as nn
from jax import numpy as jnp
from src.model.struct import SymmetryChain

from .config import PosEmbType, TransformerConfig


class LearnedPositionalEmbedding(nn.Module):
    num_embeddings: int
    features: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, batch_size: int) -> jax.Array:
        pos = jnp.arange(self.num_embeddings)
        pos = jnp.repeat(pos[None, :], batch_size, axis=0)
        return nn.Embed(
            num_embeddings=self.num_embeddings,
            features=self.features,
            dtype=self.dtype,
            name=f"tr_leran_pos_emb",
        )(pos)


class Embed(nn.Module):
    config: TransformerConfig

    def setup(self) -> None:
        self.features_2 = self.config.features
        if self.config.embed_concat:
            self.features_2 //= 2

        self.embed = nn.Embed(
            num_embeddings=self.config.n_state + 1,
            features=self.features_2,
            name="tr_embed",
            dtype=self.config.dtype,
        )

    def abs_embedding(self, features_2: int) -> jax.Array:
        cfg = self.config

        base = 10000.0
        pos_embedding = jnp.zeros((cfg.chain.n, features_2))
        position = jnp.arange(0, cfg.chain.n)[:, None]
        div_term = jnp.exp(
            (jnp.arange(0, features_2, 2) * -(jnp.log(base) / features_2))
        )
        pos_embedding.at[:, 0::2].set(jnp.sin(position * div_term))
        pos_embedding.at[:, 1::2].set(jnp.cos(position * div_term))

        return pos_embedding

    def get_embedding(self, batch_size: int, input_embedding: jax.Array) -> jax.Array:
        if self.config.pos_embed == PosEmbType.LEARN_ABSOLUTE:
            pos_embedding = LearnedPositionalEmbedding(
                num_embeddings=self.config.chain.n,
                features=self.features_2,
                dtype=self.config.dtype,
            )(batch_size)
        elif self.config.pos_embed == PosEmbType.ABSOLUTE:
            pos_embedding = self.abs_embedding(self.features_2)
        elif self.config.pos_embed == PosEmbType.ROTARY:
            pos_embedding = jnp.zeros_like(input_embedding, dtype=self.config.dtype)
        elif self.config.pos_embed == PosEmbType.ZERO:
            pos_embedding = jnp.zeros_like(input_embedding, dtype=self.config.dtype)
        else:
            raise ValueError("pos_embed error")

        if self.config.embed_concat:
            input_embedding = jnp.concat([pos_embedding, input_embedding], axis=-1)
        else:
            input_embedding += pos_embedding

        return input_embedding

    def _get_embedding(
        self, first_embedding: jax.Array, state_embedding: jax.Array
    ) -> jax.Array:
        batch_size = first_embedding.shape[0]
        state_embedding = jnp.concat([first_embedding, state_embedding], axis=1)
        state_embedding = self.get_embedding(batch_size, state_embedding)
        return state_embedding

    def get_mirror_input(
        self, first_embedding: jax.Array, state_embedding: jax.Array
    ) -> List[jax.Array]:
        state_embeddings = []

        if SymmetryChain.MIRROR in self.config.chain.symmetries:
            state_embeddings.append(
                self._get_embedding(
                    first_embedding=first_embedding,
                    state_embedding=jnp.flip(state_embedding, axis=1),
                )
            )

        return state_embeddings

    def get_z2_emb_input(
        self, first_embedding: jax.Array, state_embedding: jax.Array
    ) -> List[jax.Array]:
        state_embeddings = []

        if SymmetryChain.Z2_EMB in self.config.chain.symmetries:
            state_embeddings.append(
                self._get_embedding(
                    first_embedding=first_embedding, state_embedding=state_embedding
                )
            )

            state_embeddings.extend(
                self.get_mirror_input(
                    first_embedding=first_embedding, state_embedding=state_embedding
                )
            )

        return state_embeddings

    def get_z2_input(
        self, first_embedding: jax.Array, state_embedding: jax.Array
    ) -> List[jax.Array]:
        state_embeddings = []

        if SymmetryChain.Z2 in self.config.chain.symmetries:
            state_embeddings.append(
                self._get_embedding(
                    first_embedding=first_embedding, state_embedding=state_embedding
                )
            )

            state_embeddings.extend(
                self.get_mirror_input(
                    first_embedding=first_embedding, state_embedding=state_embedding
                )
            )

        return state_embeddings

    def _spin2state(self, σ: jax.Array) -> jax.Array:
        return ((σ + self.config.chain.spin + 0.5) / 2).astype(jnp.int32)

    def get_input(self, σ: jax.Array) -> Tuple[List[jax.Array], jax.Array]:
        batch_size = σ.shape[0]
        state_embeddings = []

        input_state = σ
        if SymmetryChain.Z2_EMB in self.config.chain.symmetries:
            input_state = jnp.abs(input_state)

        states = self._spin2state(input_state)
        states_z2 = self._spin2state(-1 * σ)

        first_embedding = self.embed(jnp.full((batch_size, 1), self.config.n_state))
        state_embedding = self.embed(states)
        state_embedding_z2 = self.embed(states_z2)
        if SymmetryChain.Z2_EMB in self.config.chain.symmetries:
            state_embedding = jnp.where(
                σ[..., None] > 0, state_embedding, -1 * state_embedding
            )

        state_embeddings.append(
            self._get_embedding(
                first_embedding=first_embedding, state_embedding=state_embedding
            )
        )

        state_embeddings.extend(
            self.get_mirror_input(
                first_embedding=first_embedding, state_embedding=state_embedding
            )
        )

        state_embeddings.extend(
            self.get_z2_emb_input(
                first_embedding=first_embedding, state_embedding=-1 * state_embedding
            )
        )

        state_embeddings.extend(
            self.get_z2_input(
                first_embedding=first_embedding, state_embedding=state_embedding_z2
            )
        )

        return state_embeddings, self._spin2state(σ)

    @nn.compact
    def __call__(self, sample) -> jax.Array:
        return self.get_input(sample)

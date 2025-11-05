from typing import Tuple

import jax
from flax import linen as nn
from jax import lax
from jax import numpy as jnp

from .cache import KVCache
from .config import TransformerConfig
from .embed import PosEmbType


class MultiheadAttention(nn.Module):
    config: TransformerConfig
    n_layer: int
    decode: bool = False

    def setup(self):
        cfg = self.config
        self.query_dense = nn.Dense(
            features=cfg.features,
            use_bias=cfg.use_bias,
            name=f"tr_multiheadattn_query_dense_{self.n_layer}",
            dtype=cfg.dtype,
            kernel_init=cfg.default_kernel_init,
            bias_init=cfg.default_bias_init,
        )
        self.key_dense = nn.Dense(
            features=cfg.features,
            use_bias=cfg.use_bias,
            name=f"tr_multiheadattn_key_dense_{self.n_layer}",
            dtype=cfg.dtype,
            kernel_init=cfg.default_kernel_init,
            bias_init=cfg.default_bias_init,
        )
        self.value_dense = nn.Dense(
            features=cfg.features,
            use_bias=cfg.use_bias,
            name=f"tr_multiheadattn_value_dense_{self.n_layer}",
            dtype=cfg.dtype,
            kernel_init=cfg.default_kernel_init,
            bias_init=cfg.default_bias_init,
        )
        self.logits_dense = nn.Dense(
            features=cfg.features,
            use_bias=cfg.use_bias,
            name=f"tr_multiheadattn_logits_dense_{self.n_layer}",
            dtype=cfg.dtype,
            kernel_init=cfg.default_kernel_init,
            bias_init=cfg.default_bias_init,
        )
        self.dropout = nn.Dropout(
            rate=cfg.dropout_rate,
            deterministic=not cfg.training,
            name=f"tr_multiheadattn_dropout_{self.n_layer}",
        )

    @nn.compact
    def __call__(
        self,
        kv: jax.Array,
        q: jax.Array,
        mask: jax.Array,
        kv_cache: KVCache,
        n_iter: int,
    ) -> Tuple[jax.Array, KVCache]:
        cfg = self.config
        query = self.query_dense(q)

        if cfg.autoregressive:
            key = self.key_dense(kv[:, n_iter, :])
            value = self.value_dense(kv[:, n_iter, :])
        else:
            key = self.key_dense(kv)
            value = self.value_dense(kv)

        if cfg.autoregressive:
            kv_cache.key = kv_cache.key.at[:, self.n_layer, n_iter, :].set(key)
            kv_cache.value = kv_cache.value.at[:, self.n_layer, n_iter, :].set(value)

        batch_size = query.shape[0]

        if cfg.autoregressive:
            key = kv_cache.key[:, self.n_layer, : (n_iter + 1), :]
            value = kv_cache.value[:, self.n_layer, : (n_iter + 1), :]

        # query = nn.LayerNorm()(query)
        # key = nn.LayerNorm()(key)
        # value = nn.LayerNorm()(value)

        assert cfg.features % cfg.num_heads == 0
        head_dim = cfg.features // cfg.num_heads

        query = query.reshape(batch_size, -1, cfg.num_heads, head_dim)
        key = key.reshape(batch_size, -1, cfg.num_heads, head_dim)
        value = value.reshape(batch_size, -1, cfg.num_heads, head_dim)

        logits = self.scaled_dot_product_attention(
            key, query, value, mask, cfg.dtype, cfg.pos_embed
        )

        logits = logits.reshape(batch_size, -1, cfg.features)

        logits = self.logits_dense(logits)
        if self.config.use_dropout:
            logits = self.dropout(logits)

        return logits, kv_cache

    def scaled_dot_product_attention(
        self,
        key: jax.Array,
        query: jax.Array,
        value: jax.Array,
        mask: jax.Array,
        dtype,
        pos_embed: str,
    ):
        """Matmul
        query: [batch, q_length, num_heads, qk_depth_per_head]
        key: [batch, kv_length, num_heads, qk_depth_per_head]
        -> qk: [batch, num_heads, q_length, kv_length]
        """
        length_q = query.shape[1]
        length_k = key.shape[1]
        hid_head = query.shape[3]

        if pos_embed == PosEmbType.ROTARY:
            pos_embedding = self.rope_embedding(length_q, hid_head)
            query *= pos_embedding
            key *= pos_embedding

        attention_weights = jnp.einsum("bqhd,bkhd->bhqk", query, key)

        if pos_embed == PosEmbType.RELATIVE:
            attention_weights_rel_pos = self.rel_embedding(
                hid_head, length_q, length_k, dtype, query
            )
            attention_weights += attention_weights_rel_pos

        d_k = query.shape[-1]
        attention_weights /= jnp.sqrt(d_k)
        if mask is not None:
            attention_weights = jnp.where(mask, attention_weights, jnp.finfo(dtype).min)

        attention_weights = nn.softmax(attention_weights, axis=-1).astype(dtype)
        """ Matmul
            qk: [batch, num_heads, q_length, kv_length]
            value: [batch, kv_length, num_heads, v_depth_per_head]
            -> Return: [batch, length, num_heads, v_depth_per_head]
        """

        return jnp.einsum("bhqk,bkhd->bqhd", attention_weights, value)

    def rope_embedding(self, length: int, features: int) -> Tuple[jax.Array, jax.Array]:
        base = 10000.0
        inv_freq = 1.0 / (base ** (jnp.arange(0, features, 2) / features))
        sinusoid_inp = jnp.einsum("i,j->ij", jnp.arange(length), inv_freq)

        sin = jnp.sin(sinusoid_inp)
        if features % 2 != 0:
            sinusoid_inp = sinusoid_inp[:, :-1]
        cos = jnp.cos(sinusoid_inp)

        emb = jnp.concat([sin, cos], axis=1)[:, None, :]

        if length > 1:
            emb_symmetric = (emb[1:] + emb[1:][::-1]) / 2.0
            emb = jnp.concatenate([emb[:1], emb_symmetric], axis=0)

        return emb

    def rel_embedding(
        self, hid_head: int, length_q: int, length_k: int, dtype, query: jax.Array
    ) -> jax.Array:
        range_vec_q = jnp.arange(length_q)
        range_vec_k = jnp.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat += length_q

        embeddings = nn.Embed(
            num_embeddings=2 * length_q + 1,
            features=hid_head,
            name="tr_rel_pos",
            dtype=dtype,
        )(distance_mat.flatten())

        embeddings = nn.Dense(
            features=hid_head,
            use_bias=False,
            name="tr_rel_pos_dense",
            dtype=dtype,
            kernel_init=self.config.default_kernel_init,
            bias_init=self.config.default_bias_init,
        )(embeddings)

        # [q_length, kv_length, qk_depth_per_head]
        embeddings = embeddings.reshape(length_q, length_k, hid_head)

        attention_weights_rel_pos = jnp.einsum("bqhd,qkd->bhqk", query, embeddings)

        return attention_weights_rel_pos

    @classmethod
    def attention_mask(cls, input_tokens):
        mask = jnp.multiply(input_tokens[:, :, None], input_tokens[:, None, :])
        mask = mask[:, None, ...]
        mask = lax.select(mask > 0, jnp.full(mask.shape, 1), jnp.full(mask.shape, 0))

        return mask

    @classmethod
    def causal_mask(cls, input_tokens):
        # mask = cls.attention_mask(input_tokens)
        mask = jnp.full((1, 1, input_tokens.shape[1], input_tokens.shape[1]), 1)
        # mask -= jnp.triu(mask, k=1)
        return mask


class PhysicsInformedAttention(nn.Module):
    config: TransformerConfig
    n_layer: int
    decode: bool = False

    def setup(self):
        cfg = self.config
        # Project to spin-aware features
        self.query_dense = nn.Dense(
            features=cfg.features,
            use_bias=cfg.use_bias,
            dtype=cfg.dtype,
            kernel_init=cfg.default_kernel_init,
            name=f"tr_physatt_query_dense_{self.n_layer}",
        )
        self.key_dense = nn.Dense(
            features=cfg.features,
            use_bias=cfg.use_bias,
            dtype=cfg.dtype,
            kernel_init=cfg.default_kernel_init,
            name=f"tr_physatt_key_dense_{self.n_layer}",
        )
        self.value_dense = nn.Dense(
            features=cfg.features,
            use_bias=cfg.use_bias,
            dtype=cfg.dtype,
            kernel_init=cfg.default_kernel_init,
            name=f"tr_physatt_value_dense_{self.n_layer}",
        )

        # Physics-informed projections
        self.entanglement_proj = nn.Dense(
            features=cfg.num_heads,
            use_bias=False,
            dtype=cfg.dtype,
            kernel_init=cfg.default_kernel_init,
            name=f"tr_physatt_entanglement_proj_{self.n_layer}",
        )

        # For symmetry-aware processing
        self.symmetry_embed = nn.Dense(
            features=cfg.features // 2,
            use_bias=False,
            dtype=cfg.dtype,
            name=f"tr_physatt_symmetry_embed_{self.n_layer}",
        )

        self.output_dense = nn.Dense(
            features=cfg.features,
            use_bias=cfg.use_bias,
            name=f"tr_physatt_output_dense_{self.n_layer}",
            dtype=cfg.dtype,
            kernel_init=cfg.default_kernel_init,
            bias_init=cfg.default_bias_init,
        )
        self.dropout = nn.Dropout(
            rate=cfg.dropout_rate,
            deterministic=not cfg.training,
            name=f"tr_physatt_dropout_{self.n_layer}",
        )

        # Learnable distance scaling
        self.distance_scale = self.param(
            f"tr_physatt_distance_scale_{self.n_layer}",
            nn.initializers.constant(-1.0, dtype=cfg.dtype),
            (1,),
        )
        self.parity_strength = self.param(
            f"tr_physatt_parity_strength_{self.n_layer}",
            nn.initializers.constant(0.5, dtype=cfg.dtype),
            (1,),
        )

        # Exponential decay of interactions
        self.localization_length = self.param(
            f"tr_physatt_localization_length_{self.n_layer}",
            nn.initializers.constant(2.0, dtype=cfg.dtype),
            (1,),
        )

    def __call__(self, kv, q, mask, kv_cache, n_iter):
        cfg = self.config

        # Add spin symmetry encoding
        q = self.add_spin_symmetry(q)
        kv_encoded = self.add_spin_symmetry(kv)

        query = self.query_dense(q)

        if cfg.autoregressive:
            key = self.key_dense(kv_encoded[:, n_iter, :])
            value = self.value_dense(kv_encoded[:, n_iter, :])
        else:
            key = self.key_dense(kv_encoded)
            value = self.value_dense(kv_encoded)

        # Cache handling (same as original)
        if cfg.autoregressive:
            kv_cache.key = kv_cache.key.at[:, self.n_layer, n_iter, :].set(key)
            kv_cache.value = kv_cache.value.at[:, self.n_layer, n_iter, :].set(value)
            key = kv_cache.key[:, self.n_layer, : (n_iter + 1), :]
            value = kv_cache.value[:, self.n_layer, : (n_iter + 1), :]

        batch_size = query.shape[0]
        head_dim = cfg.features // cfg.num_heads

        query = query.reshape(batch_size, -1, cfg.num_heads, head_dim)
        key = key.reshape(batch_size, -1, cfg.num_heads, head_dim)
        value = value.reshape(batch_size, -1, cfg.num_heads, head_dim)

        # Physics-informed attention
        attention_output = self.quantum_aware_attention(
            query, key, value, mask, cfg.dtype
        )

        attention_output = attention_output.reshape(batch_size, -1, cfg.features)
        output = self.output_dense(attention_output)

        if cfg.use_dropout:
            output = self.dropout(output)

        return output, kv_cache

    def quantum_aware_attention(self, query, key, value, mask, dtype):
        """Attention mechanism designed for quantum spin systems"""
        length_q, length_k = query.shape[1], key.shape[1]
        hid_head = query.shape[3]

        # 1. Standard dot-product attention
        attention_logits = jnp.einsum("bqhd,bkhd->bhqk", query, key)

        # 2. Entanglement-aware bias
        entanglement_bias = self.compute_entanglement_bias(
            query, key, length_q, length_k
        )
        attention_logits += entanglement_bias

        # 3. Locality bias for physical interactions
        locality_bias = self.compute_locality_bias(length_q, length_k, hid_head, dtype)
        attention_logits += locality_bias

        # 4. Scale and mask
        d_k = query.shape[-1]
        attention_logits /= jnp.sqrt(d_k)

        if mask is not None:
            attention_logits = jnp.where(mask, attention_logits, jnp.finfo(dtype).min)

        # 5. Stabilized softmax
        attention_weights = nn.softmax(attention_logits, axis=-1).astype(dtype)

        # 6. Entanglement-weighted value combination
        output = jnp.einsum("bhqk,bkhd->bqhd", attention_weights, value)

        return output

    def compute_entanglement_bias(self, query, key, length_q, length_k):
        """Compute bias based on expected entanglement patterns"""
        # Distance-based entanglement prior
        positions_q = jnp.arange(length_q)
        positions_k = jnp.arange(length_k)
        distance_matrix = jnp.abs(positions_q[:, None] - positions_k[None, :])

        # Entanglement typically decays with distance but has oscillations
        # For anti-ferromagnetic systems, odd-even patterns matter
        parity_matrix = (positions_q[:, None] + positions_k[None, :]) % 2

        distance_bias = self.distance_scale * jnp.log1p(distance_matrix)
        parity_bias = self.parity_strength * (
            1 - 2 * parity_matrix
        )  # Alternating pattern

        return distance_bias[None, None, :, :] + parity_bias[None, None, :, :]

    def compute_locality_bias(self, length_q, length_k, hid_head, dtype):
        """Encourage local interactions while allowing global correlations"""
        # Local connectivity prior
        range_vec_q = jnp.arange(length_q)
        range_vec_k = jnp.arange(length_k)
        distance_mat = jnp.abs(range_vec_k[None, :] - range_vec_q[:, None])

        locality_bias = -distance_mat / jnp.exp(self.localization_length)
        return locality_bias[None, None, :, :]

    def add_spin_symmetry(self, x):
        """Add features that help preserve spin symmetry"""
        batch_size, seq_len, features = x.shape

        # Add positional features that respect spin exchange symmetry
        positions = jnp.arange(seq_len)
        even_odd = (positions % 2).astype(x.dtype)

        # Project to symmetry-aware space
        symmetry_features = self.symmetry_embed(even_odd[None, :, None])
        symmetry_features = jnp.broadcast_to(
            symmetry_features, (batch_size, seq_len, symmetry_features.shape[-1])
        )

        return jnp.concatenate([x, symmetry_features], axis=-1)

    @classmethod
    def quantum_causal_mask(cls, input_tokens, spin_system_type="heisenberg"):
        """Causal mask adapted for quantum spin systems"""
        mask = jnp.full(
            (input_tokens.shape[0], 1, input_tokens.shape[1], input_tokens.shape[1]), 1
        )

        if spin_system_type == "heisenberg":
            # Allow some non-local connections for entanglement
            # while maintaining causal structure
            pass

        return mask

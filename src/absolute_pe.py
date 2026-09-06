"""Vectorized additive sinusoidal positional encoding (Vaswani et al.).

``p(k, i) = sin(k / base^{i/d})`` for even ``i``, and
``p(k, i) = cos(k / base^{(i-1)/d})`` for odd ``i``.
"""

from __future__ import annotations

import numpy as np


def sinusoidal_pe(seq_len: int, dim: int, base: float = 10000.0) -> np.ndarray:
    """Return a ``(seq_len, dim)`` additive PE matrix."""
    if seq_len < 0 or dim < 1:
        raise ValueError("seq_len must be >= 0 and dim >= 1")
    positions = np.arange(seq_len, dtype=np.float64)[:, None]
    dims = np.arange(dim, dtype=np.float64)[None, :]
    exponent = np.where(dims % 2 == 0, dims / dim, (dims - 1.0) / dim)
    angles = positions / (base ** exponent)
    pe = np.empty((seq_len, dim), dtype=np.float64)
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe


def add_positional_encoding(
    embeddings: np.ndarray,
    base: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(pe, embeddings + pe)`` for a ``(seq, dim)`` matrix."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D (seq, dim)")
    pe = sinusoidal_pe(embeddings.shape[0], embeddings.shape[1], base=base)
    return pe, embeddings + pe

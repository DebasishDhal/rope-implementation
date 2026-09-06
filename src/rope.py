"""Vectorized RoPE: pairwise 2D rotations of query/key slices.

Two pairing conventions:

- ``interleaved`` — paper / GPT-J style: rotate dims ``(2i, 2i+1)``.
- ``llama`` — Hugging Face Llama/Qwen/SmolLM style: rotate
  ``(i, i + dim/2)`` (the ``rotate_half`` layout).

Frequencies match the usual schedule ``ω_i = base^{-2i/d}``.
"""

from __future__ import annotations

import numpy as np

VALID_STYLES = ("interleaved", "llama")


def pair_frequencies(dim: int, base: float = 10000.0) -> np.ndarray:
    """Return ``ω`` of shape ``(dim/2,)`` with ``ω_i = base ** (-2i / dim)``."""
    if dim % 2 != 0:
        raise ValueError(f"RoPE requires an even dim, got {dim}")
    n_pairs = dim // 2
    pair_idx = np.arange(n_pairs, dtype=np.float64) * 2.0
    return 1.0 / (base ** (pair_idx / dim))


def theta_grid(seq_len: int, dim: int, base: float = 10000.0) -> np.ndarray:
    """Rotation angles ``θ[k, i] = k * ω_i``, shape ``(seq_len, dim/2)``."""
    omega = pair_frequencies(dim, base=base)
    positions = np.arange(seq_len, dtype=np.float64)[:, None]
    return positions * omega[None, :]


def _broadcast_angles(angles: np.ndarray, like: np.ndarray) -> np.ndarray:
    """Broadcast ``(seq, n_pairs)`` onto ``like`` with shape ``(..., seq, n_pairs)``."""
    lead = like.ndim - 2
    return angles.reshape((1,) * lead + angles.shape)


def rotate_pair(x_even: np.ndarray, x_odd: np.ndarray, theta: np.ndarray):
    """Apply a 2D rotation by ``theta`` (radians) to coordinate pairs."""
    cos = np.cos(theta)
    sin = np.sin(theta)
    return x_even * cos - x_odd * sin, x_even * sin + x_odd * cos


def apply_rope(
    x: np.ndarray,
    base: float = 10000.0,
    style: str = "interleaved",
) -> np.ndarray:
    """Rotate last-dimension pairs of ``x`` (``(..., seq_len, dim)``)."""
    if style not in VALID_STYLES:
        raise ValueError(f"style must be one of {VALID_STYLES}, got {style!r}")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 2:
        raise ValueError("x must be at least 2D (seq, dim)")
    dim = x.shape[-1]
    seq_len = x.shape[-2]
    angles = theta_grid(seq_len, dim, base=base)
    cos = np.cos(angles)
    sin = np.sin(angles)
    n_pairs = dim // 2
    out = np.empty_like(x)

    if style == "interleaved":
        even = x[..., 0::2]
        odd = x[..., 1::2]
        cos_b = _broadcast_angles(cos, even)
        sin_b = _broadcast_angles(sin, odd)
        out[..., 0::2] = even * cos_b - odd * sin_b
        out[..., 1::2] = even * sin_b + odd * cos_b
    else:
        first = x[..., :n_pairs]
        second = x[..., n_pairs:]
        cos_b = _broadcast_angles(cos, first)
        sin_b = _broadcast_angles(sin, second)
        out[..., :n_pairs] = first * cos_b - second * sin_b
        out[..., n_pairs:] = first * sin_b + second * cos_b
    return out


def pair_xy(x: np.ndarray, token: int, pair: int, style: str = "interleaved") -> tuple[float, float]:
    """Return the 2D coordinates of one RoPE pair at one token."""
    vec = np.asarray(x)
    if vec.ndim != 2:
        raise ValueError("pair_xy expects (seq, dim)")
    dim = vec.shape[-1]
    n_pairs = dim // 2
    if not (0 <= pair < n_pairs):
        raise IndexError(f"pair {pair} out of range 0..{n_pairs - 1}")
    if style == "interleaved":
        return float(vec[token, 2 * pair]), float(vec[token, 2 * pair + 1])
    if style == "llama":
        return float(vec[token, pair]), float(vec[token, pair + n_pairs])
    raise ValueError(f"unknown style {style!r}")


def pair_dim_labels(pair: int, dim: int, style: str = "interleaved") -> tuple[str, str]:
    n_pairs = dim // 2
    if style == "interleaved":
        return f"dim {2 * pair}", f"dim {2 * pair + 1}"
    return f"dim {pair}", f"dim {pair + n_pairs}"


def l2_norms(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(x, dtype=np.float64), axis=-1)


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    an = l2_norms(a)
    bn = l2_norms(b)
    dots = np.sum(a * b, axis=-1)
    return dots / (an * bn + 1e-12)


def attention_scores(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Unnormalized ``Q K^T`` for ``(seq, dim)`` slices."""
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    return q @ k.T

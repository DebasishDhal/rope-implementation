"""Random tensors and lazy Hugging Face Q/K extraction.

Models are not loaded at import time. The last loaded model is cached.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.rope import apply_rope

MAX_SEQ_LEN = 2048

# Ungated Llama-like checkpoints (q_proj / k_proj + rotary). No HF token required.
MODEL_CHOICES = [
    "HuggingFaceM4/tiny-random-LlamaForCausalLM",
    "HuggingFaceTB/SmolLM2-135M",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]
DEFAULT_MODEL = MODEL_CHOICES[1]

_cache: dict[str, Any] = {"name": None, "model": None, "tokenizer": None}


_config_cache: dict[str, Any] = {}


def random_matrix(seq_len: int, dim: int, seed: int = 42) -> np.ndarray:
    if dim % 2 != 0:
        raise ValueError(f"dim must be even for RoPE, got {dim}")
    seq_len = int(np.clip(seq_len, 1, MAX_SEQ_LEN))
    rng = np.random.default_rng(int(seed))
    return rng.standard_normal((seq_len, dim)).astype(np.float64)


def random_qk(
    seq_len: int,
    dim: int,
    seed: int = 42,
    base: float = 10000.0,
) -> dict[str, Any]:
    q = random_matrix(seq_len, dim, seed=seed)
    k = random_matrix(seq_len, dim, seed=seed + 1)
    return _pack_tensors(
        q_before=q,
        k_before=k,
        embeddings=q.copy(),
        tokens=[f"t{i}" for i in range(q.shape[0])],
        base=float(base),
        style="interleaved",
        n_q_heads=1,
        n_kv_heads=1,
        checksum=None,
        source="random",
        model_name=None,
    )


def _pack_tensors(
    *,
    q_before: np.ndarray,
    k_before: np.ndarray,
    embeddings: np.ndarray,
    tokens: list[str],
    base: float,
    style: str,
    n_q_heads: int,
    n_kv_heads: int,
    checksum: float | None,
    source: str,
    model_name: str | None,
    q_after_model: np.ndarray | None = None,
    k_after_model: np.ndarray | None = None,
) -> dict[str, Any]:
    q_after = apply_rope(q_before, base=base, style=style)
    k_after = apply_rope(k_before, base=base, style=style)
    return {
        "q_before": q_before,
        "k_before": k_before,
        "q_after": q_after,
        "k_after": k_after,
        "q_after_model": q_after_model,
        "k_after_model": k_after_model,
        "embeddings": embeddings,
        "tokens": tokens,
        "base": float(base),
        "style": style,
        "n_q_heads": int(n_q_heads),
        "n_kv_heads": int(n_kv_heads),
        "checksum": checksum,
        "source": source,
        "model_name": model_name,
    }


def _require_hf():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Real-model mode needs `torch` and `transformers`. "
            "Install them or use Random matrix mode."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def get_model_head_dim(model_name: str) -> int:
    """Return the Q/K dimension per attention head without loading model weights."""
    if model_name not in _config_cache:
        try:
            from transformers import AutoConfig
        except ImportError as exc:
            raise RuntimeError("Real-model mode needs `transformers`.") from exc
        _config_cache[model_name] = AutoConfig.from_pretrained(model_name)

    config = _config_cache[model_name]
    configured_head_dim = getattr(config, "head_dim", None)
    if configured_head_dim is not None:
        return int(configured_head_dim)
    return int(config.hidden_size) // int(config.num_attention_heads)


def get_model(model_name: str):
    """Load tokenizer + causal LM on CPU; cache the last selection."""
    torch, AutoModelForCausalLM, AutoTokenizer = _require_hf()
    if _cache["name"] == model_name and _cache["model"] is not None:
        return _cache["model"], _cache["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to("cpu")
    _cache["name"] = model_name
    _cache["model"] = model
    _cache["tokenizer"] = tokenizer
    return model, tokenizer


def _backbone(model):
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model
    raise RuntimeError(
        "This checkpoint is not Llama-like (expected model.model.embed_tokens)."
    )


def _rotary_module(backbone, attn):
    if hasattr(backbone, "rotary_emb"):
        return backbone.rotary_emb
    if hasattr(attn, "rotary_emb"):
        return attn.rotary_emb
    return None


def _rotate_half_torch(x, torch):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _broadcast_cos_sin(cos, sin, q):
    """Make cos/sin broadcast with q of shape (batch, heads, seq, dim)."""
    if cos.dim() == 2:
        cos, sin = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return cos, sin


def _hf_rotary(q, k, rotary, position_ids, torch):
    try:
        cos, sin = rotary(q, position_ids=position_ids)
    except TypeError:
        try:
            cos, sin = rotary(q, seq_len=q.shape[-2])
        except TypeError:
            cos, sin = rotary(q)
    if cos.shape[-1] == q.shape[-1] // 2:
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
    cos, sin = _broadcast_cos_sin(cos, sin, q)
    q_rot = (q * cos) + (_rotate_half_torch(q, torch) * sin)
    k_rot = (k * cos) + (_rotate_half_torch(k, torch) * sin)
    return q_rot, k_rot


def extract_from_model(model_name: str, sentence: str) -> dict[str, Any]:
    torch, _, _ = _require_hf()
    model, tokenizer = get_model(model_name)
    text = sentence.strip() or "RoPE rotates query and key vectors."
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    backbone = _backbone(model)
    cfg = model.config
    n_heads = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_heads))
    hidden_size = int(cfg.hidden_size)
    head_dim = int(getattr(cfg, "head_dim", hidden_size // n_heads))
    base = float(getattr(cfg, "rope_theta", 10000.0))

    with torch.no_grad():
        embeds = backbone.embed_tokens(input_ids)
        layer = backbone.layers[0]
        attn = layer.self_attn
        hidden = layer.input_layernorm(embeds)
        q = attn.q_proj(hidden)
        k = attn.k_proj(hidden)
        seq = q.shape[1]
        q = q.view(1, seq, n_heads, head_dim).transpose(1, 2).contiguous()
        k = k.view(1, seq, n_kv, head_dim).transpose(1, 2).contiguous()
        position_ids = torch.arange(seq).unsqueeze(0)
        rotary = _rotary_module(backbone, attn)
        q_model = k_model = None
        if rotary is not None:
            q_model, k_model = _hf_rotary(q, k, rotary, position_ids, torch)

    q_np = q[0].cpu().numpy().astype(np.float64)
    k_np = k[0].cpu().numpy().astype(np.float64)
    emb_np = embeds[0].cpu().numpy().astype(np.float64)
    q_model_np = k_model_np = None
    checksum = None
    if q_model is not None:
        q_model_np = q_model[0].cpu().numpy().astype(np.float64)
        k_model_np = k_model[0].cpu().numpy().astype(np.float64)
        q_edu = apply_rope(q_np, base=base, style="llama")
        checksum = float(np.max(np.abs(q_edu - q_model_np)))

    return _pack_tensors(
        q_before=q_np,
        k_before=k_np,
        embeddings=emb_np,
        tokens=tokens,
        base=base,
        style="llama",
        n_q_heads=n_heads,
        n_kv_heads=n_kv,
        checksum=checksum,
        source="model",
        model_name=model_name,
        q_after_model=q_model_np,
        k_after_model=k_model_np,
    )


def select_head(tensor: np.ndarray, head: int) -> np.ndarray:
    """Return a ``(seq, dim)`` slice; ``tensor`` is 2D or ``(heads, seq, dim)``."""
    t = np.asarray(tensor)
    if t.ndim == 2:
        return t
    if t.ndim != 3:
        raise ValueError(f"expected 2D or 3D tensor, got {t.ndim}D")
    h = int(np.clip(head, 0, t.shape[0] - 1))
    return t[h]


def expand_kv_heads(k: np.ndarray, n_q_heads: int) -> np.ndarray:
    """Repeat GQA key heads so they align with query heads."""
    t = np.asarray(k)
    if t.ndim == 2:
        return t
    n_kv = t.shape[0]
    if n_kv == n_q_heads:
        return t
    if n_q_heads % n_kv != 0:
        return t
    return np.repeat(t, n_q_heads // n_kv, axis=0)

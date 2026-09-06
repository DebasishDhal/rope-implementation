"""RoPE Explorer Gradio app. Imports only from ``src/``."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import gradio as gr

from src.absolute_pe import add_positional_encoding
from src.extract import (
    DEFAULT_MODEL,
    MAX_SEQ_LEN,
    MODEL_CHOICES,
    expand_kv_heads,
    extract_from_model,
    random_qk,
    select_head,
)
from src.plots import (
    additive_pe_heatmaps,
    attention_bars,
    attention_heatmaps,
    bulk_before_after_delta,
    frequency_strip,
    norm_compare_add_vs_rope,
    norms_and_cosine,
    position_sweep,
    rotation_2d,
    theta_heatmap,
)
from src.rope import (
    attention_scores,
    pair_dim_labels,
    pair_frequencies,
    pair_xy,
    rotate_pair,
    theta_grid,
)

HOWTO_MD = """
# How RoPE works

After each token has a vector from the **embedding table**, attention builds two extra
vectors per token with linear layers (`q_proj`, `k_proj`):

- **Q (query)** — what this token is looking for
- **K (key)** — what this token offers as a match

Attention scores are (scaled) **dot products** `Q · K`. **RoPE rotates those Q and K
vectors in 2D planes before the dot product.** It does **not** add a position vector
onto the raw token embeddings. This app shows embeddings only as context, then focuses
on Q and K before vs after RoPE.

## Pairwise rotation

For even dimension `d`, pair `i` uses frequency

$$\\omega_i = 10000^{-2i/d},\\qquad \\theta(k,i) = k\\,\\omega_i$$

**Interleaved (paper-style)** pairing `(2i, 2i+1)`:

$$
x'_{2i} = x_{2i}\\cos\\theta - x_{2i+1}\\sin\\theta,\\qquad
x'_{2i+1} = x_{2i}\\sin\\theta + x_{2i+1}\\cos\\theta
$$

Hugging Face Llama-like models use the same frequencies but pair `(i, i + d/2)`
(`rotate_half`). Random-matrix mode uses interleaved pairing; real models use the
Llama layout so the numpy implementation can be checksummed against `rotary_emb`.

Relative positions fall out of the algebra: `R(m)^T R(n) = R(n-m)`.

Shaw relative attention (learned bias `b_{m-n}` on scores) is a **different**
mechanism and is not computed here.
"""

PLACEHOLDER = go.Figure().update_layout(
    title="Run **Compute** on the Setup tab first",
    template="plotly_white",
    height=320,
)


def _safe_slider_max(n: int) -> int:
    """Gradio sliders need max > min; keep a one-step range even at edge cases."""
    return max(int(n), 1)


def compute(
    source: str,
    sentence: str,
    model_name: str,
    seq_len: int,
    dim: int,
    seed: int,
    base: float,
    progress=gr.Progress(track_tqdm=False),
):
    try:
        if source.startswith("Random"):
            progress(0.4, desc="Sampling random Q/K")
            data = random_qk(int(seq_len), int(dim), seed=int(seed), base=float(base))
        else:
            progress(0.2, desc=f"Loading {model_name} (first time downloads weights)")
            data = extract_from_model(model_name, sentence)
        seq = int(select_head(data["q_before"], 0).shape[0])
        head_dim = int(select_head(data["q_before"], 0).shape[1])
        n_pairs = head_dim // 2
        n_heads = max(int(data["n_q_heads"]) - 1, 0)
        checksum = data["checksum"]
        if checksum is None:
            status = (
                f"Random Q/K · seq={seq} · dim={head_dim} · base={data['base']:g} · "
                f"style={data['style']}"
            )
        else:
            status = (
                f"Model `{data['model_name']}` · {seq} tokens · head_dim={head_dim} · "
                f"Q heads={data['n_q_heads']} · KV heads={data['n_kv_heads']} · "
                f"rope_theta={data['base']:g} · "
                f"max |numpy RoPE − model rotary| on Q = **{checksum:.3e}**"
            )
        token_labels = ", ".join(data["tokens"][:seq])
        status = status + f"\n\nTokens: `{token_labels}`"
        return (
            data,
            status,
            gr.update(maximum=_safe_slider_max(n_heads), value=0),
            gr.update(maximum=_safe_slider_max(seq - 1), value=0),
            gr.update(maximum=_safe_slider_max(n_pairs - 1), value=0),
            gr.update(maximum=_safe_slider_max(seq - 1), value=0),
        )
    except Exception as exc:
        return (
            None,
            f"**Error:** {exc}",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )


def _qk_slice(data: dict, which: str, head: int):
    before = data["q_before"] if which == "Q" else data["k_before"]
    after = data["q_after"] if which == "Q" else data["k_after"]
    return select_head(before, head), select_head(after, head)


def update_bulk(data, which, head, mod_2pi):
    if not data:
        fig = PLACEHOLDER
        return fig, fig, fig, fig
    before, after = _qk_slice(data, which, int(head))
    dim = before.shape[-1]
    seq = before.shape[0]
    return (
        bulk_before_after_delta(before, after, tokens=data["tokens"]),
        norms_and_cosine(before, after, tokens=data["tokens"]),
        theta_heatmap(seq, dim, data["base"], mod_2pi=bool(mod_2pi)),
        frequency_strip(dim, data["base"]),
    )


def update_individual(data, which, head, token, pair, sweep):
    if not data:
        return "Compute on the Setup tab first.", PLACEHOLDER, PLACEHOLDER
    before, after = _qk_slice(data, which, int(head))
    token = int(np.clip(token, 0, before.shape[0] - 1))
    n_pairs = before.shape[1] // 2
    pair = int(np.clip(pair, 0, n_pairs - 1))
    style = data["style"]
    xb, yb = pair_xy(before, token, pair, style=style)
    xa, ya = pair_xy(after, token, pair, style=style)
    theta = float(theta_grid(before.shape[0], before.shape[1], data["base"])[token, pair])
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
    xe_chk, xo_chk = rotate_pair(np.array([xb]), np.array([yb]), np.array([theta]))
    d0, d1 = pair_dim_labels(pair, before.shape[1], style=style)
    table = f"""
### Token `{token}` · pair `{pair}` (`{d0}`, `{d1}`)

| | {d0} | {d1} |
|---|---:|---:|
| before | {xb:.6f} | {yb:.6f} |
| after | {xa:.6f} | {ya:.6f} |
| check (`rotate_pair`) | {float(xe_chk):.6f} | {float(xo_chk):.6f} |

**θ(k,i) = {theta:.6f} rad** · cos = {cos_t:.6f} · sin = {sin_t:.6f}

`x'_even = x_even cos θ − x_odd sin θ`  
`x'_odd  = x_even sin θ + x_odd cos θ`
"""
    neighbors = [n for n in (token - 1, token + 1, token + 2) if 0 <= n < before.shape[0]]
    rot = rotation_2d(before, after, token, pair, style, theta, neighbor_tokens=neighbors)
    if sweep:
        omega = float(pair_frequencies(before.shape[1], data["base"])[pair])
        sweep_fig = position_sweep(xb, yb, omega, before.shape[0], token)
    else:
        sweep_fig = PLACEHOLDER
        sweep_fig.update_layout(title="Enable “replay same pair at every k” to see position-only spin")
    return table, rot, sweep_fig


def update_attention(data, head, query_token):
    if not data:
        return PLACEHOLDER, PLACEHOLDER, ""
    q_b = select_head(data["q_before"], int(head))
    q_a = select_head(data["q_after"], int(head))
    k_b_all = expand_kv_heads(data["k_before"], data["n_q_heads"])
    k_a_all = expand_kv_heads(data["k_after"], data["n_q_heads"])
    k_b = select_head(k_b_all, int(head))
    k_a = select_head(k_a_all, int(head))
    sb = attention_scores(q_b, k_b)
    sa = attention_scores(q_a, k_a)
    qt = int(np.clip(query_token, 0, sb.shape[0] - 1))
    note = (
        "Additive PE changes values by **addition**. RoPE encodes **relative** offset "
        "because `R(m)^T R(n) = R(n−m)`: the score depends on the position difference, "
        "not on absolute indices alone."
    )
    return attention_heatmaps(sb, sa), attention_bars(sb[qt], sa[qt], qt), note


def update_compare(data):
    if not data:
        return PLACEHOLDER, PLACEHOLDER, ""
    emb = np.asarray(data["embeddings"], dtype=np.float64)
    # Compare tab always uses additive PE on the embedding matrix (may be wider than a head).
    pe, combined = add_positional_encoding(emb, base=data["base"])
    q_b = select_head(data["q_before"], 0)
    q_a = select_head(data["q_after"], 0)
    heat = additive_pe_heatmaps(emb, pe, combined)
    norms = norm_compare_add_vs_rope(emb, combined, q_b, q_a)
    copy = """
**Absolute sinusoidal PE** *adds* a position-shaped vector, so both **norm and direction** change.

**RoPE** *rotates* query/key pairs: **norm stays**, and the relative angle depends on `m − n`.

Shaw-style relative attention (`q_m^T k_n + b_{m-n}`) is a third, learned-bias mechanism — not shown as a plot.
"""
    return heat, norms, copy


def toggle_source(source: str):
    is_random = source.startswith("Random")
    return (
        gr.update(visible=is_random),
        gr.update(visible=is_random),
        gr.update(visible=is_random),
        gr.update(visible=not is_random),
        gr.update(visible=not is_random),
    )


with gr.Blocks(title="RoPE Explorer") as demo:
    state = gr.State(None)
    gr.Markdown("# RoPE Explorer")
    gr.Markdown(
        "Interactive view of **Rotary Position Embedding**: random Q/K matrices or "
        "query/key vectors from a small ungated Hugging Face model."
    )
    with gr.Tabs():
        with gr.Tab("How RoPE works"):
            gr.Markdown(HOWTO_MD)

        with gr.Tab("Setup"):
            source = gr.Radio(
                ["Random matrix", "Real model"],
                value="Random matrix",
                label="Source",
            )
            with gr.Row():
                sentence = gr.Textbox(
                    value="RoPE rotates query and key vectors.",
                    label="Sentence (real model)",
                    visible=False,
                )
                model_name = gr.Dropdown(
                    MODEL_CHOICES,
                    value=DEFAULT_MODEL,
                    label="Model (ungated, Llama-like)",
                    visible=False,
                )
            with gr.Row():
                seq_len = gr.Slider(2, MAX_SEQ_LEN, value=16, step=1, label="Sequence length")
                dim = gr.Slider(4, 128, value=32, step=2, label="Dimension (even)")
                seed = gr.Number(value=0, label="Seed", precision=0)
            base = gr.Number(
                value=10000,
                label="RoPE base (overridden by config.rope_theta for real models)",
            )
            compute_btn = gr.Button("Compute", variant="primary")
            status = gr.Markdown("Choose a source and click Compute.")
            head = gr.Slider(minimum=0, maximum=2, step=1, value=0, label="Head index (real models)")

        with gr.Tab("Bulk changes"):
            which = gr.Radio(["Q", "K"], value="Q", label="Tensor")
            mod_2pi = gr.Checkbox(False, label="θ heatmap: wrap mod 2π")
            bulk_main = gr.Plot(label="Before / after / delta")
            bulk_norm = gr.Plot(label="Norms and cosine")
            bulk_theta = gr.Plot(label="θ(k, i)")
            bulk_freq = gr.Plot(label="ω_i")

        with gr.Tab("Individual changes"):
            with gr.Row():
                token_k = gr.Slider(0, 15, step=1, value=0, label="Token index k")
                pair_i = gr.Slider(0, 15, step=1, value=0, label="Pair index i")
            sweep = gr.Checkbox(True, label="Replay the same content pair at every position k")
            pair_table = gr.Markdown()
            pair_plot = gr.Plot()
            sweep_plot = gr.Plot()

        with gr.Tab("Attention effect"):
            query_token = gr.Slider(0, 15, step=1, value=0, label="Query token")
            attn_heat = gr.Plot()
            attn_bar = gr.Plot()
            attn_note = gr.Markdown()

        with gr.Tab("Compare to additive PE"):
            pe_heat = gr.Plot()
            pe_norm = gr.Plot()
            pe_note = gr.Markdown()

    compute_btn.click(
        compute,
        inputs=[source, sentence, model_name, seq_len, dim, seed, base],
        outputs=[state, status, head, token_k, pair_i, query_token],
    )
    source.change(
        toggle_source,
        inputs=[source],
        outputs=[seq_len, dim, seed, sentence, model_name],
    )

    bulk_inputs = [state, which, head, mod_2pi]
    bulk_outputs = [bulk_main, bulk_norm, bulk_theta, bulk_freq]
    for ctrl in bulk_inputs:
        ctrl.change(update_bulk, inputs=bulk_inputs, outputs=bulk_outputs)

    ind_inputs = [state, which, head, token_k, pair_i, sweep]
    ind_outputs = [pair_table, pair_plot, sweep_plot]
    for ctrl in ind_inputs:
        ctrl.change(update_individual, inputs=ind_inputs, outputs=ind_outputs)

    attn_inputs = [state, head, query_token]
    attn_outputs = [attn_heat, attn_bar, attn_note]
    for ctrl in attn_inputs:
        ctrl.change(update_attention, inputs=attn_inputs, outputs=attn_outputs)

    state.change(update_compare, inputs=[state], outputs=[pe_heat, pe_norm, pe_note])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

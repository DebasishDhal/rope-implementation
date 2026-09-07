"""Plotly figures for bulk heatmaps, pair rotation, and attention."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.rope import l2_norms, pair_frequencies, pair_xy, row_cosine, theta_grid


def _empty(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, template="plotly_white")
    return fig


def downsample(mat: np.ndarray, max_cols: int = 64, max_rows: int = 64) -> np.ndarray:
    m = np.asarray(mat)
    row_step = max(1, int(np.ceil(m.shape[0] / max_rows)))
    if m.ndim == 1:
        return m[::row_step]
    col_step = max(1, int(np.ceil(m.shape[1] / max_cols)))
    return m[::row_step, ::col_step]


def heatmap(z: np.ndarray, title: str, xtitle: str, ytitle: str, tokens=None) -> go.Figure:
    z = downsample(np.asarray(z, dtype=np.float64))
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            colorbar=dict(title="value"),
            hovertemplate="y=%{y}<br>x=%{x}<br>z=%{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        height=380,
        margin=dict(l=60, r=40, t=50, b=50),
    )
    if tokens is not None and len(tokens) == z.shape[0]:
        fig.update_yaxes(tickmode="array", tickvals=list(range(len(tokens))), ticktext=tokens)
    return fig


def bulk_before_after_delta(before: np.ndarray, after: np.ndarray, tokens=None) -> go.Figure:
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    delta = after - before
    mats = [downsample(before), downsample(after), downsample(delta)]
    titles = ["Before RoPE", "After RoPE", "Delta (after − before)"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=titles)
    for i, mat in enumerate(mats, start=1):
        fig.add_trace(
            go.Heatmap(
                z=mat,
                showscale=(i == 3),
                colorbar=dict(title="matrix value") if i == 3 else None,
                hovertemplate="token position k=%{y}<br>dimension index j=%{x}<br>value=%{z:.4f}<extra></extra>",
            ),
            row=1,
            col=i,
        )
        fig.update_xaxes(title_text="dimension index j (within selected head)", row=1, col=i)
        fig.update_yaxes(title_text="token position k", row=1, col=i)
        fig.update_yaxes(autorange="reversed", row=1, col=i)
    fig.update_layout(template="plotly_white", height=430, margin=dict(t=60, b=65))
    return fig


def norms_and_cosine(before: np.ndarray, after: np.ndarray, tokens=None) -> go.Figure:
    nb = l2_norms(before)
    na = l2_norms(after)
    cos = row_cosine(before, after)
    xs = list(range(len(nb)))
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Per-token L2 norm", "Cosine (original vs rotated)"])
    fig.add_trace(go.Scatter(x=xs, y=nb, name="before", mode="lines+markers"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=na, name="after", mode="lines+markers"), row=1, col=1)
    fig.add_trace(go.Bar(x=xs, y=cos, name="cosine", showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text="token position k", row=1, col=1)
    fig.update_xaxes(title_text="token position k", row=1, col=2)
    fig.update_yaxes(title_text="L2 norm of Q/K vector", row=1, col=1)
    fig.update_yaxes(title_text="cosine similarity (before, after)", range=[min(0.0, float(np.min(cos)) - 0.05), 1.02], row=1, col=2)
    fig.update_layout(template="plotly_white", height=380, barmode="group")
    return fig


def theta_heatmap(seq_len: int, dim: int, base: float, mod_2pi: bool = False) -> go.Figure:
    grid = theta_grid(seq_len, dim, base=base)
    if mod_2pi:
        grid = np.mod(grid, 2 * np.pi)
        title = "θ(k, i) mod 2π"
    else:
        title = "θ(k, i) = k · ω_i"
    z = downsample(grid, max_cols=64, max_rows=64)
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            colorbar=dict(title="rotation angle θ (radians)"),
            hovertemplate="token k=%{y}<br>pair i=%{x}<br>θ=%{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="pair index i (dimension pair)",
        yaxis_title="token position k",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        height=380,
    )
    return fig


def frequency_strip(dim: int, base: float) -> go.Figure:
    omega = pair_frequencies(dim, base=base)
    fig = go.Figure(data=go.Bar(x=list(range(len(omega))), y=omega, name="ω_i"))
    fig.update_layout(
        title="Pair frequencies ω_i (pair 0 is fastest)",
        xaxis_title="pair index i (dimension pair)",
        yaxis_title="frequency ω_i (radians per position)",
        yaxis_type="log",
        template="plotly_white",
        height=280,
    )
    return fig


def _arc_points(x0, y0, x1, y1, n=40):
    r0 = float(np.hypot(x0, y0))
    r1 = float(np.hypot(x1, y1))
    r = 0.5 * (r0 + r1)
    if r < 1e-9:
        return [], []
    a0 = float(np.arctan2(y0, x0))
    a1 = float(np.arctan2(y1, x1))
    delta = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
    ts = np.linspace(0.0, 1.0, n)
    angs = a0 + ts * delta
    rr = r * 0.55
    return list(rr * np.cos(angs)), list(rr * np.sin(angs))


def rotation_2d(
    matrix_before: np.ndarray,
    matrix_after: np.ndarray,
    token: int,
    pair: int,
    style: str,
    theta: float,
    neighbor_tokens: list[int] | None = None,
) -> go.Figure:
    xb, yb = pair_xy(matrix_before, token, pair, style=style)
    xa, ya = pair_xy(matrix_after, token, pair, style=style)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, xb],
            y=[0, yb],
            mode="lines+markers",
            name="before",
            line=dict(width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, xa],
            y=[0, ya],
            mode="lines+markers",
            name="after",
            line=dict(width=3),
        )
    )
    xs, ys = _arc_points(xb, yb, xa, ya)
    if xs:
        fig.add_trace(
            go.Scatter(x=xs, y=ys, mode="lines", name=f"θ={theta:.3f} rad", line=dict(dash="dot"))
        )
    if neighbor_tokens:
        for n in neighbor_tokens:
            if n == token:
                continue
            nx, ny = pair_xy(matrix_after, n, pair, style=style)
            fig.add_trace(
                go.Scatter(
                    x=[0, nx],
                    y=[0, ny],
                    mode="lines+markers",
                    name=f"token {n} after",
                    opacity=0.45,
                )
            )
    lim = max(abs(xb), abs(yb), abs(xa), abs(ya), 1e-3) * 1.3
    fig.update_layout(
        title=f"Pair {pair} at token {token}: 2D rotation",
        xaxis=dict(scaleanchor="y", scaleratio=1, range=[-lim, lim], zeroline=True),
        yaxis=dict(range=[-lim, lim], zeroline=True),
        template="plotly_white",
        height=460,
        legend=dict(orientation="h"),
    )
    return fig


def position_sweep(
    x_even: float,
    x_odd: float,
    omega: float,
    seq_len: int,
    highlight: int,
) -> go.Figure:
    """Same content pair spun by θ(k)=k ω — position is the only change."""
    ks = np.arange(seq_len)
    thetas = ks * omega
    xs = x_even * np.cos(thetas) - x_odd * np.sin(thetas)
    ys = x_even * np.sin(thetas) + x_odd * np.cos(thetas)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+lines", name="path over k"))
    fig.add_trace(
        go.Scatter(
            x=[0, xs[highlight]],
            y=[0, ys[highlight]],
            mode="lines+markers",
            name=f"k={highlight}",
            line=dict(width=3),
        )
    )
    lim = max(float(np.max(np.abs(xs))), float(np.max(np.abs(ys))), 1e-3) * 1.3
    fig.update_layout(
        title="Same (x_even, x_odd) rotated at every position k",
        xaxis=dict(scaleanchor="y", scaleratio=1, range=[-lim, lim]),
        yaxis=dict(range=[-lim, lim]),
        template="plotly_white",
        height=420,
    )
    return fig


def attention_heatmaps(
    scores_before: np.ndarray,
    scores_after: np.ndarray,
    tokens=None,
) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=["QKᵀ without RoPE", "QKᵀ with RoPE"])
    seq_len = scores_before.shape[0]
    row_step = max(1, int(np.ceil(seq_len / 64)))
    indices = np.arange(0, seq_len, row_step)
    labels = [str(tokens[i]) if tokens is not None else f"token {i}" for i in indices]
    customdata = np.empty((len(indices), len(indices), 2), dtype=object)
    customdata[:, :, 0] = np.asarray(labels)[:, None]
    customdata[:, :, 1] = np.asarray(labels)[None, :]
    for i, mat in enumerate([scores_before, scores_after], start=1):
        fig.add_trace(
            go.Heatmap(
                z=np.asarray(mat)[::row_step, ::row_step],
                x=indices,
                y=indices,
                customdata=customdata,
                showscale=(i == 2),
                hovertemplate=(
                    "query position k=%{y}<br>query token=%{customdata[0]}<br>"
                    "key position k=%{x}<br>key token=%{customdata[1]}<br>"
                    "Q·K score=%{z:.4f}<extra></extra>"
                ),
            ),
            row=1,
            col=i,
        )
        fig.update_yaxes(autorange="reversed", title_text="query token", row=1, col=i)
        fig.update_xaxes(title_text="key token", row=1, col=i)
    fig.update_layout(template="plotly_white", height=420)
    return fig


def attention_bars(logits_before: np.ndarray, logits_after: np.ndarray, query_token: int) -> go.Figure:
    xs = list(range(len(logits_before)))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=logits_before, name="without RoPE"))
    fig.add_trace(go.Bar(x=xs, y=logits_after, name="with RoPE"))
    fig.update_layout(
        title=f"Attention logits from query token {query_token}",
        xaxis_title="key token",
        yaxis_title="Q·K",
        barmode="group",
        template="plotly_white",
        height=360,
    )
    return fig


def additive_pe_heatmaps(emb: np.ndarray, pe: np.ndarray, combined: np.ndarray) -> go.Figure:
    titles = ["Token embeddings", "Additive sinusoidal PE", "Embeddings + PE"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=titles)
    for i, mat in enumerate([emb, pe, combined], start=1):
        fig.add_trace(
            go.Heatmap(z=downsample(mat), showscale=(i == 3)),
            row=1,
            col=i,
        )
        fig.update_yaxes(autorange="reversed", row=1, col=i)
    fig.update_layout(template="plotly_white", height=400)
    return fig


def norm_compare_add_vs_rope(
    emb: np.ndarray,
    emb_plus_pe: np.ndarray,
    q_before: np.ndarray,
    q_after: np.ndarray,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=l2_norms(emb), mode="lines+markers", name="embeddings"))
    fig.add_trace(go.Scatter(y=l2_norms(emb_plus_pe), mode="lines+markers", name="embeddings + PE"))
    fig.add_trace(go.Scatter(y=l2_norms(q_before), mode="lines+markers", name="Q before RoPE"))
    fig.add_trace(go.Scatter(y=l2_norms(q_after), mode="lines+markers", name="Q after RoPE"))
    fig.update_layout(
        title="L2 norms: additive PE changes magnitude; RoPE does not",
        xaxis_title="token",
        yaxis_title="L2",
        template="plotly_white",
        height=360,
    )
    return fig

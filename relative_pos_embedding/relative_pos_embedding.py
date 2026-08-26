# In simple terms, we are calculating this
### Sm,n = q_m^T * k_n + b_m-n ## here m and n are the positions of the query and key vectors respectively. The similarity score must depend on the relative position of the query and key vectors as well. b is some learned function of the relative position.

# Some minor lacuna pending here in the implementation, needs to be cleared.

import numpy as np

import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def shaw_relative_attention(
    X,
    Wq,
    Wk,
    Wv,
    relative_key_embeddings,
    relative_value_embeddings,
    max_relative_position
):
    n_tokens, d_model = X.shape

    # --------------------------------------------------
    # 1. Relative positions
    # --------------------------------------------------

    positions = np.arange(n_tokens)

    relative_positions = (
        positions[:, None]
        - positions[None, :]
    )

    relative_positions = np.clip(
        relative_positions,
        -max_relative_position,
        max_relative_position
    )

    # Convert [-max, ..., +max] → [0, ..., 2*max]
    relative_indices = (
        relative_positions
        + max_relative_position
    )

    # --------------------------------------------------
    # 2. Query
    # --------------------------------------------------

    Q = X @ Wq

    # --------------------------------------------------
    # 3. Relative Key embeddings
    # --------------------------------------------------

    relative_key = (
        relative_key_embeddings[
            relative_indices
        ]
    )

    # Shape:
    # (n_tokens, n_tokens, d_model)

    # x_n + relative positional embedding
    K_input = (
        X[None, :, :]
        + relative_key
    )

    # Apply Wk
    K_relative = K_input @ Wk

    # --------------------------------------------------
    # 4. Attention scores
    # --------------------------------------------------

    scores = np.einsum(
        "md,mnd->mn",
        Q,
        K_relative
    )

    # --------------------------------------------------
    # 5. Relative Value embeddings
    # --------------------------------------------------

    relative_value = (
        relative_value_embeddings[
            relative_indices
        ]
    )

    V_input = (
        X[None, :, :]
        + relative_value
    )

    V_relative = V_input @ Wv

    # --------------------------------------------------
    # 6. Attention weights
    # --------------------------------------------------

    attention_weights = softmax(
        scores,
        axis=-1
    )

    # --------------------------------------------------
    # 7. Weighted Values
    # --------------------------------------------------

    output = np.einsum(
        "mn,mnd->md",
        attention_weights,
        V_relative
    )

    return output
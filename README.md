---
title: RoPE Embedding Visualization
emoji: 📚
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 6.26.0
python_version: 3.11
app_file: app.py
pinned: false
short_description: Visualize how RoPE rotates query and key vectors
---

# RoPE Explorer

Interactive Gradio app for **Rotary Position Embedding**. The Hugging Face Space
runs `app.py` (`sdk: gradio`). Local Docker (`Dockerfile`, `compose.yaml`) is for
running `python app.py` on port **7860**.

Production math lives in [`src/`](src/) (`rope.py`, `absolute_pe.py`, `extract.py`,
`plots.py`). Scratch scripts under `rope_implementation/`,
`absolute_sinusoidal_position_embedding/`, and `relative_pos_embedding/` are
learning notes only and are **not** imported by the app.

## Modes

1. **Random matrix** — sample even-width Q (and K) tensors, apply numpy RoPE,
   inspect heatmaps, pairwise 2D rotation, `QK^T`, and additive sinusoidal PE.
2. **Real model** — lazy-load an ungated Llama-like checkpoint (default
   `HuggingFaceTB/SmolLM2-135M`; `HuggingFaceM4/tiny-random-LlamaForCausalLM`
   is included for a very small test model), take `embed_tokens`, first-layer `q_proj` /
   `k_proj` (GQA-aware), and compare educational numpy RoPE (`llama` pairing)
   to the model's `rotary_emb`. No Hugging Face token is required. Gated models
   are not used.

First load of a model downloads weights into the cache; later runs reuse the
last loaded model in memory.

CPU is enough for SmolLM2 and Qwen2.5-0.5B. TinyLlama is included for a larger
example and may be slow on CPU. This Space does **not** require ZeroGPU
(`@spaces.GPU` is unused).

## Local run

```bash
pip install -r requirements.txt
python app.py
```

Or `docker compose up`. Hugging Face Cloud uses the README YAML (`sdk: gradio`),
not the Docker image, unless the Space SDK is switched to Docker.

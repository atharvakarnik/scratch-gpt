# Roadmap — Phase 1: scratch-trained GPT

Each assignment: spec in `docs/assignments/A<n>.md`, learner-owned stubs with TODOs, pytest
tests, acceptance criteria. An assignment is complete when its tests pass, its acceptance
criteria are met, and `docs/STATE.md` records the verified results.

| # | Milestone | Learner implements | Acceptance |
|---|---|---|---|
| M0 | Environment (scaffolded) | — | `pytest` runs; MPS available; `bench_mps.py` tokens/s + peak memory recorded |
| A1 | Preprocessing + byte-level BPE | streaming story parser, cleaning policy, BPE train/encode/decode/save/load, special tokens | tests pass; exact round-trip; bytes/token & tokens/story reported; context length (256 vs 512) decided |
| A2 | Dataset pipeline | parallel corpus encoding → `uint16` memmap; random-window `get_batch` | y = x shifted by 1; dtypes; decoded batches readable; encode throughput recorded |
| A3 | Embeddings + causal attention | token/position embeddings; multi-head causal self-attention (manual) | causality test; matches `F.scaled_dot_product_attention` within 1e-5 |
| A4 | GPT model | MLP, pre-norm block, residuals, weight tying, init, forward → (logits, loss) | param count = formula; init loss ≈ ln(4096) = 8.32; overfits one batch < 0.1 |
| A5 | Training loop | AdamW + decay groups, warmup + cosine LR, grad clip, grad accumulation, eval, checkpoint/resume | debug run converges; resume reproduces loss curve; first `base` run (~1–2 h) |
| A6 | Inference | temperature / top-k / top-p sampling; KV cache | cached greedy == uncached greedy; speedup measured |
| A7 | Evaluation | val loss, perplexity, bits-per-byte; fixed-prompt sample set; 2–3-size scaling study | results table in STATE.md |
| A8 | Modernization (optional) | RMSNorm, RoPE, SwiGLU as isolated ablations | Δ val bits-per-byte vs baseline per change |

## Model presets (planned; finalized in A4)

| Preset | vocab | ctx | d_model | layers | heads | params (approx.) |
|---|---|---|---|---|---|---|
| debug | 4096 | 128 | 128 | 2 | 4 | ~0.9 M |
| base | 4096 | 256* | 384 | 6 | 6 | ~12.3 M |
| stretch (optional, 16 GB) | 4096 | 256* | 512 | 8 | 8 | ~27 M |

\* 256 vs 512 decided from the tokens-per-story distribution measured in A1.

Out of scope for Phase 1: fine-tuning pretrained models, chat UI.

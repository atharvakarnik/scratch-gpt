# STATE — authoritative project status

**Current milestone:** M0 (environment check on Mac) → then **A1** (preprocessing + BPE tokenizer)
**Last updated:** 2026-11-10

## Hardware (verified, 2026-11-10)

| Item | Value |
|---|---|
| Machine | MacBook Air M2, fanless |
| Unified memory | 16 GB |
| GPU cores | 10 |
| Free disk | ~18 GB (96% used) — **constraint**: keep project ≤ ~5 GB |
| macOS | 27.0.1 |
| Python | 3.13 in mamba env `scratch-gpt` (`environment.yml`) |

## Decision log

| ID | Decision | Rationale | Revisit if |
|---|---|---|---|
| D1 | PyTorch on MPS, FP32 default | Transfers to later fine-tuning phase; correctness first | measured throughput < ~5k tok/s (consider MLX) |
| D2 | Dataset: TinyStoriesV2-GPT4 (HF `roneneldan/TinyStories`, CDLA-Sharing-1.0) | Narrow domain → coherent text from ~10M-param models; ~550M tokens (est.) | — |
| D3 | Self-implemented byte-level BPE, vocab 4096 (incl. `<\|endoftext\|>`) | Small embedding/logit cost; core learning exercise | encode throughput blocks A2 |
| D4 | Baseline arch: GPT-2-style pre-norm (LayerNorm, learned pos-emb, GELU MLP, tied embeddings) | Simple to verify; modern parts become ablations (A8) | — |
| D5 | Manual attention for learning; SDPA for training runs | Verify manual vs fused | — |
| D6 | Deps: torch, numpy, regex; dev: pytest. Env: mamba (`environment.yml`: Python 3.13 + pip); packages via pip from `pyproject.toml` | Minimal; one dependency source; official torch MPS wheels | — |
| D7 | Data never committed; raw train text may be deleted after tokenization | License + disk | — |

## Verified results

_None on the Mac yet._ Pending M0: `python scripts/bench_mps.py` (record tok/s, achieved TFLOP/s,
device memory) and `python scripts/bench_mps.py --duration 600` (thermal throttling).

Scaffold verification (2026-11-10, Linux CPU sandbox, not the Mac):
- A1 tests validated against a private reference implementation (not in repo); 9 injected bugs all caught.
- `download_data.sh` tested against a mock server (fetch, skip-existing, disk guard, bad args).
- `bench_mps.py` stand-in: 12.32 M params for `base`, initial loss ≈ ln(4096) as expected.

## Planning estimates (not yet verified)

- `base` (~12.3M params): ~80 MFLOP/token training; ~10k tok/s on M2 GPU → ~36M tokens/h.
- Peak memory (micro-batch 32 × ctx 256, FP32): ~2–3 GB.
- Fanless thermal throttling: magnitude unknown — measure in long runs.

## Open questions

- Cleaning policy for encoding artifacts / non-ASCII characters (decide in A1 from `inspect_data.py`).
- Context length 256 vs 512 (decide in A1 from tokens-per-story percentiles).

# STATE — authoritative project status

**Current milestone:** **A1** (preprocessing + BPE tokenizer). M0 complete.
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
| D1 | PyTorch on MPS, FP32 (no mixed precision) | Transfers to fine-tuning phase. M0: FP16 matmul only 1.12× FP32, BF16 0.54× (slower on M2) → AMP not worth its complexity | moving to a newer Apple GPU |
| D2 | Dataset: TinyStoriesV2-GPT4 (HF `roneneldan/TinyStories`, CDLA-Sharing-1.0) | Narrow domain → coherent text from ~10M-param models; ~550M tokens (est.) | — |
| D3 | Self-implemented byte-level BPE, vocab 4096 (incl. `<\|endoftext\|>`) | Small embedding/logit cost; core learning exercise | encode throughput blocks A2 |
| D4 | Baseline arch: GPT-2-style pre-norm (LayerNorm, learned pos-emb, GELU MLP, tied embeddings) | Simple to verify; modern parts become ablations (A8) | — |
| D5 | Manual attention for learning; SDPA for training runs | Verify manual vs fused | — |
| D6 | Deps: torch, numpy, regex; dev: pytest. Env: mamba (`environment.yml`: Python 3.13 + pip); packages via pip from `pyproject.toml` | Minimal; one dependency source; official torch MPS wheels | — |
| D7 | Data never committed; raw train text may be deleted after tokenization | License + disk | — |

## Verified results

### M0 — 2026-11-10, MacBook Air M2 (10-core GPU), torch 2.15.0, MPS, stand-in `base` (12.32 M params, B=32, T=256)

| Metric | Value |
|---|---|
| Matmul 2048² FP32 / FP16 / BF16 | 2.89 / 3.24 / 1.56 TFLOP/s |
| Training throughput, cold (60 steps) | 17.5k tok/s (1.40 TFLOP/s ≈ 48% of FP32 matmul) |
| Training throughput, 10-min sustained | 13.6k tok/s avg; windows 11.2k–17.9k; last vs first −26.5% (thermal) |
| Device memory (driver-allocated) | 3.11 GiB |
| Initial loss | 8.40 (ln 4096 = 8.32) |

Implications: plan on **~12k tok/s sustained → ~45 M tokens/h** for `base`. 1 h first run ≈ 45 M tokens;
Chinchilla-scale (~250 M tokens) ≈ 5–6 h. `stretch` (~27 M) expected ~2.2× slower → only as an
undertrained comparison. Memory is not a constraint at B=32. Your hand-written model may be slower
than the fused stand-in; re-measure in A5.

Scaffold verification (2026-11-10, Linux CPU sandbox, not the Mac):
- A1 tests validated against a private reference implementation (not in repo); 9 injected bugs all caught.
- `download_data.sh` tested against a mock server (fetch, skip-existing, disk guard, bad args).
- `bench_mps.py` stand-in: 12.32 M params for `base`, initial loss ≈ ln(4096) as expected.

## Planning estimates (not yet verified)

- Thermal throttling beyond 10 min (multi-hour runs): unmeasured. Compare runs by tokens seen, not wall time.

## Open questions

- Cleaning policy for encoding artifacts / non-ASCII characters (decide in A1 from `inspect_data.py`).
- Context length 256 vs 512 (decide in A1 from tokens-per-story percentiles).

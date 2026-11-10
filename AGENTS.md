# AGENTS.md — rules for coding agents (Codex, Claude, etc.)

This is an **educational** repository. The human owner is learning LLM engineering by
implementing the core components personally. Agents support that; they do not replace it.

## Ownership rules (strict)

1. Code marked `# LEARNER-OWNED` (module header or function docstring) is the owner's
   assignment. **Never implement, complete, or "fix" its TODOs** unless the owner explicitly
   asks for that specific function in the current conversation.
2. For learner-owned code, agents may: explain concepts, review diffs, point to the failing
   line/test, suggest a direction in prose or pseudocode, add *extra* tests, and debug when asked.
3. Agents may freely write non-educational code: scripts in `scripts/`, CLI glue, logging,
   plotting, packaging, CI — when asked.
4. Never weaken, delete, or skip an existing test to make it pass. Propose test changes
   explicitly with a reason.
5. Do not start a future milestone's files. Scope is the current milestone in `docs/STATE.md`.

## Source of truth

- `docs/STATE.md` — current milestone, verified results, decision log. Authoritative.
- `docs/ROADMAP.md` — milestone plan.
- `docs/assignments/A*.md` — specs and acceptance criteria per assignment.
- Previous chat proposals are not implemented unless present in the repo.

When reporting results, label them **verified** (measured/tested here), **expected**
(theory/literature), or **assumed** (untested).

## Commands

```bash
mamba activate scratch-gpt  # env from environment.yml; packages via `pip install -e ".[dev]"`
pytest                      # all tests (dataset-dependent tests skip if data missing)
pytest -m "not slow"        # quick loop
python scripts/bench_mps.py # M0 throughput/memory benchmark
bash scripts/download_data.sh          # validation split only (~22.5 MB)
bash scripts/download_data.sh --train  # training split (~2.23 GB), checks free disk first
```

## Conventions

- Python ≥ 3.11, PyTorch (MPS on Apple Silicon, CPU fallback), FP32 by default.
- Dependencies: `torch`, `numpy`, `regex`; dev: `pytest` — declared only in `pyproject.toml`,
  installed with pip into the `scratch-gpt` mamba env (conda supplies Python + pip only).
  Adding a dependency needs a reason recorded in `docs/STATE.md`.
- Never commit anything under `data/`, `runs/`, `artifacts/` (dataset license: CDLA-Sharing-1.0;
  also size). Disk on the dev machine is limited (~18 GB free): keep project footprint ≤ ~5 GB.

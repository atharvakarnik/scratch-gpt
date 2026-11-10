# scratch-gpt

A small decoder-only Transformer (GPT) implemented and trained from scratch on a
MacBook Air M2, for learning — not production. Trained on TinyStories V2 (GPT-4 subset)
with a self-implemented byte-level BPE tokenizer.

- Plan: [`docs/ROADMAP.md`](docs/ROADMAP.md)
- Current status, verified results, decisions: [`docs/STATE.md`](docs/STATE.md)
- Current assignment: [`docs/assignments/`](docs/assignments/)
- Agent rules: [`AGENTS.md`](AGENTS.md)

## Setup (macOS, Apple Silicon)

```bash
cd "/Users/atharvakarnik/IU/Codes and Repos/scratch-gpt"
mamba env create -f environment.yml   # Python 3.13 + pip (conda-forge)
mamba activate scratch-gpt
pip install -e ".[dev]"               # torch, numpy, regex, pytest (from pyproject.toml)
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
pytest -m "not slow"                  # learner TODO tests fail until implemented — expected
```

Conda provides only the interpreter; all Python packages come from `pyproject.toml` via pip
(one source of truth; official PyTorch macOS wheels with MPS). Don't `mamba install` Python
packages into this env, to avoid conda/pip conflicts.

## Data

```bash
bash scripts/download_data.sh           # TinyStoriesV2-GPT4-valid.txt (~22.5 MB)
bash scripts/download_data.sh --train   # TinyStoriesV2-GPT4-train.txt (~2.23 GB); needs free disk
```

Files land in `data/raw/tinystories_v2/`. The dataset is licensed CDLA-Sharing-1.0
(`roneneldan/TinyStories` on Hugging Face); it is git-ignored and must not be redistributed
from this repo.

## Layout

```
configs/        training configs (added in A4/A5)
docs/           STATE, ROADMAP, assignment specs
scripts/        non-educational tooling (download, benchmarks, CLIs)
src/scratchgpt/ the model/training code — learner-owned modules
tests/          pytest suites per assignment
data/ runs/ artifacts/   git-ignored
```

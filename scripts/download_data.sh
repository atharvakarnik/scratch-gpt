#!/usr/bin/env bash
# Download TinyStories V2 (GPT-4-generated subset) into data/raw/tinystories_v2/.
# Source: https://huggingface.co/datasets/roneneldan/TinyStories (license: CDLA-Sharing-1.0)
#
#   bash scripts/download_data.sh           # validation split only (~22.5 MB)
#   bash scripts/download_data.sh --train   # also the training split (~2.23 GB)
#
# Downloads are resumable (re-run after interruption). Existing complete files are skipped.
set -euo pipefail

BASE_URL="${TINYSTORIES_BASE_URL:-https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/data/raw/tinystories_v2"
VALID="TinyStoriesV2-GPT4-valid.txt"
TRAIN="TinyStoriesV2-GPT4-train.txt"
MIN_FREE_GB_FOR_TRAIN=8   # 2.23 GB download + ~1.1 GB tokenized + checkpoints + OS headroom

want_train=0
for arg in "$@"; do
  case "$arg" in
    --train) want_train=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$DEST"

free_gb() { df -Pk "$DEST" | awk 'NR==2 { printf "%d", $4 / 1048576 }'; }

fetch() {
  local name="$1" out="$DEST/$1"
  if [[ -f "$out" ]]; then
    echo "exists: $out ($(du -h "$out" | cut -f1))"
    return 0
  fi
  echo "downloading $name ..."
  # -C - resumes a partial .part file; --fail aborts on HTTP errors instead of saving an error page
  curl -L --fail --retry 3 --retry-delay 2 -C - -o "$out.part" "$BASE_URL/$name"
  mv "$out.part" "$out"
  echo "saved: $out ($(du -h "$out" | cut -f1))"
}

fetch "$VALID"

if [[ "$want_train" -eq 1 ]]; then
  if [[ ! -f "$DEST/$TRAIN" ]]; then
    free="$(free_gb)"
    if (( free < MIN_FREE_GB_FOR_TRAIN )); then
      echo "refusing: only ${free} GB free; need >= ${MIN_FREE_GB_FOR_TRAIN} GB for the training split." >&2
      exit 1
    fi
    echo "free disk: ${free} GB (ok)"
  fi
  fetch "$TRAIN"
fi

echo "done. files in $DEST:"
ls -lh "$DEST"

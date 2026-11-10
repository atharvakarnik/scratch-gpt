"""A1 helper: train your BPE on cleaned stories, save it, and report the numbers A1 asks for.
(Scaffold — not learner-owned. All tokenizer logic is yours, in src/scratchgpt/tokenizer.py.)

    python scripts/train_tokenizer.py                    # valid split, vocab 4096
    python scripts/train_tokenizer.py --vocab-size 1024 --max-stories 5000   # quick iteration
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from scratchgpt.text import EOT, clean_story, iter_stories
from scratchgpt.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "data/raw/tinystories_v2/TinyStoriesV2-GPT4-valid.txt"


def pct(xs: list[int], q: float) -> int:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT)
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--max-stories", type=int, default=0, help="cap on stories read (0 = all)")
    ap.add_argument("--heldout", type=int, default=2000, help="last N stories held out for evaluation")
    ap.add_argument("--out", type=Path, help="default: artifacts/tokenizer/bpe<V>.json")
    args = ap.parse_args()
    out = args.out or ROOT / f"artifacts/tokenizer/bpe{args.vocab_size}.json"

    t0 = time.perf_counter()
    stories = []
    for s in iter_stories(args.input):
        c = clean_story(s)
        if c is not None:
            stories.append(c)
        if args.max_stories and len(stories) >= args.max_stories:
            break
    if len(stories) <= args.heldout:
        raise SystemExit(f"only {len(stories)} stories; reduce --heldout")
    train, held = stories[: -args.heldout], stories[-args.heldout:]
    train_bytes = sum(len(s.encode("utf-8")) for s in train)
    print(f"loaded {len(stories):,} cleaned stories in {time.perf_counter() - t0:.1f}s "
          f"(train {len(train):,} / {train_bytes / 1e6:.1f} MB, held-out {len(held):,})")

    t0 = time.perf_counter()
    tok = BPETokenizer.train(train, vocab_size=args.vocab_size, special_tokens=[EOT])
    t_train = time.perf_counter() - t0
    tok.save(out)
    print(f"trained {len(tok.merges):,} merges in {t_train:.1f}s → {out}")

    v = tok.vocab
    show = lambda ids: ", ".join(repr(v[i].decode("utf-8", errors="replace")) for i in ids)  # noqa: E731
    merge_ids = list(tok.merges.values())
    print(f"first merges: {show(merge_ids[:15])}")
    print(f"last merges:  {show(merge_ids[-15:])}")
    longest = sorted(merge_ids, key=lambda i: len(v[i]), reverse=True)[:10]
    print(f"longest tokens: {show(longest)}")

    t0 = time.perf_counter()
    encoded = [tok.encode_ordinary(s) for s in held]
    t_enc = time.perf_counter() - t0
    held_bytes = sum(len(s.encode("utf-8")) for s in held)
    n_tok = sum(map(len, encoded))
    bad = sum(tok.decode(ids) != s for ids, s in zip(encoded, held))
    print(f"\nheld-out round-trip failures: {bad}")
    print(f"held-out bytes/token: {held_bytes / n_tok:.3f}  | encode speed: {held_bytes / t_enc / 1e6:.2f} MB/s")

    lens = [len(ids) + 1 for ids in encoded]  # +1 for the EOT separator
    print(f"tokens/story (+EOT): p50 {pct(lens, .5)} | p90 {pct(lens, .9)} | p99 {pct(lens, .99)} | max {max(lens)}")
    for ctx in (256, 512):
        frac = sum(n > ctx for n in lens) / len(lens)
        print(f"  stories longer than ctx={ctx}: {frac:.1%}")


if __name__ == "__main__":
    main()

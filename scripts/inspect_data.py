"""A1 helper: corpus statistics to decide the cleaning policy. (Scaffold — not learner-owned.)

Needs your `iter_stories`; also reports post-cleaning stats once `clean_story` is implemented.

    python scripts/inspect_data.py                      # validation split
    python scripts/inspect_data.py --limit 20000 --top 50
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

import regex

from scratchgpt.text import clean_story, iter_stories

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "data/raw/tinystories_v2/TinyStoriesV2-GPT4-valid.txt"
# Typical UTF-8-read-as-cp1252 artifacts: â€™ â€œ â€” Ã© Â …  (heuristic, expect false positives)
MOJIBAKE = regex.compile(r"â€|Ã[\x80-\xBF\p{L}]|Â[\s\p{P}]")


def pct(xs: list[int], q: float) -> int:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def describe(name: str, xs: list[int]) -> str:
    return (f"{name:>14}: mean {statistics.fmean(xs):8.1f} | p50 {pct(xs, .5):6d} | p90 {pct(xs, .9):6d}"
            f" | p99 {pct(xs, .99):6d} | max {max(xs):6d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT)
    ap.add_argument("--limit", type=int, default=0, help="max stories (0 = all)")
    ap.add_argument("--top", type=int, default=40, help="non-ASCII characters to list")
    args = ap.parse_args()

    chars, words, nbytes = [], [], 0
    hashes: Counter[str] = Counter()
    non_ascii: Counter[str] = Counter()
    stories_with: Counter[str] = Counter()
    example: dict[str, str] = {}
    n_fffd = n_moji = 0
    moji_examples: list[str] = []
    cleaned_kept = cleaned_dropped = 0
    cleaned_non_ascii: Counter[str] = Counter()
    clean_ok = True

    for i, s in enumerate(iter_stories(args.input)):
        if args.limit and i >= args.limit:
            break
        chars.append(len(s))
        words.append(len(s.split()))
        nbytes += len(s.encode("utf-8"))
        hashes[hashlib.md5(s.encode("utf-8")).hexdigest()] += 1
        n_fffd += "�" in s
        m = MOJIBAKE.search(s)
        if m:
            n_moji += 1
            if len(moji_examples) < 5:
                moji_examples.append(s[max(0, m.start() - 30): m.end() + 30].replace("\n", " "))
        seen = set()
        for j, ch in enumerate(s):
            if ord(ch) > 127:
                non_ascii[ch] += 1
                if ch not in seen:
                    stories_with[ch] += 1
                    seen.add(ch)
                example.setdefault(ch, s[max(0, j - 25): j + 25].replace("\n", " "))
        if clean_ok:
            try:
                c = clean_story(s)
            except NotImplementedError:
                clean_ok = False
                continue
            if c is None:
                cleaned_dropped += 1
            else:
                cleaned_kept += 1
                cleaned_non_ascii.update(ch for ch in c if ord(ch) > 127)

    n = len(chars)
    if n == 0:
        print("no stories found")
        return
    print(f"file: {args.input}")
    print(f"stories: {n:,} | UTF-8 bytes: {nbytes / 1e6:.1f} MB | chars: {sum(chars):,} | words: {sum(words):,}")
    print(describe("chars/story", chars))
    print(describe("words/story", words))
    dups = sum(c - 1 for c in hashes.values() if c > 1)
    print(f"exact duplicate stories: {dups:,} ({dups / n:.2%})")
    print(f"stories with U+FFFD (undecodable bytes): {n_fffd:,}")
    print(f"stories with mojibake candidates: {n_moji:,} ({n_moji / n:.2%})")
    for ex in moji_examples:
        print(f"    …{ex}…")

    total_na = sum(non_ascii.values())
    print(f"\nnon-ASCII chars: {total_na:,} total, {len(non_ascii)} distinct")
    print(f"{'char':>6} {'U+':>7} {'count':>9} {'stories':>8}  name / example")
    for ch, cnt in non_ascii.most_common(args.top):
        name = unicodedata.name(ch, "?")
        print(f"{ch!r:>6} {ord(ch):07X} {cnt:9,d} {stories_with[ch]:8,d}  {name}  | …{example[ch]}…")

    if clean_ok:
        print(f"\nafter clean_story: kept {cleaned_kept:,}, dropped {cleaned_dropped:,} "
              f"({cleaned_dropped / n:.2%}); remaining non-ASCII distinct: {len(cleaned_non_ascii)}")
        if cleaned_non_ascii:
            print("  top remaining:", ", ".join(f"{c!r}×{k}" for c, k in cleaned_non_ascii.most_common(15)))
    else:
        print("\n(clean_story not implemented yet — post-cleaning stats skipped)")


if __name__ == "__main__":
    main()

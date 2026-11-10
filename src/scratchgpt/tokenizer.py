"""A1 — byte-level BPE tokenizer.

# LEARNER-OWNED: implement the TODOs yourself (see docs/assignments/A1.md). Agents: do not fill in.
# Provided (not learner-owned): GPT2_SPLIT_PATTERN, pretokenize, vocab_size, save, load.

Token-id layout for vocab_size V with S special tokens:
    0 .. 255              raw bytes
    256 .. V-S-1          merges, in the order they were learned (id = rank)
    V-S .. V-1            special tokens, in the order given to `train`
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import regex

from scratchgpt.text import EOT

# GPT-2 pre-tokenization pattern: contractions, optional-space + letters, optional-space + digits,
# optional-space + punctuation runs, whitespace. Merges never cross these chunk boundaries.
GPT2_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

Pair = tuple[int, int]


def pretokenize(text: str, pattern: str = GPT2_SPLIT_PATTERN) -> list[str]:
    """Split text into chunks; BPE runs inside each chunk independently. (Provided.)"""
    return regex.findall(pattern, text)


def get_pair_counts(word_freqs: dict[tuple[int, ...], int]) -> Counter[Pair]:
    """Count adjacent id pairs across all words, weighted by word frequency.

    Example: {(1, 2, 3): 2, (2, 3): 1} -> {(1, 2): 2, (2, 3): 3}
    Overlaps count: {(1, 1, 1): 1} -> {(1, 1): 2}
    """
    # TODO(A1)
    raise NotImplementedError


def merge_pair(ids: tuple[int, ...], pair: Pair, new_id: int) -> tuple[int, ...]:
    """Replace every occurrence of `pair` in `ids` with `new_id`, scanning left to right,
    non-overlapping. Example: merge_pair((1, 1, 1), (1, 1), 256) -> (256, 1)
    """
    # TODO(A1)
    raise NotImplementedError


class BPETokenizer:
    def __init__(
        self,
        merges: dict[Pair, int],
        special_tokens: dict[str, int],
        pattern: str = GPT2_SPLIT_PATTERN,
    ):
        """merges: pair -> new id, insertion-ordered by rank. special_tokens: str -> id."""
        self.merges = dict(merges)
        self.special_tokens = dict(special_tokens)
        self.pattern = pattern
        self._vocab: dict[int, bytes] | None = None

    # ------------------------------------------------------------------ training
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        special_tokens: list[str] = [EOT],  # noqa: B006 — never mutated
        pattern: str = GPT2_SPLIT_PATTERN,
    ) -> BPETokenizer:
        """Learn merges from `texts` (each text = one document; specials are NOT parsed here).

        Algorithm contract:
          1. Pretokenize every text; count chunk frequencies (word-frequency trick:
             work on unique chunks, not on the raw stream).
          2. Represent each unique chunk as a tuple of its UTF-8 byte values.
          3. Repeat (vocab_size - 256 - len(special_tokens)) times:
               - count pairs (weighted by chunk frequency);
               - stop early if no pairs remain;
               - pick the most frequent pair; ties -> the smallest pair, compared as (int, int);
               - assign the next id, apply the merge to every chunk.
          4. Special tokens get ids immediately after the last *possible* merge id, so that
             special ids are always V-S .. V-1 even if training stopped early.
        Raise ValueError if vocab_size < 256 + len(special_tokens).
        """
        # TODO(A1)
        raise NotImplementedError

    # ------------------------------------------------------------------ vocab
    @property
    def vocab(self) -> dict[int, bytes]:
        """id -> bytes for raw bytes and merged tokens (not special tokens). Cache the result."""
        # TODO(A1)
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """Total id space, including special tokens. (Provided.)"""
        ids = [255, *self.merges.values(), *self.special_tokens.values()]
        return max(ids) + 1

    # ------------------------------------------------------------------ encode / decode
    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text, treating special-token strings as ordinary text.

        Per chunk: start from bytes; repeatedly merge the adjacent pair with the lowest merge id
        (= earliest learned) until no adjacent pair is in `self.merges`.
        """
        # TODO(A1)
        raise NotImplementedError

    def encode(
        self,
        text: str,
        allowed_special: Literal["all", "none"] | set[str] = "none",
    ) -> list[int]:
        """Like encode_ordinary, but occurrences of allowed special-token strings map to their id.

        allowed_special: "all", "none", or a subset of self.special_tokens keys.
        """
        # TODO(A1)
        raise NotImplementedError

    def decode(self, ids: Iterable[int]) -> str:
        """Concatenate token bytes (special ids -> their UTF-8 string) and decode with
        errors="replace", so partial/invalid UTF-8 becomes U+FFFD instead of raising."""
        # TODO(A1)
        raise NotImplementedError

    # ------------------------------------------------------------------ persistence (provided)
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "scratchgpt-bpe-v1",
            "pattern": self.pattern,
            "merges": [[a, b, i] for (a, b), i in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "scratchgpt-bpe-v1":
            raise ValueError(f"unknown tokenizer format in {path}")
        merges = {(a, b): i for a, b, i in payload["merges"]}
        return cls(merges, payload["special_tokens"], payload["pattern"])

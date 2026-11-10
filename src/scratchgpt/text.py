"""A1 — corpus parsing and cleaning.

# LEARNER-OWNED: implement the TODOs yourself (see docs/assignments/A1.md). Agents: do not fill in.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

EOT = "<|endoftext|>"


def iter_stories(
    path: str | Path,
    delimiter: str = EOT,
    chunk_size: int = 1 << 20,
) -> Iterator[str]:
    """Stream stories from a delimiter-separated UTF-8 text file.

    Contract (tested in tests/test_text.py):
      - Must be a generator: read the file incrementally, `chunk_size` *characters* per read.
        Never load the whole file (the train split is 2.2 GB).
      - Open with encoding="utf-8", errors="replace": undecodable bytes become U+FFFD.
      - Split on `delimiter`. A delimiter may straddle two reads — handle it.
      - Yield each story with leading/trailing whitespace stripped.
      - Skip parts that are empty after stripping (e.g. leading/trailing/consecutive delimiters).
      - The delimiter text itself never appears in a yielded story.
    """
    # TODO(A1): implement. Hint: keep a carry-over buffer between reads.
    raise NotImplementedError


def clean_story(text: str) -> str | None:
    """Normalize one story, or return None to drop it.

    Required contract (tested):
      - "\\r\\n" and "\\r" become "\\n".
      - Result is stripped of leading/trailing whitespace.
      - Return None if the result is empty.
      - Return None if the story contains U+FFFD (undecodable bytes).
      - Idempotent: clean_story(clean_story(x)) == clean_story(x) whenever the first is not None.

    Your policy (decide from `scripts/inspect_data.py` output, document in docs/STATE.md,
    and add tests for it in tests/test_text.py):
      - What to do with mojibake (e.g. "â€™"), curly quotes, dashes, other non-ASCII characters:
        map to ASCII, keep, or drop the story.
    """
    # TODO(A1): implement.
    raise NotImplementedError

"""A1 tests — corpus parsing and cleaning."""

from collections.abc import Iterator

import pytest

from scratchgpt.text import EOT, clean_story, iter_stories

RAW = f"A story.\n{EOT}\nSecond story,\nwith two lines.\n{EOT}\n\n{EOT}\n   \n{EOT}\nThird.\n"
EXPECTED = ["A story.", "Second story,\nwith two lines.", "Third."]


@pytest.fixture
def corpus(tmp_path):
    p = tmp_path / "stories.txt"
    p.write_text(RAW, encoding="utf-8")
    return p


# ---------------------------------------------------------------- iter_stories
def test_basic_split(corpus):
    assert list(iter_stories(corpus)) == EXPECTED


def test_is_lazy_generator(corpus):
    it = iter_stories(corpus)
    assert isinstance(it, Iterator)
    assert next(it) == "A story."


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 7, 11, 13, 64])
def test_delimiter_straddles_read_boundary(corpus, chunk_size):
    assert list(iter_stories(corpus, chunk_size=chunk_size)) == EXPECTED


def test_leading_and_missing_trailing_delimiter(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text(f"{EOT}one{EOT}two", encoding="utf-8")
    assert list(iter_stories(p, chunk_size=4)) == ["one", "two"]


def test_custom_delimiter(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("a||b||||c", encoding="utf-8")
    assert list(iter_stories(p, delimiter="||", chunk_size=3)) == ["a", "b", "c"]


def test_no_delimiter_in_output(corpus):
    assert all(EOT not in s for s in iter_stories(corpus, chunk_size=5))


def test_invalid_utf8_becomes_replacement_char(tmp_path):
    p = tmp_path / "s.txt"
    p.write_bytes(b"ok\n" + EOT.encode() + b"\nbad \xff byte\n")
    stories = list(iter_stories(p))
    assert stories[0] == "ok"
    assert "�" in stories[1]


def test_empty_file(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("", encoding="utf-8")
    assert list(iter_stories(p)) == []


# ---------------------------------------------------------------- clean_story
def test_clean_strips_and_normalizes_newlines():
    assert clean_story("  Line one.\r\nLine two.\rLine three.  \n") == "Line one.\nLine two.\nLine three."


@pytest.mark.parametrize("text", ["", "   ", "\n\r\n\t"])
def test_clean_drops_empty(text):
    assert clean_story(text) is None


def test_clean_drops_undecodable():
    assert clean_story("Once upon a � time.") is None


def test_clean_returns_str_for_plain_story():
    out = clean_story("Once upon a time, there was a cat. The end.")
    assert out == "Once upon a time, there was a cat. The end."


@pytest.mark.parametrize(
    "text",
    [
        "Plain ASCII story.",
        "Quotes: “hello” and it’s fine — really.",
        "Mojibake: itâ€™s here.",
        "Accents: café naïve.",
        "Para one.\r\n\r\nPara two.",
    ],
)
def test_clean_is_idempotent(text):
    once = clean_story(text)
    if once is not None:
        assert clean_story(once) == once


def test_your_cleaning_policy():
    # TODO(A1): replace this skip with tests that pin down YOUR policy for
    # mojibake / curly quotes / dashes / other non-ASCII (decided from inspect_data.py).
    pytest.skip("TODO(A1): add tests for your cleaning policy")

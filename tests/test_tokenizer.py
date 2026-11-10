"""A1 tests — byte-level BPE tokenizer."""

from pathlib import Path

import pytest

from scratchgpt.text import EOT, clean_story, iter_stories
from scratchgpt.tokenizer import BPETokenizer, get_pair_counts, merge_pair, pretokenize

# Small original corpus for training tests (deliberately repetitive so merges are meaningful).
CORPUS = [
    "Once upon a time, there was a little dog named Max. Max liked to run in the park.",
    "One day, Max saw a big red ball. He ran to the ball and played with it all day.",
    "The little girl, Lily, saw Max and the ball. She smiled and said, \"Can I play too?\"",
    "Max wagged his tail. They played together in the park until the sun went down.",
    "At night, Max was tired but happy. He slept next to the red ball and dreamed.",
] * 4

ROUNDTRIP_CASES = [
    "",
    "Max",
    "Once upon a time, a zebra named Quixote juggled 12345 apples!",
    "naïve café — “quotes” 🙂 日本語",
    "  leading and trailing spaces  ",
    "tabs\tand\nnewlines\n\n\npara",
    "it's we'll they're I'd you've",
    "\x00\x01 control chars \x7f",
]


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    return BPETokenizer.train(CORPUS, vocab_size=320, special_tokens=[EOT])


# ---------------------------------------------------------------- helpers
def test_get_pair_counts_weighted():
    assert get_pair_counts({(1, 2, 3): 2, (2, 3): 1}) == {(1, 2): 2, (2, 3): 3}


def test_get_pair_counts_overlapping_and_singletons():
    assert get_pair_counts({(1, 1, 1): 1, (7,): 5}) == {(1, 1): 2}


@pytest.mark.parametrize(
    "ids, pair, expected",
    [
        ((1, 2, 3, 1, 2), (1, 2), (256, 3, 256)),
        ((1, 1, 1), (1, 1), (256, 1)),
        ((1, 1, 1, 1), (1, 1), (256, 256)),
        ((1, 2, 3), (3, 1), (1, 2, 3)),
        ((), (1, 2), ()),
        ((1,), (1, 2), (1,)),
    ],
)
def test_merge_pair(ids, pair, expected):
    assert merge_pair(ids, pair, 256) == expected


# ---------------------------------------------------------------- training (hand-computed)
def test_train_toy_merges_hand_computed():
    # "aaabdaaabac": counts (a,a)=4 -> 256; then (a,b)=2 ties (256,a)=2 -> smaller pair (97,98) -> 257;
    # then (256,257)=2 -> 258. Special token takes the last id.
    t = BPETokenizer.train(["aaabdaaabac"], vocab_size=260, special_tokens=[EOT])
    assert list(t.merges.items()) == [((97, 97), 256), ((97, 98), 257), ((256, 257), 258)]
    assert t.special_tokens == {EOT: 259}
    assert t.vocab_size == 260


def test_encode_toy_matches_training_segmentation():
    t = BPETokenizer.train(["aaabdaaabac"], vocab_size=260, special_tokens=[EOT])
    assert t.encode_ordinary("aaabdaaabac") == [258, 100, 258, 97, 99]


def test_encode_applies_merges_in_rank_order():
    # (b,c) was learned first, so it must win over (a,b) — even though "abc" exists as id 258.
    # Catches "apply highest id first" and "greedy longest match" encoders.
    t = BPETokenizer({(98, 99): 256, (97, 98): 257, (257, 99): 258}, {})
    assert t.encode_ordinary("abc") == [97, 256]
    assert t.encode_ordinary("abd") == [257, 100]


def test_train_stops_early_but_keeps_special_id_at_end():
    t = BPETokenizer.train(["ab"], vocab_size=300, special_tokens=[EOT])
    assert list(t.merges.items()) == [((97, 98), 256)]
    assert t.special_tokens == {EOT: 299}
    assert t.vocab_size == 300


def test_train_rejects_too_small_vocab():
    with pytest.raises(ValueError):
        BPETokenizer.train(["abc"], vocab_size=256, special_tokens=[EOT])


def test_train_is_deterministic():
    a = BPETokenizer.train(CORPUS, vocab_size=300)
    b = BPETokenizer.train(CORPUS, vocab_size=300)
    assert a.merges == b.merges


def test_merges_never_cross_pretokenization_boundaries(tok):
    # GPT-2 pattern attaches a space to the *following* word, so no merged non-whitespace token
    # may contain a space after its first byte, or span letters followed by punctuation.
    for i in range(256, 256 + len(tok.merges)):
        b = tok.vocab[i]
        if not b.isspace():
            assert b" " not in b[1:], b
        assert len(pretokenize(b.decode("utf-8", errors="ignore"))) <= 1, b


def test_merge_ids_are_contiguous_and_ordered(tok):
    assert list(tok.merges.values()) == list(range(256, 256 + len(tok.merges)))
    for (a, b), new in tok.merges.items():
        assert a < new and b < new  # can only merge tokens that already exist


# ---------------------------------------------------------------- vocab
def test_vocab(tok):
    v = tok.vocab
    assert len(v) == 256 + len(tok.merges)
    assert all(v[i] == bytes([i]) for i in range(256))
    for (a, b), new in tok.merges.items():
        assert v[new] == v[a] + v[b]


# ---------------------------------------------------------------- encode / decode
@pytest.mark.parametrize("text", ROUNDTRIP_CASES)
def test_roundtrip(tok, text):
    ids = tok.encode(text)
    assert all(0 <= i < tok.vocab_size for i in ids)
    assert tok.decode(ids) == text


def test_compresses_training_text(tok):
    text = " ".join(CORPUS)
    assert len(tok.encode(text)) < 0.6 * len(text.encode("utf-8"))


def test_special_tokens_ignored_by_default(tok):
    eot = tok.special_tokens[EOT]
    text = f"The end.{EOT}Once"
    ids = tok.encode(text)
    assert eot not in ids
    assert ids == tok.encode_ordinary(text)
    assert tok.decode(ids) == text


@pytest.mark.parametrize("allowed", ["all", {EOT}])
def test_special_tokens_when_allowed(tok, allowed):
    eot = tok.special_tokens[EOT]
    text = f"The end.{EOT}Once upon a time.{EOT}"
    ids = tok.encode(text, allowed_special=allowed)
    assert ids.count(eot) == 2
    assert ids[-1] == eot
    assert tok.decode(ids) == text


def test_decode_invalid_utf8_is_replaced(tok):
    assert tok.decode([0xFF]) == "�"
    assert tok.decode([0xC3]) == "�"  # first byte of "é" alone
    assert tok.decode([0xC3, 0xA9]) == "é"


# ---------------------------------------------------------------- persistence
def test_save_load_roundtrip(tok, tmp_path):
    path = tmp_path / "tok.json"
    tok.save(path)
    t2 = BPETokenizer.load(path)
    assert t2.merges == tok.merges and t2.special_tokens == tok.special_tokens
    for text in ROUNDTRIP_CASES:
        assert t2.encode(text) == tok.encode(text)


# ---------------------------------------------------------------- real data (optional)
VALID = Path(__file__).resolve().parents[1] / "data/raw/tinystories_v2/TinyStoriesV2-GPT4-valid.txt"


@pytest.mark.data
@pytest.mark.slow
@pytest.mark.skipif(not VALID.exists(), reason="run scripts/download_data.sh first")
def test_real_data_smoke():
    stories = []
    for s in iter_stories(VALID):
        c = clean_story(s)
        if c is not None:
            stories.append(c)
        if len(stories) >= 1200:
            break
    train, held = stories[:1000], stories[1000:]
    t = BPETokenizer.train(train, vocab_size=512)
    n_bytes = sum(len(s.encode("utf-8")) for s in held)
    n_tok = sum(len(t.encode(s)) for s in held)
    assert all(t.decode(t.encode(s)) == s for s in held)
    assert n_bytes / n_tok > 2.0  # sanity floor; record the real number for vocab 4096 in STATE.md

"""Detect a generation that re-speaks the reference clip instead of the requested text."""

from __future__ import annotations

import difflib
import re
import unicodedata

CHARS_PER_SECOND = 14  # typical speech rate, as the Qwen engine assumes
# A chunk running longer than this is checked with speech-to-text...
SUSPECT_RATIO, SUSPECT_SLACK = 1.35, 0.8
# ...and is rejected on length alone, when there is no speech-to-text, beyond this.
SURE_RATIO, SURE_SLACK = 1.8, 1.5
MIN_LEAK_WORDS = 4  # words of the reference heard in a row that are not in the requested text


def words_of(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    return [w for w in re.sub(r"[^\w\s]+", " ", text).split() if w]


def expected_seconds(text: str) -> float:
    return len(text) / CHARS_PER_SECOND


def duration_suspect(text: str, seconds: float) -> bool:
    return seconds > expected_seconds(text) * SUSPECT_RATIO + SUSPECT_SLACK


def duration_certain(text: str, seconds: float) -> bool:
    return seconds > expected_seconds(text) * SURE_RATIO + SURE_SLACK


def leaks_reference(heard: str, reference: str, requested: str) -> bool:
    """True when ``heard`` repeats a run of the reference's words that ``requested`` does not contain."""
    heard_w, ref_w, req_w = words_of(heard), words_of(reference), words_of(requested)
    if len(heard_w) < MIN_LEAK_WORDS or len(ref_w) < MIN_LEAK_WORDS:
        return False
    req_runs = {tuple(req_w[i:i + MIN_LEAK_WORDS]) for i in range(len(req_w) - MIN_LEAK_WORDS + 1)}
    matcher = difflib.SequenceMatcher(None, heard_w, ref_w, autojunk=False)
    for a, _, size in matcher.get_matching_blocks():
        if size >= MIN_LEAK_WORDS:
            run = tuple(heard_w[a:a + MIN_LEAK_WORDS])
            if run not in req_runs:
                return True
    return False


def _plain(word: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", word) if not unicodedata.combining(c))


def leading_spill(heard: list[str], reference: str, requested: str, max_words: int = 12) -> int:
    """Number of words at the start of ``heard`` that are the last words of ``reference``.

    The model sometimes opens with the end of the reference clip (after the cut that removes
    it).  Returns 0 unless at least 2 words match and ``requested`` does not start with them.
    """
    heard_w = [_plain(w) for w in (w for t in heard for w in words_of(t))]
    ref_w = [_plain(w) for w in words_of(reference)][-max_words:]
    req_w = [_plain(w) for w in words_of(requested)]
    for k in range(min(len(ref_w), len(heard_w), max_words), 1, -1):
        if heard_w[:k] == ref_w[-k:] and req_w[:k] != heard_w[:k]:
            return k
    return 0

"""Text normalization and error rates for the ASR-based intelligibility check.

Whisper transcribes Quebec forms as standard French ("pis" → "puis") and writes
numbers as digits, which would inflate WER for exactly the model that gets the
accent right.  Both reference and hypothesis therefore go through the same
normalization: lowercase, no punctuation or diacritic-insensitive apostrophes,
digits spelled out, and a small Quebec→standard mapping.
"""

from __future__ import annotations

import re
import unicodedata

# Applied to both sides, so a form counts as correct whichever spelling the ASR picks.
QC_TO_STANDARD = {
    "pis": "puis",
    "tsé": "tu sais",
    "chu": "je suis",
    "faque": "ça fait que",
    "fait qu": "ça fait qu",
    "y'a": "il y a",
    "j'vas": "je vais",
    "icitte": "ici",
    "frette": "froid",
    "ben": "bien",
    "à soir": "ce soir",
    "à matin": "ce matin",
    "coudonc": "coup donc",
    "y fait": "il fait",
}
_MAPPING_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted(QC_TO_STANDARD, key=len, reverse=True))
                         + r")\b")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _spell_numbers(text: str) -> str:
    try:
        from num2words import num2words
    except ImportError:  # optional: without it digits simply stay digits
        return text

    def spell(match: re.Match[str]) -> str:
        value = match.group(0).replace(",", ".")
        return " " + num2words(float(value) if "." in value else int(value), lang="fr") + " "

    return _NUMBER_RE.sub(spell, text)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = _spell_numbers(text)
    text = _MAPPING_RE.sub(lambda m: QC_TO_STANDARD[m.group(1)], text)
    text = re.sub(r"[-'’]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h))
        previous = current
    return previous[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    return edit_distance(ref, hyp) / max(len(ref), 1)


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = list(normalize(reference).replace(" ", "")), list(normalize(hypothesis).replace(" ", ""))
    return edit_distance(ref, hyp) / max(len(ref), 1)

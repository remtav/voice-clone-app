"""Sentence-aligned prompt clip for engines that clone in context (Qwen3-TTS).

The model continues the reference as if the speaker kept talking.  Fed a long
clip, or one that stops mid-word or mid-sentence, it tends to re-speak the
reference.  So the model gets a shorter *prompt clip*: the start of the stored
voice, ended on a sentence boundary, with the matching part of its transcript.
The stored voice is never changed.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from app.stt import Transcriber, Word

log = logging.getLogger(__name__)

MIN_PROMPT_SECONDS = 3.0
SENTENCE_END = re.compile(r"[.!?…]+[\"'»”)\]]*$")
PAUSE_SECONDS = 0.25  # a gap this long between words counts as a boundary when no sentence ends in time
TAIL_SECONDS = 0.15  # silence kept after the last word


@dataclass
class PromptClip:
    path: Path
    text: str
    seconds: float
    trimmed: bool


def _norm(word: str) -> str:
    word = unicodedata.normalize("NFKC", word).lower()
    return re.sub(r"[^\w]+", "", word)


def pick_cut(words: list[Word], limit: float) -> int | None:
    """Index of the last word to keep: a sentence end before ``limit`` s, else a pause, else any word."""
    inside = [i for i, w in enumerate(words) if w.end <= limit]
    if not inside:
        return None
    ends = [i for i in inside if SENTENCE_END.search(words[i].text) and words[i].end >= MIN_PROMPT_SECONDS]
    if ends:
        return ends[-1]
    pauses = [i for i in inside if i + 1 < len(words) and words[i].end >= MIN_PROMPT_SECONDS
              and words[i + 1].start - words[i].end >= PAUSE_SECONDS]
    return pauses[-1] if pauses else inside[-1]


def cut_transcript(transcript: str, words: list[Word], cut: int) -> str | None:
    """The part of the (user-checked) ``transcript`` that matches ``words[: cut + 1]``, or None."""
    tokens = transcript.split()
    user = [(i, _norm(t)) for i, t in enumerate(tokens)]
    user = [(i, n) for i, n in user if n]
    heard = [_norm(w.text) for w in words]
    keep = [i for i, n in enumerate(heard) if n]
    if not user or not keep:
        return None
    matcher = difflib.SequenceMatcher(None, [heard[i] for i in keep], [n for _, n in user], autojunk=False)
    end = None  # exclusive index into ``user``
    for a, b, size in matcher.get_matching_blocks():
        if size == 0:
            continue
        last = [k for k in range(a, a + size) if keep[k] <= cut]
        if last:
            end = b + (last[-1] - a) + 1
    if end is None:
        return None
    return " ".join(tokens[: user[end - 1][0] + 1])


def _key(wav: Path, transcript: str, limit: int) -> str:
    raw = f"{wav.stat().st_mtime_ns}|{wav.stat().st_size}|{limit}|{transcript}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def prompt_paths(wav: Path) -> tuple[Path, Path]:
    return wav.with_suffix(".prompt.wav"), wav.with_suffix(".prompt.json")


def remove_prompt_clip(wav: Path) -> None:
    for path in prompt_paths(wav):
        path.unlink(missing_ok=True)


def prepare_prompt_clip(wav: Path, transcript: str | None, limit: int, transcriber: Transcriber | None,
                        language: str | None = None) -> PromptClip | None:
    """The clip and transcript to give the model; None when the stored voice should be used as it is.

    That is the case without a transcript (timbre-only mode), when the clip already fits in
    ``limit`` seconds, or when the cut cannot be placed (no speech-to-text, no usable timestamps).
    """
    text = " ".join((transcript or "").split())
    if not text or limit <= 0:
        return None
    info = sf.info(str(wav))
    if info.frames / info.samplerate <= limit:
        return None
    clip_path, meta_path = prompt_paths(wav)
    key = _key(wav, text, limit)
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("key") == key and clip_path.exists():
            return PromptClip(clip_path, meta["text"], meta["seconds"], True) if meta["text"] else None
    except (OSError, ValueError):
        pass

    result = None
    if transcriber is not None and transcriber.available:
        try:
            heard = transcriber.transcribe(wav, language, words=True)
        except Exception:  # noqa: BLE001 - fall back to the whole clip rather than fail the job
            log.exception("Could not time %s for a prompt clip", wav.name)
            return None
        cut = pick_cut(heard.words or [], limit)
        if cut is not None:
            words = heard.words
            nxt = words[cut + 1].start if cut + 1 < len(words) else float("inf")
            end = min(words[cut].end + TAIL_SECONDS, nxt - 0.05, limit)
            part = cut_transcript(text, words, cut)
            if part:
                result = (end, part)
    if result is None:
        log.warning("Voice %s: no sentence boundary found within %ds; using the whole clip", wav.name, limit)
        meta_path.write_text(json.dumps({"key": key, "text": ""}), encoding="utf-8")
        clip_path.unlink(missing_ok=True)
        return None
    end, part = result
    data, sr = sf.read(str(wav), dtype="float32", always_2d=False)
    sf.write(str(clip_path), data[: int(end * sr)], sr, subtype="PCM_16")
    meta_path.write_text(json.dumps({"key": key, "text": part, "seconds": end}), encoding="utf-8")
    log.info("Voice %s: prompt clip %.1fs of %.1fs, ends on %r", wav.name, end, info.frames / info.samplerate,
             part[-40:])
    return PromptClip(clip_path, part, end, True)

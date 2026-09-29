"""Cut long recordings of your own voice into transcribed 2-12 s training clips.

faster-whisper transcribes each recording with word timestamps; clips are cut
between words, preferably at pauses, and written in the same ``audio_data``
layout as the Quebec corpus (``client_id`` = ``--speaker``).

**Review the transcriptions before training.**  Whisper writes standard French
("puis", "je suis", "ça fait que") where you said "pis", "chu", "faque"; put back
what you actually said, since that spelling is what the model learns to pronounce.
Delete rows (and clips) with mistakes you do not want to fix.

    python -m scripts.finetune.segment_recording --out data/finetune/me/audio_data \\
        --speaker me recordings/*.m4a
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from scripts.finetune.prepare_qc_dataset import Clip, normalize_text, print_report, write_dataset

MIN_SECONDS = 2.0
MAX_SECONDS = 12.0
MIN_PAUSE = 0.3
PAD = 0.1


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def plan_segments(words: list[Word], min_seconds: float = MIN_SECONDS, max_seconds: float = MAX_SECONDS,
                  min_pause: float = MIN_PAUSE) -> list[Segment]:
    """Group words into clips: cut at a pause once long enough, never exceed max_seconds.

    Segments shorter than ``min_seconds`` are dropped (a lone "euh" is no training data).
    """
    segments: list[Segment] = []
    current: list[Word] = []

    def flush() -> None:
        if current and current[-1].end - current[0].start >= min_seconds:
            # faster-whisper words carry their own leading space ("-moi", "'il" have none).
            text = " ".join("".join(w.text for w in current).split())
            segments.append(Segment(current[0].start, current[-1].end, text))
        current.clear()

    for word in words:
        if current:
            gap = word.start - current[-1].end
            long_enough = current[-1].end - current[0].start >= min_seconds
            too_long = word.end - current[0].start > max_seconds
            if (gap >= min_pause and long_enough) or too_long:
                flush()
        current.append(word)
    flush()
    return segments


def padded_bounds(segments: list[Segment], total: float, pad: float = PAD) -> list[tuple[float, float]]:
    """Add a little silence around each clip without reaching into a neighbour."""
    bounds = []
    for i, seg in enumerate(segments):
        lo = max(0.0, seg.start - pad)
        hi = min(total, seg.end + pad)
        if i > 0:
            lo = max(lo, (segments[i - 1].end + seg.start) / 2)
        if i + 1 < len(segments):
            hi = min(hi, (seg.end + segments[i + 1].start) / 2)
        bounds.append((lo, hi))
    return bounds


def transcribe_words(wav: Path, model: str, device: str) -> list[Word]:
    from faster_whisper import WhisperModel

    try:
        import torch

        cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
    except ImportError:
        cuda = device == "cuda"
    whisper = WhisperModel(model, device="cuda" if cuda else "cpu", compute_type="float16" if cuda else "int8")
    segments, _ = whisper.transcribe(str(wav), language="fr", word_timestamps=True, vad_filter=True, beam_size=5)
    return [Word(w.start, w.end, w.word) for s in segments for w in (s.words or [])]


def cut_recording(recording: Path, work: Path, words: list[Word], speaker: str, name: str | None = None,
                  **limits) -> list[Clip]:
    """Write one wav per planned segment of ``recording`` (already 24 kHz mono) and return the clips."""
    name = name or recording.stem
    audio, sr = sf.read(str(recording), dtype="float32")
    segments = plan_segments(words, **limits)
    clips = []
    for i, (seg, (lo, hi)) in enumerate(zip(segments, padded_bounds(segments, len(audio) / sr), strict=True)):
        target = work / f"{speaker}_{name}_{i:04d}.wav"
        piece = audio[int(lo * sr):int(hi * sr)]
        sf.write(str(target), piece, sr, subtype="PCM_16")
        clips.append(Clip(target, normalize_text(seg.text), len(piece) / sr, speaker))
    return clips


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("recordings", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True, help="audio_data directory to create")
    parser.add_argument("--speaker", default="me")
    parser.add_argument("--asr-model", default="large-v3")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-seconds", type=float, default=MIN_SECONDS)
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    parser.add_argument("--holdout", type=int, default=5, help="clips kept aside as eval references")
    args = parser.parse_args(argv)

    from app.audio import TARGET_SR, convert_to_wav

    clips: list[Clip] = []
    with tempfile.TemporaryDirectory(prefix="segment-") as tmp:
        work = Path(tmp)
        for recording in args.recordings:
            wav = work / f"{recording.stem}.full.wav"
            convert_to_wav(recording, wav, TARGET_SR)
            words = transcribe_words(wav, args.asr_model, args.device)
            found = cut_recording(wav, work, words, args.speaker, name=recording.stem,
                                  min_seconds=args.min_seconds, max_seconds=args.max_seconds)
            print(f"{recording.name}: {len(words)} words -> {len(found)} clips")
            clips += found
        if not clips:
            print("error: no clip long enough was found", file=sys.stderr)
            return 1
        # Clips live in the temp dir: copy them into the dataset.
        # One speaker: hold out individual clips, not the whole speaker.
        summary = write_dataset(clips, args.out, args.holdout, link="copy", by_speaker=False)
    print_report(summary, {})
    print(f"\nNow review {args.out / 'metadata.csv'} (and holdout.csv): restore the Quebec forms you said "
          "(pis, chu, faque, tsé...) and delete clips with mistakes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

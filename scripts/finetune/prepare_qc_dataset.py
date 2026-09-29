"""Build a Quebec French training set in the layout the LoRA toolkit expects.

Output (``--out``)::

    audio_data/
    ├── metadata.csv   file_name,transcription,duration_seconds,client_id
    ├── holdout.csv    same columns; never trained on, used as eval references
    └── audio/*.wav

Sources:

* ``download`` fetches the prepared CC0 corpus (Common Voice fr clips tagged
  Québécois/Canadien, already 24 kHz mono and trimmed) from the Hugging Face
  repo ``tontate/f5-tts-quebec-french-finetune`` at a pinned revision.
* ``from-processed`` converts that corpus (option A of the plan).
* ``from-common-voice`` filters a full Common Voice fr release (option B): accent
  tag, votes, a per-speaker cap, deduplication, then ffmpeg conversion.  With
  ``--accent europe`` it builds the France/Belgium/Switzerland set that the
  phase 3 accent probe needs as negatives.

Examples::

    python -m scripts.finetune.prepare_qc_dataset download --dest data/finetune/qc_src
    python -m scripts.finetune.prepare_qc_dataset from-processed \\
        --src data/finetune/qc_src/dataset/processed --out data/finetune/qc/audio_data
    python -m scripts.finetune.prepare_qc_dataset from-common-voice \\
        --cv-dir ~/cv-corpus-fr --out data/finetune/qc/audio_data
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

QC_CORPUS_REPO = "tontate/f5-tts-quebec-french-finetune"
QC_CORPUS_REVISION = "c59b5de7008a03336c19caa75a126fb902c327f5"
TARGET_SR = 24000
MIN_SECONDS = 1.0
MAX_SECONDS = 15.0
MIN_TEXT_CHARS = 3
HOLDOUT_CLIPS = 40
FIELDS = ["file_name", "transcription", "duration_seconds", "client_id"]

# Common Voice free-text `accents` values for Quebec / Canadian French.
ACCENT_RE = re.compile(r"quebecois|quebec|canadien|canadian|canada|\bqc\b|montreal")
EUROPE_RE = re.compile(r"\bfrance\b|belgique|belgian|belgium|suisse|swiss|switzerland")

# Typography only.  Quebec spellings and elisions ("pis", "tsé", "icitte") are
# kept verbatim: they are what the model must learn to pronounce.
_TYPOGRAPHY = str.maketrans({
    "’": "'", "‘": "'", "ʼ": "'",
    "“": '"', "”": '"', "«": '"', "»": '"',
    "…": "...", " ": " ", " ": " ", " ": " ",
    "–": "-", "—": "-",
})


@dataclass
class Clip:
    source: Path
    text: str
    duration: float
    client_id: str = ""

    @property
    def name(self) -> str:
        return self.source.stem + ".wav"


def fold(text: str) -> str:
    """Lowercase and strip diacritics (for matching, never for training text)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def matches_qc_accent(accents: str) -> bool:
    return bool(accents) and bool(ACCENT_RE.search(fold(accents)))


def matches_europe_accent(accents: str) -> bool:
    """France/Belgium/Switzerland, and never a speaker who also tags Quebec."""
    return bool(accents) and bool(EUROPE_RE.search(fold(accents))) and not matches_qc_accent(accents)


ACCENT_MATCHERS = {"qc": matches_qc_accent, "europe": matches_europe_accent}


def normalize_text(text: str) -> str:
    text = text.translate(_TYPOGRAPHY)
    text = re.sub(r"\s+([,.])", r"\1", text)  # French keeps the space before ; : ! ?
    return re.sub(r"\s+", " ", text).strip()


def dedupe_key(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", fold(text))).strip()


def _stable_rank(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def split_holdout(
    clips: list[Clip], n: int = HOLDOUT_CLIPS, by_speaker: bool = True
) -> tuple[list[Clip], list[Clip]]:
    """Deterministically hold out ~n clips.

    With ``by_speaker`` and known speakers, whole speakers are held out so
    evaluation voices are never seen in training; otherwise individual clips are
    (e.g. a single speaker's own recordings).
    """
    if n <= 0 or not clips:
        return list(clips), []

    def group(clip: Clip) -> str:
        return clip.client_id if by_speaker and clip.client_id else f"clip:{clip.name}"

    groups: dict[str, list[Clip]] = {}
    for clip in clips:
        groups.setdefault(group(clip), []).append(clip)
    held: set[str] = set()
    count = 0
    for key in sorted(groups, key=_stable_rank):
        if count >= n:
            break
        if len(groups) - len(held) <= 1:
            break  # always keep at least one group for training
        held.add(key)
        count += len(groups[key])
    train = [c for c in clips if group(c) not in held]
    holdout = [c for c in clips if group(c) in held]
    return train, holdout


def keep_clip(text: str, duration: float, min_seconds: float, max_seconds: float) -> bool:
    return len(text) >= MIN_TEXT_CHARS and min_seconds <= duration <= max_seconds


# --------------------------------------------------------------------------- sources
def read_processed(
    src: Path, min_seconds: float = MIN_SECONDS, max_seconds: float = MAX_SECONDS
) -> tuple[list[Clip], Counter]:
    """Read ``metadata.csv`` (``audio_file|text``) from the prepared QC corpus.

    Audio paths in that CSV are absolute paths from the machine that built it;
    only the file name is used, looked up under ``wavs/qc/`` then ``wavs/``.
    """
    stats: Counter = Counter()
    clips: list[Clip] = []
    seen: set[str] = set()
    with (src / "metadata.csv").open(encoding="utf-8") as fh:
        header = fh.readline()
        if not header.startswith("audio_file|"):
            raise ValueError(f"Unexpected header in {src / 'metadata.csv'}: {header.strip()!r}")
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            path, _, text = line.partition("|")
            name = Path(path).name
            wav = next((p for p in (src / "wavs" / "qc" / name, src / "wavs" / name) if p.is_file()), None)
            if wav is None:
                stats["missing_audio"] += 1
                continue
            text = normalize_text(text)
            duration = sf.info(str(wav)).duration
            if not keep_clip(text, duration, min_seconds, max_seconds):
                stats["out_of_bounds"] += 1
                continue
            key = dedupe_key(text)
            if key in seen:
                stats["duplicate_text"] += 1
                continue
            seen.add(key)
            clips.append(Clip(wav, text, duration))
    return clips, stats


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))


def select_cv_rows(
    rows: Iterable[dict[str, str]],
    min_up_votes: int = 2,
    max_down_votes: int = 0,
    max_per_speaker: int = 300,
    accent: str = "qc",
) -> tuple[list[dict[str, str]], Counter]:
    """Filter Common Voice rows: accent tag, votes, dedupe, per-speaker cap."""
    matches = ACCENT_MATCHERS[accent]
    stats: Counter = Counter()
    per_speaker: Counter = Counter()
    seen: set[str] = set()
    selected = []
    for row in rows:
        if not matches(row.get("accents", "")):
            stats[f"not_{accent}"] += 1
            continue
        if int(row.get("up_votes") or 0) < min_up_votes or int(row.get("down_votes") or 0) > max_down_votes:
            stats["votes"] += 1
            continue
        text = normalize_text(row.get("sentence", ""))
        key = dedupe_key(text)
        if len(text) < MIN_TEXT_CHARS or key in seen:
            stats["duplicate_or_empty_text"] += 1
            continue
        speaker = row.get("client_id", "")
        if per_speaker[speaker] >= max_per_speaker:
            stats["speaker_cap"] += 1
            continue
        seen.add(key)
        per_speaker[speaker] += 1
        selected.append({**row, "sentence": text})
    return selected, stats


def convert_clip(src: Path, dst: Path) -> float:
    """Decode with ffmpeg to 24 kHz mono 16-bit WAV, trimming leading/trailing silence."""
    trim = "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.1"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
        "-af", f"{trim},areverse,{trim},areverse",
        "-ac", "1", "-ar", str(TARGET_SR), "-acodec", "pcm_s16le", str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[:300] or "ffmpeg failed")
    return sf.info(str(dst)).duration


def read_common_voice(
    cv_dir: Path,
    work_dir: Path,
    min_seconds: float = MIN_SECONDS,
    max_seconds: float = MAX_SECONDS,
    **filters,
) -> tuple[list[Clip], Counter]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to convert Common Voice clips")
    tsv = cv_dir / "validated.tsv"
    if tsv.exists():
        rows = _read_tsv(tsv)
    else:
        rows = [r for name in ("train.tsv", "dev.tsv", "test.tsv") if (cv_dir / name).exists()
                for r in _read_tsv(cv_dir / name)]
    selected, stats = select_cv_rows(rows, **filters)
    work_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for row in selected:
        src = cv_dir / "clips" / row["path"]
        if not src.is_file():
            stats["missing_audio"] += 1
            continue
        dst = work_dir / (Path(row["path"]).stem + ".wav")
        try:
            duration = convert_clip(src, dst)
        except RuntimeError as exc:
            stats["decode_error"] += 1
            print(f"skip {src.name}: {exc}", file=sys.stderr)
            continue
        if not keep_clip(row["sentence"], duration, min_seconds, max_seconds):
            stats["out_of_bounds"] += 1
            dst.unlink(missing_ok=True)
            continue
        clips.append(Clip(dst, row["sentence"], duration, row.get("client_id", "")))
    return clips, stats


# --------------------------------------------------------------------------- output
def place(src: Path, dst: Path, mode: str) -> None:
    dst.unlink(missing_ok=True)
    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # other filesystem: fall back to a copy
    shutil.copy2(src, dst)


def _write_csv(path: Path, clips: list[Clip]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(FIELDS)
        for clip in clips:
            writer.writerow([f"audio/{clip.name}", clip.text, f"{clip.duration:.3f}", clip.client_id])


def write_dataset(
    clips: list[Clip], out: Path, holdout_clips: int = HOLDOUT_CLIPS, link: str = "hardlink", by_speaker: bool = True
) -> dict:
    names = Counter(c.name for c in clips)
    clashes = [n for n, k in names.items() if k > 1]
    if clashes:
        raise ValueError(f"Duplicate audio file names: {clashes[:5]}")
    train, holdout = split_holdout(clips, holdout_clips, by_speaker)
    audio = out / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        place(clip.source, audio / clip.name, link)
    _write_csv(out / "metadata.csv", train)
    _write_csv(out / "holdout.csv", holdout)
    return summarize(train, holdout)


def summarize(train: list[Clip], holdout: list[Clip]) -> dict:
    edges = [1, 2, 4, 6, 8, 10, 12, 15]
    histogram = {f"{lo}-{hi}s": sum(lo <= c.duration < hi or (hi == edges[-1] and c.duration == hi) for c in train)
                 for lo, hi in zip(edges, edges[1:], strict=False)}
    return {
        "train_clips": len(train),
        "train_hours": round(sum(c.duration for c in train) / 3600, 2),
        "holdout_clips": len(holdout),
        "speakers": len({c.client_id for c in train if c.client_id}) or None,
        "histogram": histogram,
    }


def print_report(summary: dict, stats: Counter) -> None:
    print(f"train: {summary['train_clips']} clips, {summary['train_hours']} h"
          + (f", {summary['speakers']} speakers" if summary["speakers"] else ""))
    print(f"holdout: {summary['holdout_clips']} clips")
    for bucket, count in summary["histogram"].items():
        print(f"  {bucket:>7}: {count}")
    if stats:
        print("skipped: " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))


# --------------------------------------------------------------------------- CLI
def download(dest: Path, revision: str = QC_CORPUS_REVISION) -> Path:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=QC_CORPUS_REPO,
        revision=revision,
        allow_patterns=["dataset/processed/metadata.csv", "dataset/processed/wavs/*"],
        local_dir=str(dest),
    )
    return Path(path) / "dataset" / "processed"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help=f"fetch the prepared corpus from {QC_CORPUS_REPO}")
    dl.add_argument("--dest", type=Path, default=Path("data/finetune/qc_src"))
    dl.add_argument("--revision", default=QC_CORPUS_REVISION)

    for name, help_text in (("from-processed", "option A: prepared corpus"),
                            ("from-common-voice", "option B: full Common Voice fr release")):
        p = sub.add_parser(name, help=help_text)
        if name == "from-processed":
            p.add_argument("--src", type=Path, required=True, help="dir with metadata.csv and wavs/")
        else:
            p.add_argument("--cv-dir", type=Path, required=True, help="dir with validated.tsv and clips/")
            p.add_argument("--min-up-votes", type=int, default=2)
            p.add_argument("--max-down-votes", type=int, default=0)
            p.add_argument("--max-per-speaker", type=int, default=300)
            p.add_argument("--accent", choices=sorted(ACCENT_MATCHERS), default="qc",
                           help="europe: France/Belgium/Switzerland clips (accent probe negatives)")
            p.add_argument("--work-dir", type=Path, help="where converted wavs go (default: a temp dir)")
        p.add_argument("--out", type=Path, required=True, help="audio_data directory to create")
        p.add_argument("--holdout", type=int, default=HOLDOUT_CLIPS)
        p.add_argument("--min-seconds", type=float, default=MIN_SECONDS)
        p.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
        p.add_argument("--link", choices=["hardlink", "symlink", "copy"], default="hardlink")

    args = parser.parse_args(argv)
    if args.command == "download":
        print(download(args.dest, args.revision))
        return 0

    if args.command == "from-processed":
        clips, stats = read_processed(args.src, args.min_seconds, args.max_seconds)
        summary = write_dataset(clips, args.out, args.holdout, args.link)
    else:
        with tempfile.TemporaryDirectory(prefix="cv-qc-") as tmp:
            work = args.work_dir or Path(tmp)
            clips, stats = read_common_voice(
                args.cv_dir, work, args.min_seconds, args.max_seconds,
                min_up_votes=args.min_up_votes, max_down_votes=args.max_down_votes,
                max_per_speaker=args.max_per_speaker, accent=args.accent,
            )
            # Converted files may live in a temp dir: always copy them out.
            summary = write_dataset(clips, args.out, args.holdout, "copy" if args.work_dir is None else args.link)
    print_report(summary, stats)
    if not summary["train_clips"]:
        print("error: no usable clips", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

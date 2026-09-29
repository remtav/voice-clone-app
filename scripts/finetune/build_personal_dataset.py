"""Mix your own clips with part of the Quebec corpus for the second fine-tuning stage.

Your clips are repeated ``--oversample`` times so they dominate, and a random
``--qc-fraction`` of the Quebec training set is kept so the model does not
forget the accent it learned in stage 1.  Only training rows are mixed; your
holdout clips are copied as ``holdout.csv`` for evaluation.

    python -m scripts.finetune.build_personal_dataset --own data/finetune/me/audio_data \\
        --qc data/finetune/qc/audio_data --out data/finetune/me_mix/audio_data
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

from scripts.finetune.prepare_qc_dataset import FIELDS, place


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def mix_rows(own: list[dict[str, str]], qc: list[dict[str, str]], oversample: int, qc_fraction: float,
             seed: int) -> list[dict[str, str]]:
    """Return rows (file_name prefixed by source) in a deterministic shuffled order."""
    if oversample < 1:
        raise ValueError("oversample must be >= 1")
    if not 0 <= qc_fraction <= 1:
        raise ValueError("qc_fraction must be between 0 and 1")
    rng = random.Random(seed)
    kept_qc = rng.sample(qc, round(len(qc) * qc_fraction))

    def relabel(row: dict[str, str], prefix: str) -> dict[str, str]:
        return {**row, "file_name": f"audio/{prefix}_{Path(row['file_name']).name}"}

    rows = [relabel(r, "own") for r in own] * oversample + [relabel(r, "qc") for r in kept_qc]
    rng.shuffle(rows)
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--own", type=Path, required=True, help="your audio_data (segment_recording output)")
    parser.add_argument("--qc", type=Path, required=True, help="Quebec corpus audio_data (phase 1)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--oversample", type=int, default=3)
    parser.add_argument("--qc-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--link", choices=["hardlink", "symlink", "copy"], default="hardlink")
    args = parser.parse_args(argv)

    own, qc = read_rows(args.own / "metadata.csv"), read_rows(args.qc / "metadata.csv")
    if not own:
        print(f"error: no clips in {args.own / 'metadata.csv'}", file=sys.stderr)
        return 1
    rows = mix_rows(own, qc, args.oversample, args.qc_fraction, args.seed)
    own_holdout = read_rows(args.own / "holdout.csv")
    holdout = [{**r, "file_name": f"audio/own_{Path(r['file_name']).name}"} for r in own_holdout]

    audio = args.out / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    sources = [(args.own, r, "own") for r in own + own_holdout]
    sources += [(args.qc, r, "qc") for r in qc]
    wanted = {r["file_name"] for r in rows + holdout}
    for root, row, prefix in sources:
        name = f"audio/{prefix}_{Path(row['file_name']).name}"
        if name in wanted and not (args.out / name).exists():
            place(root / row["file_name"], args.out / name, args.link)
    write_rows(args.out / "metadata.csv", rows)
    write_rows(args.out / "holdout.csv", holdout)

    own_rows = sum(r["file_name"].startswith("audio/own_") for r in rows)
    print(f"{len(rows)} training rows: {own_rows} of yours ({len(own)} clips x{args.oversample}), "
          f"{len(rows) - own_rows} Quebec corpus; holdout: {len(holdout)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Synthesize the evaluation battery: checkpoints × CFG weights × voices × sentences.

Each checkpoint is loaded once through the app's own engine (so the eval runs
exactly what production runs), then every (cfg, voice, sentence) cell is
generated with a fixed seed.  ``manifest.csv`` lists every clip for scoring.

Example (the plan's grid: 2 checkpoints × 2 CFG × 3 voices × 50 sentences = 600 clips)::

    python -m scripts.finetune.eval.battery --out data/finetune/eval/fr_ca_r16 \\
        --checkpoint base=v3 --checkpoint fr_ca=data/models/t3_fr_ca.safetensors \\
        --voice me=data/voices/me.wav --voice qc1=<holdout clip> --voice qc2=<holdout clip>
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

SENTENCES = Path(__file__).with_name("sentences_qc.tsv")
MANIFEST_FIELDS = ["checkpoint", "cfg_weight", "voice", "reference", "sentence_id", "category", "text", "wav",
                   "seconds"]


@dataclass(frozen=True)
class Cell:
    checkpoint: str
    cfg_weight: float
    voice: str
    reference: str
    sentence_id: int
    category: str
    text: str

    @property
    def wav(self) -> str:
        return f"{self.checkpoint}/cfg{self.cfg_weight:.2f}/{self.voice}/{self.sentence_id:02d}.wav"


def load_sentences(path: Path = SENTENCES, limit: int | None = None) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        rows = [(r["category"], r["sentence"]) for r in csv.DictReader(fh, delimiter="\t")]
    return rows[:limit] if limit else rows


def parse_pairs(values: list[str], what: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for value in values:
        name, sep, spec = value.partition("=")
        if not sep or not name or not spec:
            raise ValueError(f"--{what} expects name=value, got {value!r}")
        if name in pairs:
            raise ValueError(f"duplicate --{what} name {name!r}")
        pairs[name] = spec
    return pairs


def build_grid(checkpoints: dict[str, str], cfgs: list[float], voices: dict[str, str],
               sentences: list[tuple[str, str]]) -> list[Cell]:
    return [
        Cell(ckpt, cfg, voice, voices[voice], i, category, text)
        for ckpt, cfg, voice, (i, (category, text)) in product(checkpoints, cfgs, voices, enumerate(sentences))
    ]


def make_engine(spec: str, device: str):
    from app.engines.chatterbox import ChatterboxEngine

    if spec in ("v2", "v3"):
        return ChatterboxEngine(t3_model=spec, device=device)
    path = Path(spec)
    return ChatterboxEngine(t3_model=path.name, t3_path=path, device=device)


def generate(cells: list[Cell], checkpoints: dict[str, str], out: Path, device: str, exaggeration: float,
             temperature: float, seed: int, overwrite: bool = False) -> list[dict]:
    import soundfile as sf

    from app.engines.base import SynthesisParams

    rows = []
    for name, spec in checkpoints.items():
        engine = None
        for cell in (c for c in cells if c.checkpoint == name):
            target = out / cell.wav
            if target.exists() and not overwrite:
                seconds = sf.info(str(target)).duration  # resume an interrupted run
            else:
                if engine is None:
                    engine = make_engine(spec, device)
                    engine.load()
                params = SynthesisParams(language="fr", exaggeration=exaggeration, cfg_weight=cell.cfg_weight,
                                         temperature=temperature, seed=seed + cell.sentence_id)
                audio = engine.synthesize(cell.text, Path(cell.reference), params)
                target.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(target), audio, engine.sample_rate)
                seconds = len(audio) / engine.sample_rate
            rows.append({**asdict(cell), "wav": cell.wav, "seconds": round(seconds, 3)})
            print(f"{cell.wav}  {seconds:.1f}s", flush=True)
        del engine  # free VRAM before the next checkpoint
    return rows


def write_manifest(rows: list[dict], out: Path) -> Path:
    path = out / "manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="name=v3|path.safetensors; the first one is the baseline")
    parser.add_argument("--voice", action="append", required=True, help="name=reference.wav")
    parser.add_argument("--cfg", type=float, nargs="+", default=[0.5, 0.8])
    parser.add_argument("--exaggeration", type=float, default=0.4)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sentences", type=Path, default=SENTENCES)
    parser.add_argument("--limit", type=int, help="only the first N sentences (quick runs)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true", help="regenerate clips that already exist")
    args = parser.parse_args(argv)

    try:
        checkpoints = parse_pairs(args.checkpoint, "checkpoint")
        voices = parse_pairs(args.voice, "voice")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for name, ref in voices.items():
        if not Path(ref).is_file():
            print(f"error: reference for voice {name!r} not found: {ref}", file=sys.stderr)
            return 1
    voices = {name: str(Path(ref).absolute()) for name, ref in voices.items()}
    cells = build_grid(checkpoints, args.cfg, voices, load_sentences(args.sentences, args.limit))
    args.out.mkdir(parents=True, exist_ok=True)
    rows = generate(cells, checkpoints, args.out, args.device, args.exaggeration, args.temperature, args.seed,
                    args.overwrite)
    print(f"{len(rows)} clips -> {write_manifest(rows, args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

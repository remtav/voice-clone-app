"""Check a fine-tuned T3 checkpoint before it goes anywhere near the app.

Checks, against an official T3 (v3 by default):

1. same tensor names, shapes and dtypes (float32), no leftover ``lora_*`` keys;
2. the tensors that differ are all LoRA targets (q/k/v/o, gate/up/down
   projection weights), and at least one differs (else the LoRA did nothing);
3. ``--strict-load``: ``T3(T3Config.multilingual()).load_state_dict(strict=True)``;
4. ``--smoke-reference ref.wav``: loads it through the app's engine,
   synthesizes a few Quebec French sentences into ``--smoke-out`` and checks
   their speaking rate.  A damaged model keeps talking until the token limit
   (seen on CPU: 40 s for a 7-word sentence after an over-aggressive run),
   which reads as a rate far below natural speech.

Exit code 0 only if every requested check passes.  Example::

    python -m scripts.finetune.validate_t3_checkpoint \\
        data/finetune/runs/fr_ca_r16/merged_model/t3_fr_ca.safetensors --strict-load
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scripts.finetune.t3_state import PROJECTION_RE, LazyState, resolve_t3

SMOKE_SENTENCES = [
    "Tu viens-tu souper icitte à soir ?",
    "Il faut que je stationne mon char devant le dépanneur.",
    "Fait qu'on a fini par jaser toute la soirée sur la galerie.",
]


# Natural French runs at ~12-16 characters per second; the official v3 stays above 5.
MIN_CHARS_PER_SECOND = 3.0
MAX_CHARS_PER_SECOND = 40.0


def speaking_rate_problem(text: str, seconds: float) -> str | None:
    rate = len(text) / seconds if seconds > 0 else float("inf")
    if rate < MIN_CHARS_PER_SECOND:
        return f"{seconds:.1f}s for {len(text)} chars ({rate:.1f} chars/s): runaway generation, no end of speech"
    if rate > MAX_CHARS_PER_SECOND:
        return f"{seconds:.1f}s for {len(text)} chars ({rate:.1f} chars/s): truncated output"
    return None


@dataclass
class Report:
    tensors: int = 0
    changed: list[str] = field(default_factory=list)
    max_abs_delta: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def compare_state(
    base: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    allowed: re.Pattern[str] = PROJECTION_RE,
    require_change: bool = True,
) -> Report:
    report = Report(tensors=len(candidate))
    base_keys, cand_keys = set(base), set(candidate)
    if missing := sorted(base_keys - cand_keys):
        report.errors.append(f"{len(missing)} tensor(s) missing, e.g. {missing[:3]}")
    if extra := sorted(cand_keys - base_keys):
        report.errors.append(f"{len(extra)} unexpected tensor(s), e.g. {extra[:3]}")
    if lora := sorted(k for k in cand_keys if "lora" in k.lower()):
        report.errors.append(f"LoRA tensors left in the checkpoint (merge them first): {lora[:3]}")
    unexpected = []
    for key in sorted(base_keys & cand_keys):
        b, c = base[key], candidate[key]
        if b.shape != c.shape:
            report.errors.append(f"{key}: shape {c.shape} != base {b.shape}")
            continue
        if c.dtype != np.float32 or c.dtype != b.dtype:
            report.errors.append(f"{key}: dtype {c.dtype} (base {b.dtype}, expected float32)")
            continue
        if np.array_equal(b, c):
            continue
        report.changed.append(key)
        report.max_abs_delta = max(report.max_abs_delta, float(np.max(np.abs(c - b))))
        if not allowed.match(key):
            unexpected.append(key)
    if unexpected:
        report.errors.append(f"{len(unexpected)} non-LoRA tensor(s) changed, e.g. {unexpected[:3]}")
    if require_change and not report.changed:
        report.errors.append("no tensor differs from the base: the LoRA had no effect")
    return report


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_load(path: Path) -> None:
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config
    from safetensors.torch import load_file

    T3(T3Config.multilingual()).load_state_dict(load_file(str(path)), strict=True)


def smoke(path: Path, reference: Path, out_dir: Path, device: str) -> list[tuple[Path, str, float]]:
    import soundfile as sf

    from app.engines.base import SynthesisParams
    from app.engines.chatterbox import ChatterboxEngine

    engine = ChatterboxEngine(t3_model=path.name, t3_path=path, device=device)
    engine.load()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, sentence in enumerate(SMOKE_SENTENCES):
        params = SynthesisParams(language="fr", exaggeration=0.4, cfg_weight=0.8, temperature=0.6, seed=i)
        audio = engine.synthesize(sentence, reference, params)
        target = out_dir / f"smoke_{i:02d}.wav"
        sf.write(str(target), audio, engine.sample_rate)
        written.append((target, sentence, len(audio) / engine.sample_rate))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--base", default="v3", help="official v3/v2 or a .safetensors to compare against")
    parser.add_argument("--allow-unchanged", action="store_true", help="do not fail when nothing changed")
    parser.add_argument("--strict-load", action="store_true", help="also load into a T3 module (needs chatterbox)")
    parser.add_argument("--smoke-reference", type=Path, help="reference wav: synthesize test sentences")
    parser.add_argument("--smoke-out", type=Path, help="where smoke wavs go (default: next to the checkpoint)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        print(f"error: {args.checkpoint} not found", file=sys.stderr)
        return 1
    report = compare_state(LazyState(resolve_t3(args.base)), LazyState(args.checkpoint),
                           require_change=not args.allow_unchanged)
    size_gb = args.checkpoint.stat().st_size / 1e9
    print(f"{args.checkpoint.name}: {report.tensors} tensors, {size_gb:.2f} GB, sha256 {sha256_file(args.checkpoint)}")
    print(f"changed vs {args.base}: {len(report.changed)} tensor(s), max |delta| {report.max_abs_delta:.3g}")

    if report.ok and args.strict_load:
        try:
            strict_load(args.checkpoint)
            print("strict load into T3(T3Config.multilingual()): OK")
        except Exception as exc:  # noqa: BLE001 - report any loader failure
            report.errors.append(f"strict load failed: {exc}")
    if report.ok and args.smoke_reference:
        out_dir = args.smoke_out or args.checkpoint.parent / "smoke"
        for wav, sentence, seconds in smoke(args.checkpoint, args.smoke_reference, out_dir, args.device):
            print(f"smoke: {wav} ({seconds:.1f}s)")
            if problem := speaking_rate_problem(sentence, seconds):
                report.errors.append(f"smoke {wav.name}: {problem}")

    for error in report.errors:
        print(f"FAIL: {error}", file=sys.stderr)
    print("OK" if report.ok else "FAILED")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())

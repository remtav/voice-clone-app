"""Score every clip of a battery: WER/CER (faster-whisper), speaker similarity, P(Quebec).

Writes ``scores.csv`` next to ``manifest.csv``; rerunning skips clips already scored.

* WER/CER: ``faster-whisper`` (``large-v3`` by default) with ``language="fr"``,
  normalized by :mod:`scripts.finetune.eval.text`.
* Speaker similarity: cosine between the clip and its reference voice, with an
  independent speaker encoder (SpeechBrain ECAPA) rather than the voice encoder
  Chatterbox itself conditions on.
* P(Quebec): the probe from :mod:`scripts.finetune.eval.accent_probe`, if given.

Example::

    python -m scripts.finetune.eval.score --battery data/finetune/eval/fr_ca_r16 \\
        --probe data/finetune/eval/accent_probe.npz
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from scripts.finetune.eval.battery import MANIFEST_FIELDS
from scripts.finetune.eval.text import cer, wer

SCORE_FIELDS = MANIFEST_FIELDS + ["hypothesis", "wer", "cer", "speaker_sim", "p_qc"]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.ravel(a), np.ravel(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class Transcriber:
    def __init__(self, model: str, device: str) -> None:
        from faster_whisper import WhisperModel

        cuda = device == "cuda" or (device == "auto" and _cuda())
        self.model = WhisperModel(model, device="cuda" if cuda else "cpu", compute_type="float16" if cuda else "int8")

    def __call__(self, wav: Path) -> str:
        segments, _ = self.model.transcribe(str(wav), language="fr", beam_size=5, vad_filter=False)
        return " ".join(s.text.strip() for s in segments)


class SpeakerEncoder:
    def __init__(self, device: str) -> None:
        from speechbrain.inference.speaker import EncoderClassifier

        run_device = "cuda" if device == "cuda" or (device == "auto" and _cuda()) else "cpu"
        self.model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                    run_opts={"device": run_device})
        self._cache: dict[str, np.ndarray] = {}

    def __call__(self, wav: Path) -> np.ndarray:
        key = str(wav)
        if key not in self._cache:
            import librosa
            import torch

            audio, _ = librosa.load(key, sr=16000, mono=True)
            with torch.inference_mode():
                self._cache[key] = self.model.encode_batch(torch.from_numpy(audio)[None]).squeeze().cpu().numpy()
        return self._cache[key]


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _missing(row: dict, column: str) -> bool:
    return str(row.get(column, "")).strip() == ""


def score_rows(rows, battery: Path, transcribe=None, speaker=None, probe=None, embed=None, done=None):
    """Yield scored rows; each scorer is optional (None leaves its columns as they were).

    Rows found in ``done`` (an earlier scores.csv) keep their values and only get the
    columns still missing, so a later run can add P(Quebec) without redoing the ASR.
    """
    done = done or {}
    for row in rows:
        out = {**row, **done.get(row["wav"], {})}
        wav = battery / row["wav"]
        if transcribe is not None and _missing(out, "wer"):
            out["hypothesis"] = transcribe(wav)
            out["wer"] = round(wer(row["text"], out["hypothesis"]), 4)
            out["cer"] = round(cer(row["text"], out["hypothesis"]), 4)
        if speaker is not None and _missing(out, "speaker_sim"):
            out["speaker_sim"] = round(cosine(speaker(wav), speaker(Path(row["reference"]))), 4)
        if probe is not None and embed is not None and _missing(out, "p_qc"):
            out["p_qc"] = round(float(probe.predict_proba(embed(wav))[0]), 4)
        yield out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--battery", type=Path, required=True, help="dir with manifest.csv")
    parser.add_argument("--probe", type=Path, help="accent_probe.npz (omit to skip P(Quebec))")
    parser.add_argument("--asr-model", default="large-v3")
    parser.add_argument("--no-asr", action="store_true")
    parser.add_argument("--no-speaker", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    manifest = args.battery / "manifest.csv"
    if not manifest.is_file():
        print(f"error: {manifest} not found (run eval.battery first)", file=sys.stderr)
        return 1
    out_path = args.battery / "scores.csv"
    done = {r["wav"]: r for r in read_csv(out_path)} if out_path.is_file() else {}

    transcribe = None if args.no_asr else Transcriber(args.asr_model, args.device)
    speaker = None if args.no_speaker else SpeakerEncoder(args.device)
    probe = embed = None
    if args.probe:
        from scripts.finetune.eval.accent_probe import Embedder, Probe

        probe = Probe.load(args.probe)
        embed = Embedder(probe.model, probe.layer, args.device)

    tmp = out_path.with_suffix(".partial")
    count = 0
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SCORE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in score_rows(read_csv(manifest), args.battery, transcribe, speaker, probe, embed, done):
            writer.writerow(row)
            fh.flush()
            count += 1
            print(f"{row['wav']}  wer={row.get('wer', '')} sim={row.get('speaker_sim', '')} "
                  f"p_qc={row.get('p_qc', '')}", flush=True)
    tmp.replace(out_path)
    print(f"{count} clips scored -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

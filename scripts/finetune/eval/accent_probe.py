"""Accent probe: P(Quebec) for a clip, from mid-layer XLS-R embeddings.

A logistic regression separates real Quebec clips from real European French
clips (France / Belgium / Switzerland, e.g. from
``prepare_qc_dataset from-common-voice --accent europe``).  It must reach at
least ``--min-accuracy`` on held-out clips before being trusted; then the
mean P(Quebec) of synthesized clips measures how Quebec the model sounds.

Train::

    python -m scripts.finetune.eval.accent_probe train --out data/finetune/eval/accent_probe.npz \\
        --positive data/finetune/qc/audio_data --negative data/finetune/eu/audio_data

The probe is stored as plain arrays (.npz), not a pickle.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "facebook/wav2vec2-xls-r-300m"
DEFAULT_LAYER = 12
SAMPLE_RATE = 16000


@dataclass
class Probe:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    model: str = DEFAULT_MODEL
    layer: int = DEFAULT_LAYER
    heldout_accuracy: float = float("nan")

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        z = ((np.atleast_2d(features) - self.mean) / self.scale) @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, scale=self.scale, coef=self.coef, intercept=self.intercept,
                 model=self.model, layer=self.layer, heldout_accuracy=self.heldout_accuracy)

    @classmethod
    def load(cls, path: Path) -> Probe:
        data = np.load(path, allow_pickle=False)
        return cls(mean=data["mean"], scale=data["scale"], coef=data["coef"], intercept=float(data["intercept"]),
                   model=str(data["model"]), layer=int(data["layer"]),
                   heldout_accuracy=float(data["heldout_accuracy"]))


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0, iterations: int = 500) -> Probe:
    """L2-regularized logistic regression on standardized features (Newton steps)."""
    mean = x.mean(axis=0)
    scale = x.std(axis=0) + 1e-6
    xs = np.hstack([(x - mean) / scale, np.ones((len(x), 1))])
    w = np.zeros(xs.shape[1])
    reg = np.full(xs.shape[1], l2)
    reg[-1] = 0.0  # no penalty on the intercept
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-np.clip(xs @ w, -50, 50)))
        grad = xs.T @ (p - y) + reg * w
        hessian = (xs * (p * (1 - p))[:, None]).T @ xs + np.diag(reg + 1e-9)
        step = np.linalg.solve(hessian, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return Probe(mean=mean, scale=scale, coef=w[:-1], intercept=float(w[-1]))


def split_indices(n: int, heldout: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(n)
    cut = max(1, int(round(n * heldout)))
    return order[cut:], order[:cut]


def train_probe(pos: np.ndarray, neg: np.ndarray, heldout: float = 0.2, seed: int = 0, l2: float = 1.0) -> Probe:
    x = np.vstack([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    train_idx, test_idx = split_indices(len(x), heldout, seed)
    probe = fit_logistic(x[train_idx], y[train_idx], l2=l2)
    probe.heldout_accuracy = float(np.mean((probe.predict_proba(x[test_idx]) >= 0.5) == y[test_idx]))
    final = fit_logistic(x, y, l2=l2)  # use every clip once the accuracy is known
    final.heldout_accuracy = probe.heldout_accuracy
    return final


class Embedder:
    """Mean-pooled hidden states of one XLS-R layer."""

    def __init__(self, model: str = DEFAULT_MODEL, layer: int = DEFAULT_LAYER, device: str = "auto") -> None:
        import torch
        from transformers import AutoFeatureExtractor, AutoModel

        self.torch = torch
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto"
                                                                                     else device)
        self.extractor = AutoFeatureExtractor.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(self.device).eval()
        self.name, self.layer = model, layer

    def __call__(self, wav: Path) -> np.ndarray:
        import librosa

        audio, _ = librosa.load(str(wav), sr=SAMPLE_RATE, mono=True)
        inputs = self.extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            hidden = self.model(**inputs, output_hidden_states=True).hidden_states[self.layer]
        return hidden.mean(dim=1).squeeze(0).float().cpu().numpy()


def dataset_wavs(audio_data: Path, limit: int | None, seed: int) -> list[Path]:
    """Every clip listed in an audio_data dir (metadata.csv and holdout.csv)."""
    wavs = []
    for name in ("metadata.csv", "holdout.csv"):
        path = audio_data / name
        if path.is_file():
            with path.open(encoding="utf-8", newline="") as fh:
                wavs += [audio_data / row["file_name"] for row in csv.DictReader(fh)]
    if limit and len(wavs) > limit:
        wavs = [wavs[i] for i in sorted(np.random.default_rng(seed).choice(len(wavs), limit, replace=False))]
    return wavs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--positive", type=Path, required=True, help="audio_data dir of Quebec clips")
    train.add_argument("--negative", type=Path, required=True, help="audio_data dir of European French clips")
    train.add_argument("--per-class", type=int, default=1000)
    train.add_argument("--model", default=DEFAULT_MODEL)
    train.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    train.add_argument("--min-accuracy", type=float, default=0.8)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    embed = Embedder(args.model, args.layer, args.device)
    features = {}
    for label, root in (("positive", args.positive), ("negative", args.negative)):
        wavs = dataset_wavs(root, args.per_class, args.seed)
        if len(wavs) < 10:
            print(f"error: only {len(wavs)} {label} clips in {root}", file=sys.stderr)
            return 1
        features[label] = np.stack([embed(w) for w in wavs])
        print(f"{label}: {len(wavs)} clips embedded")
    probe = train_probe(features["positive"], features["negative"], seed=args.seed)
    probe.model, probe.layer = args.model, args.layer
    probe.save(args.out)
    print(f"held-out accuracy {probe.heldout_accuracy:.3f} -> {args.out}")
    if probe.heldout_accuracy < args.min_accuracy:
        print(f"error: accuracy below {args.min_accuracy}; the probe cannot be trusted", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Merge a LoRA adapter, or any training checkpoint, into a T3 ``.safetensors``.

``lora.py`` only exports the final epoch.  Phase 3 compares epochs, so this
turns any ``checkpoint_epoch*_step*.pt`` (or ``final_lora_adapter.pt``) into a
checkpoint the app can load.  The merge works directly on the state dict,
without building the model::

    tfmr.<layer>.weight += (lora_B @ lora_A) * alpha / rank

which is exactly what the toolkit's ``merge_lora_weights`` does to the module.

Example::

    python -m scripts.finetune.merge_adapter \\
        --adapter data/finetune/runs/fr_ca_r16/checkpoint_epoch1_step20000.pt \\
        --out data/finetune/runs/fr_ca_r16/t3_fr_ca_e1.safetensors

The base T3 and LoRA alpha default to the values in the run's ``toolkit.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from scripts.finetune.t3_state import resolve_t3

Pairs = dict[str, tuple[np.ndarray, np.ndarray]]


def adapter_pairs(obj: Mapping) -> tuple[Pairs, float | None]:
    """Return ``{layer: (lora_A, lora_B)}`` and alpha (None when the file does not record it).

    Accepts the toolkit's two formats: ``final_lora_adapter.pt`` (``lora_config`` +
    ``lora_weights``) and ``checkpoint_epoch*_step*.pt`` (``lora_state_dict``).
    """
    if "lora_weights" in obj:
        pairs = {name: (w["lora_A"], w["lora_B"]) for name, w in obj["lora_weights"].items()}
        return pairs, float(obj["lora_config"]["alpha"])
    if "lora_state_dict" in obj:
        flat = obj["lora_state_dict"]
        names = {key.rsplit(".", 1)[0] for key in flat}
        missing = [n for n in names if f"{n}.lora_A" not in flat or f"{n}.lora_B" not in flat]
        if missing:
            raise ValueError(f"Incomplete LoRA pairs in checkpoint: {sorted(missing)[:3]}")
        return {n: (flat[f"{n}.lora_A"], flat[f"{n}.lora_B"]) for n in names}, None
    raise ValueError("Not a LoRA adapter or checkpoint: expected 'lora_weights' or 'lora_state_dict'")


def merge_lora(state: Mapping[str, np.ndarray], pairs: Pairs, alpha: float) -> dict[str, np.ndarray]:
    """Return a new state dict with every LoRA delta added to its projection weight."""
    if not pairs:
        raise ValueError("The adapter holds no LoRA layers")
    merged = dict(state.items())
    for name, (lora_a, lora_b) in sorted(pairs.items()):
        key = f"tfmr.{name}.weight"
        if key not in merged:
            raise KeyError(f"LoRA layer {name!r} has no {key!r} in the base state dict")
        weight = merged[key]
        rank = lora_a.shape[0]
        if lora_b.shape[1] != rank or weight.shape != (lora_b.shape[0], lora_a.shape[1]):
            raise ValueError(f"{name}: LoRA shapes A{lora_a.shape} B{lora_b.shape} do not fit W{weight.shape}")
        delta = (lora_b.astype(np.float32) @ lora_a.astype(np.float32)) * (alpha / rank)
        merged[key] = (weight.astype(np.float32) + delta).astype(weight.dtype)
    return merged


def run_config(adapter: Path) -> dict:
    manifest = adapter.parent / "toolkit.json"
    if manifest.is_file():
        return json.loads(manifest.read_text(encoding="utf-8"))["config"]
    return {}


def _to_numpy(value):
    if isinstance(value, dict):
        return {k: _to_numpy(v) for k, v in value.items()}
    if hasattr(value, "detach"):
        return value.detach().cpu().float().numpy()
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--adapter", type=Path, required=True, help="final_lora_adapter.pt or checkpoint_*.pt")
    parser.add_argument("--out", type=Path, required=True, help="merged .safetensors to write")
    parser.add_argument("--base", help="v3, v2 or a .safetensors (default: base_t3 from toolkit.json, else v3)")
    parser.add_argument("--alpha", type=float, help="LoRA alpha (default: from the adapter or toolkit.json)")
    args = parser.parse_args(argv)

    import torch
    from safetensors.numpy import load_file, save_file

    cfg = run_config(args.adapter)
    base_spec = args.base or cfg.get("base_t3", "v3")
    raw = torch.load(args.adapter, map_location="cpu", weights_only=True)
    pairs, recorded_alpha = adapter_pairs(_to_numpy({k: raw[k] for k in ("lora_weights", "lora_config",
                                                                         "lora_state_dict") if k in raw}))
    alpha = args.alpha if args.alpha is not None else recorded_alpha or cfg.get("lora_alpha")
    if alpha is None:
        print("error: LoRA alpha unknown; pass --alpha (the run's toolkit.json was not found)", file=sys.stderr)
        return 1

    base_path = resolve_t3(base_spec)
    merged = merge_lora(load_file(str(base_path)), pairs, float(alpha))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rank = next(iter(pairs.values()))[0].shape[0]
    save_file(merged, str(args.out), metadata={
        "format": "resembletron_t3_state_dict",
        "base_t3": base_spec,
        "adapter": args.adapter.name,
        "lora_rank": str(rank),
        "lora_alpha": str(alpha),
        "merged_by": "voice-clone-app scripts/finetune/merge_adapter.py",
    })
    print(f"Merged {len(pairs)} LoRA layers (rank {rank}, alpha {alpha}) into {base_spec} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

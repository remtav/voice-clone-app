"""Shared helpers to read and write T3 state dicts without building the model."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np

CHATTERBOX_REPO = "ResembleAI/chatterbox"
OFFICIAL_T3 = {"v2": "t3_mtl23ls_v2.safetensors", "v3": "t3_mtl23ls_v3.safetensors"}
# LoRA targets of the toolkit, as named in the T3 state dict (30 layers × 7 = 210 tensors in v3).
PROJECTION_RE = re.compile(r"^tfmr\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight$")


def resolve_t3(spec: str) -> Path:
    """``v2``/``v3`` → the official file from the Hugging Face cache; anything else is a path."""
    if spec in OFFICIAL_T3:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(CHATTERBOX_REPO, OFFICIAL_T3[spec]))
    path = Path(spec).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"T3 checkpoint not found: {path}")
    return path


class LazyState(Mapping[str, np.ndarray]):
    """Read-only view of a .safetensors file that loads one tensor at a time."""

    def __init__(self, path: Path) -> None:
        from safetensors import safe_open

        self.path = path
        self._file = safe_open(str(path), framework="numpy")
        self._keys = list(self._file.keys())

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self._keys:
            raise KeyError(key)
        return self._file.get_tensor(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def metadata(self) -> dict[str, str]:
        return self._file.metadata() or {}

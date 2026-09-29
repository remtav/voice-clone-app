"""Turn the LoRA toolkit's console output into progress fields and a readable log."""

from __future__ import annotations

import re
from dataclasses import dataclass

# e.g. "Epoch 1/4:  33%|███▎      | 1200/3600 [07:27<14:54,  0.37s/it, loss=2.8600, lr=0.000325]"
_TQDM = re.compile(
    r"Epoch (?P<epoch>\d+)/(?P<epochs>\d+):\s*\d+%\|[^|]*\|\s*(?P<n>\d+)/(?P<total>\d+)"
    r"\s*\[[^<\]]*<[^,\]]*,\s*(?P<rate>[\d.]+)(?P<unit>s/it|it/s)(?:,\s*loss=(?P<loss>[-\d.eE+]+))?"
)
# The toolkit prints tensor shapes for every step; they drown the useful lines.
_NOISE = re.compile(
    r"^(Text tokens shape|Target tokens shape|Batch size|Embeds shape|Conditioning length|Hidden states shape"
    r"|Speech logits slice|Speech logits shape|Target shifted shape|Final shapes|Computed loss)"
)


@dataclass
class Progress:
    epoch: int = 0
    epochs: int = 0
    step: int = 0
    total_steps: int = 0
    loss: float | None = None
    eta_seconds: float | None = None


class ProgressParser:
    def __init__(self) -> None:
        self.progress = Progress()

    def feed(self, line: str) -> bool:
        """Update from one console line; True if it was a training progress line."""
        match = _TQDM.search(line)
        if not match:
            return False
        epoch, epochs = int(match["epoch"]), int(match["epochs"])
        n, per_epoch = int(match["n"]), int(match["total"])
        rate = float(match["rate"])
        seconds_per_step = rate if match["unit"] == "s/it" else (1.0 / rate if rate else 0.0)
        step, total = (epoch - 1) * per_epoch + n, epochs * per_epoch
        p = self.progress
        p.epoch, p.epochs, p.step, p.total_steps = epoch, epochs, step, total
        if match["loss"] is not None:
            p.loss = float(match["loss"])
        p.eta_seconds = round(max(total - step, 0) * seconds_per_step, 1) if seconds_per_step else None
        return True


# Any tqdm bar ("Sampling:  12%|█▏   | 120/1000 [...]"): redrawn many times a second.
_BAR = re.compile(r"\d+%\|.*\|\s*\d+/\d+")


def is_progress_bar(line: str) -> bool:
    return bool(_BAR.search(line))


def is_noise(line: str) -> bool:
    return bool(_NOISE.match(line.strip()))

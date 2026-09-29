"""Fetch the Chatterbox LoRA toolkit at a pinned commit and patch it for a run.

The toolkit (github.com/Ahmed-Ezzat20/chatterbox-finetuning-multilingual, MIT)
is written for Arabic on the v2 checkpoint.  Used as-is it would train a LoRA
on the wrong base and the wrong language token.  This script downloads its two
scripts, checks their SHA-256, and rewrites them for one training run:

* the base model is loaded through ``load_base_model()``: an official T3 by
  name (``v3`` by default) or a fine-tuned ``.safetensors`` file, for both
  training and the final merge (the two must match);
* ``language_id`` comes from ``LANGUAGE_ID`` (``fr``) instead of ``'ar'``;
* the frozen base weights (T3, S3Gen, voice encoder) skip gradient
  computation: only LoRA parameters were ever optimized, so this saves memory
  without changing the result;
* speech targets are framed like inference: ``start_speech_token`` first and
  ``stop_speech_token`` last.  Upstream trains on bare S3 tokens: the model is
  fed no BOS although inference always starts with one, and it never sees the
  token that ends generation.  This is correct by construction, not proven by
  experiment; ``--upstream-speech-targets`` restores the upstream framing to
  compare both on a real run;
* the train/validation split is seeded, the merged T3 is saved as
  ``<output-name>.pt`` and every path is absolute, so the run does not depend
  on the working directory;
* the hyperparameters come from the command line.

The toolkit declares ``WARMUP_STEPS`` but never uses it (the schedule is a
plain cosine decay), so it is not exposed here.

Layout of ``--run-dir``::

    toolkit/lora.py, toolkit/fix_merged_model.py   patched scripts
    toolkit.json                                    config + upstream hashes
    checkpoint_epoch*_step*.pt, final_lora_adapter.pt, training_metrics.png
    merged_model/<output-name>.pt → .safetensors (after fix_merged_model.py)

Example::

    python -m scripts.finetune.setup_toolkit --run-dir data/finetune/runs/fr_ca_r16 \\
        --data-dir data/finetune/qc/audio_data
    python data/finetune/runs/fr_ca_r16/toolkit/lora.py
    python data/finetune/runs/fr_ca_r16/toolkit/fix_merged_model.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

TOOLKIT_REPO = "Ahmed-Ezzat20/chatterbox-finetuning-multilingual"
TOOLKIT_COMMIT = "e9816b4e8292ebf49ae59ce55f07c790764faa96"
UPSTREAM_SHA256 = {
    "lora.py": "fbb7774e189006b73cf421770517a1fe689da43f91d426322ec5afd6948b26d5",
    "fix_merged_model.py": "5deba64b0b6bf96c18bab975eabfff487159d48cd3650a4e768ffc0770cf12f6",
}
# Chatterbox assets shared by every multilingual T3 checkpoint.
SHARED_ASSETS = ["ve.pt", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"]


class PatchError(RuntimeError):
    """An expected snippet was not found exactly once: the upstream file changed."""


@dataclass
class RunConfig:
    data_dir: str
    run_dir: str
    base_t3: str = "v3"
    output_name: str = "t3_fr_ca"
    language_id: str = "fr"
    epochs: int = 4
    learning_rate: float = 2e-5
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    batch_size: int = 1
    grad_accum: int = 8
    min_seconds: float = 1.0
    max_seconds: float = 15.0
    max_text_chars: int = 300
    save_every: int = 500
    val_split: float = 0.05
    seed: int = 1234
    speech_bos_eos: bool = True

    def validate(self) -> None:
        if self.batch_size != 1:
            raise ValueError("The toolkit's collate_fn stacks unpadded audio: batch_size must stay 1")
        if not (self.base_t3 in ("v2", "v3") or self.base_t3.endswith(".safetensors")):
            raise ValueError("--base-t3 must be v2, v3 or a .safetensors file")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.output_name):
            raise ValueError("--output-name must be a plain file stem")
        if not 0 < self.val_split < 1:
            raise ValueError("--val-split must be between 0 and 1")


def replace_once(text: str, old: str, new: str, what: str) -> str:
    count = text.count(old)
    if count != 1:
        raise PatchError(f"{what}: expected 1 occurrence of {old.strip()!r}, found {count}")
    return text.replace(old, new)


def set_constant(text: str, name: str, value: object) -> str:
    pattern = re.compile(rf"^{name} = .*$", re.M)
    if len(pattern.findall(text)) != 1:
        raise PatchError(f"config: expected one '{name} = ...' line")
    return pattern.sub(lambda _: f"{name} = {value!r}", text)


def patch_lora(source: str, cfg: RunConfig) -> str:
    text = source
    for name, value in (
        ("AUDIO_DATA_DIR", cfg.data_dir),
        ("CHECKPOINT_DIR", cfg.run_dir),
        ("BATCH_SIZE", cfg.batch_size),
        ("EPOCHS", cfg.epochs),
        ("LEARNING_RATE", cfg.learning_rate),
        ("MAX_AUDIO_LENGTH", cfg.max_seconds),
        ("MIN_AUDIO_LENGTH", cfg.min_seconds),
        ("LORA_RANK", cfg.lora_rank),
        ("LORA_ALPHA", cfg.lora_alpha),
        ("LORA_DROPOUT", cfg.lora_dropout),
        ("GRADIENT_ACCUMULATION_STEPS", cfg.grad_accum),
        ("SAVE_EVERY_N_STEPS", cfg.save_every),
        ("MAX_TEXT_LENGTH", cfg.max_text_chars),
        ("VALIDATION_SPLIT", cfg.val_split),
    ):
        text = set_constant(text, name, value)

    anchor = f"VALIDATION_SPLIT = {cfg.val_split!r}\n"
    text = replace_once(text, anchor, anchor + f'''
# --- added by scripts/finetune/setup_toolkit.py ------------------------------
LANGUAGE_ID = {cfg.language_id!r}
BASE_T3 = {cfg.base_t3!r}
OUTPUT_NAME = {cfg.output_name!r}
SEED = {cfg.seed!r}
SHARED_ASSETS = {SHARED_ASSETS!r}
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)


def load_base_model(device):
    """Official T3 by name (v2/v3) or a fine-tuned .safetensors file."""
    if BASE_T3.endswith(".safetensors"):
        from huggingface_hub import snapshot_download

        ckpt_dir = snapshot_download(repo_id="ResembleAI/chatterbox", allow_patterns=SHARED_ASSETS)
        return ChatterboxMultilingualTTS.from_local(ckpt_dir, device, t3_model=BASE_T3)
    return ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model=BASE_T3)
# -----------------------------------------------------------------------------
''', "config anchor")

    text = replace_once(
        text,
        "    model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)\n",
        "    model = load_base_model(DEVICE)\n"
        "    # Only LoRA parameters are optimized: skip gradients for the frozen base.\n"
        "    for frozen in (model.t3, model.s3gen, model.ve):\n"
        "        frozen.requires_grad_(False)\n",
        "training base model",
    )
    text = replace_once(
        text,
        "    merged_model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)\n",
        "    merged_model = load_base_model(DEVICE)\n",
        "merge base model",
    )
    text = replace_once(text, '    language_id: str = "ar"\n', "    language_id: str = LANGUAGE_ID\n",
                        "AudioSample default language")
    text = replace_once(text, "language_id='ar'  # Arabic language ID\n", "language_id=LANGUAGE_ID\n",
                        "sample language")
    if cfg.speech_bos_eos:
        text = replace_once(
            text,
            "        target_tokens_list.append(tokens)\n",
            "        # Frame speech like inference: BOS as the first input, EOS as the last target.\n"
            "        tokens = F.pad(tokens.long(), (1, 0), value=model.t3.hp.start_speech_token)\n"
            "        tokens = F.pad(tokens, (0, 1), value=model.t3.hp.stop_speech_token)\n"
            "        target_tokens_list.append(tokens)\n",
            "speech BOS/EOS",
        )
    text = replace_once(text, "    random.shuffle(samples)\n", "    random.seed(SEED)\n    random.shuffle(samples)\n",
                        "seeded split")
    text = replace_once(text, 'merged_dir / "t3_mtl23ls_v2.pt"', 'merged_dir / f"{OUTPUT_NAME}.pt"',
                        "merged T3 name")
    text = replace_once(
        text,
        'metrics_tracker = MetricsTracker(save_path="training_metrics.png", update_interval=2.0)',
        'metrics_tracker = MetricsTracker(save_path=str(Path(CHECKPOINT_DIR) / "training_metrics.png"), '
        "update_interval=2.0)",
        "metrics path",
    )
    return text


def patch_fix_merged(source: str, cfg: RunConfig) -> str:
    text = replace_once(source, 'merged_dir = Path("checkpoints_lora/merged_model")',
                        f'merged_dir = Path({cfg.run_dir!r}) / "merged_model"', "merged dir")
    text = replace_once(text, '"t3_mtl23ls_v2.pt"', f'"{cfg.output_name}.pt"', "merged .pt name")
    return replace_once(text, '"t3_mtl23ls_v2.safetensors"', f'"{cfg.output_name}.safetensors"',
                        "merged .safetensors name")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_upstream(name: str, source_dir: Path | None = None) -> str:
    if source_dir is not None:
        data = (source_dir / name).read_bytes()
    else:
        url = f"https://raw.githubusercontent.com/{TOOLKIT_REPO}/{TOOLKIT_COMMIT}/{name}"
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    digest = sha256(data)
    if digest != UPSTREAM_SHA256[name]:
        raise PatchError(f"{name}: SHA-256 {digest} does not match the pinned {UPSTREAM_SHA256[name]}")
    return data.decode("utf-8")


def setup(cfg: RunConfig, source_dir: Path | None = None) -> Path:
    cfg.validate()
    run_dir = Path(cfg.run_dir)
    toolkit = run_dir / "toolkit"
    toolkit.mkdir(parents=True, exist_ok=True)
    outputs = {
        "lora.py": patch_lora(fetch_upstream("lora.py", source_dir), cfg),
        "fix_merged_model.py": patch_fix_merged(fetch_upstream("fix_merged_model.py", source_dir), cfg),
    }
    for name, text in outputs.items():
        compile(text, name, "exec")  # a bad patch fails here, not hours into training
        (toolkit / name).write_text(text, encoding="utf-8")
    manifest = {
        "config": asdict(cfg),
        "upstream": {"repo": TOOLKIT_REPO, "commit": TOOLKIT_COMMIT, "sha256": UPSTREAM_SHA256},
        "patched_sha256": {name: sha256(text.encode("utf-8")) for name, text in outputs.items()},
    }
    (run_dir / "toolkit.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return toolkit


def _absolute(path: str) -> str:
    # absolute(), not resolve(): keep a symlink's own ".safetensors" name.
    return str(Path(path).expanduser().absolute())


def main(argv: list[str] | None = None) -> int:
    defaults = RunConfig(data_dir="", run_dir="")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, help="output directory of this run")
    parser.add_argument("--data-dir", required=True, help="audio_data dir from prepare_qc_dataset")
    parser.add_argument("--base-t3", default=defaults.base_t3,
                        help="v3 (default), v2, or a fine-tuned .safetensors to continue from")
    parser.add_argument("--output-name", default=defaults.output_name)
    parser.add_argument("--language", default=defaults.language_id, help="Chatterbox language id")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--rank", type=int, default=defaults.lora_rank)
    parser.add_argument("--alpha", type=float, help="LoRA alpha (default: 2 × rank)")
    parser.add_argument("--dropout", type=float, default=defaults.lora_dropout)
    parser.add_argument("--grad-accum", type=int, default=defaults.grad_accum)
    parser.add_argument("--min-seconds", type=float, default=defaults.min_seconds)
    parser.add_argument("--max-seconds", type=float, default=defaults.max_seconds)
    parser.add_argument("--max-text-chars", type=int, default=defaults.max_text_chars)
    parser.add_argument("--save-every", type=int, default=defaults.save_every)
    parser.add_argument("--val-split", type=float, default=defaults.val_split)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--upstream-speech-targets", action="store_true",
                        help="keep upstream's bare speech targets (no BOS/EOS), for comparison")
    parser.add_argument("--source-dir", type=Path, help="use already-downloaded upstream files (offline)")
    args = parser.parse_args(argv)

    base_t3 = args.base_t3 if args.base_t3 in ("v2", "v3") else _absolute(args.base_t3)
    cfg = RunConfig(
        data_dir=_absolute(args.data_dir), run_dir=_absolute(args.run_dir), base_t3=base_t3,
        output_name=args.output_name, language_id=args.language, epochs=args.epochs,
        learning_rate=args.lr, lora_rank=args.rank,
        lora_alpha=args.alpha if args.alpha is not None else 2.0 * args.rank,
        lora_dropout=args.dropout, grad_accum=args.grad_accum, min_seconds=args.min_seconds,
        max_seconds=args.max_seconds, max_text_chars=args.max_text_chars, save_every=args.save_every,
        val_split=args.val_split, seed=args.seed, speech_bos_eos=not args.upstream_speech_targets,
    )
    try:
        toolkit = setup(cfg, args.source_dir)
    except (PatchError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Patched toolkit written to {toolkit}")
    print(f"  train:  python {toolkit / 'lora.py'}")
    print(f"  export: python {toolkit / 'fix_merged_model.py'}")
    print(f"  result: {Path(cfg.run_dir) / 'merged_model' / (cfg.output_name + '.safetensors')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

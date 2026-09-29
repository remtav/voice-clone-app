"""Write a Hugging Face model card for a fine-tuned T3, in the style of Resemble's language packs.

Only for checkpoints trained on public data.  A checkpoint whose training set
contains your own recordings (``audio/own_*`` rows from build_personal_dataset)
encodes your voice: the script refuses to document it for publication.

    python -m scripts.finetune.model_card \\
        --checkpoint data/finetune/runs/fr_ca_r16/merged_model/t3_fr_ca.safetensors \\
        --report data/finetune/eval/fr_ca_r16/report.md --out data/finetune/publish/README.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from scripts.finetune.validate_t3_checkpoint import sha256_file


class PersonalDataError(RuntimeError):
    """The checkpoint was trained on the user's own voice."""


def run_config_for(checkpoint: Path) -> dict:
    """toolkit.json of the run that produced ``<run>/merged_model/<name>.safetensors``, if any."""
    for parent in checkpoint.parents[:2]:
        manifest = parent / "toolkit.json"
        if manifest.is_file():
            return json.loads(manifest.read_text(encoding="utf-8"))
    return {}


def ensure_public(config: dict) -> None:
    data_dir = config.get("config", {}).get("data_dir")
    base = config.get("config", {}).get("base_t3", "v3")
    if base not in ("v2", "v3"):
        raise PersonalDataError(f"trained on top of {base}: a second-stage (personal) checkpoint")
    metadata = Path(data_dir) / "metadata.csv" if data_dir else None
    if metadata and metadata.is_file():
        with metadata.open(encoding="utf-8", newline="") as fh:
            if any(row["file_name"].startswith("audio/own_") for row in csv.DictReader(fh)):
                raise PersonalDataError(f"{metadata} contains your own recordings")


def tensor_summary(checkpoint: Path) -> dict:
    from scripts.finetune.t3_state import LazyState

    state = LazyState(checkpoint)
    shapes = {key: state.shape(key) for key in state}
    dtypes = {state.dtype(key) for key in state}
    return {"tensors": len(shapes), "dtypes": sorted(dtypes), "text_emb": shapes.get("text_emb.weight"),
            "speech_emb": shapes.get("speech_emb.weight"), "metadata": state.metadata()}


def render_card(name: str, repo: str, locale: str, language_id: str, summary: dict, sha256: str, size: int,
                config: dict, report: str | None) -> str:
    cfg = config.get("config", {})
    upstream = config.get("upstream", {})
    training = [f"- Base: `ResembleAI/chatterbox` multilingual T3 `{cfg.get('base_t3', 'v3')}`"]
    if cfg:
        training.append(f"- LoRA rank {cfg.get('lora_rank')}, alpha {cfg.get('lora_alpha')}, "
                        f"{cfg.get('epochs')} epoch(s), learning rate {cfg.get('learning_rate')}; "
                        "LoRA merged into the attention and MLP projection weights")
    if upstream:
        training.append(f"- Trainer: [{upstream.get('repo')}](https://github.com/{upstream.get('repo')}) "
                        f"@ `{str(upstream.get('commit'))[:7]}`, patched by voice-clone-app `setup_toolkit`")
    if report:
        body = report.split("\n", 1)[1].strip()  # drop the report's own title
        evaluation = "\n".join("#" + line if line.startswith("## ") else line for line in body.splitlines())
    else:
        evaluation = "_Not evaluated._"
    return f"""---
license: mit
language:
- {language_id}
pipeline_tag: text-to-speech
base_model: ResembleAI/chatterbox
base_model_relation: finetune
tags:
- chatterbox
- text-to-speech
- voice-cloning
- chatterbox-v3
datasets:
- mozilla-foundation/common_voice_17_0
---

# {name}

Fine-tune of the Chatterbox Multilingual V3 T3 model for **{locale}**, trained on Common Voice
clips tagged with that accent. Load it in place of the official T3; every other asset is shared
with [`ResembleAI/chatterbox`](https://huggingface.co/ResembleAI/chatterbox).

## Usage

```python
from huggingface_hub import hf_hub_download, snapshot_download
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

shared = snapshot_download("ResembleAI/chatterbox", allow_patterns=[
    "ve.pt", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"])
t3 = hf_hub_download("{repo}", "{name}.safetensors")
model = ChatterboxMultilingualTTS.from_local(shared, "cuda", t3_model=t3)
wav = model.generate("Tu viens-tu souper à soir ?", language_id="{language_id}",
                     audio_prompt_path="reference.wav", cfg_weight=0.8)
```

## Language

- Locale: `{locale}`
- Chatterbox language ID: `{language_id}`

## Training

{chr(10).join(training)}
- Data: Common Voice fr (CC0-1.0), clips tagged Québécois / Canadien, prepared with
  [`tontate/f5-tts-quebec-french-finetune`](https://huggingface.co/tontate/f5-tts-quebec-french-finetune)'s
  processed set or voice-clone-app `prepare_qc_dataset`

## Evaluation

{evaluation}

## Checkpoint Metadata

- Tensor count: `{summary['tensors']}`
- Dtype: `{', '.join(summary['dtypes'])}`
- Text embedding shape: `{summary['text_emb']}`
- Speech embedding shape: `{summary['speech_emb']}`
- Size: `{size}` bytes
- SHA256: `{sha256}`

Outputs carry Resemble AI's PerTh watermark, applied by the decoder.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", default="<user>/Chatterbox-Multilingual-fr-ca")
    parser.add_argument("--locale", default="fr-CA")
    parser.add_argument("--language-id", default="fr")
    parser.add_argument("--report", type=Path, help="report.md from eval.report")
    args = parser.parse_args(argv)

    config = run_config_for(args.checkpoint)
    try:
        ensure_public(config)
    except PersonalDataError as exc:
        print(f"error: refusing to write a publication card: {exc}. Keep this checkpoint private.",
              file=sys.stderr)
        return 1
    report = args.report.read_text(encoding="utf-8") if args.report else None
    card = render_card(args.checkpoint.stem, args.repo, args.locale, args.language_id,
                       tensor_summary(args.checkpoint), sha256_file(args.checkpoint),
                       args.checkpoint.stat().st_size, config, report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(card, encoding="utf-8")
    print(f"Model card written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Blind A/B listening test: base vs fine-tune, same sentence, voice and settings.

``make`` copies N random pairs into ``abx/`` with the order shuffled, writes
``pairs.csv`` for the listener and keeps the answer key in ``key.json``.  The
listener fills the ``choice`` column of ``pairs.csv`` with A or B ("which one
sounds most like my accent?"), or leaves it empty for no preference.
``score`` prints the share of answered pairs won by the fine-tune.

    python -m scripts.finetune.eval.abx make --battery data/finetune/eval/fr_ca_r16 \\
        --base base --candidate fr_ca --cfg 0.8 --voice me
    python -m scripts.finetune.eval.abx score --battery data/finetune/eval/fr_ca_r16
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

PAIR_FIELDS = ["pair", "text", "a", "b", "choice"]


def make_pairs(rows: list[dict[str, str]], base: str, candidate: str, cfg: float, voice: str | None,
               n: int, seed: int) -> list[dict]:
    def key(r):
        return r["voice"], r["sentence_id"]

    def pick(name):
        return {key(r): r for r in rows if r["checkpoint"] == name and abs(float(r["cfg_weight"]) - cfg) < 1e-6
                and (voice is None or r["voice"] == voice)}

    base_rows, cand_rows = pick(base), pick(candidate)
    common = sorted(set(base_rows) & set(cand_rows))
    if not common:
        raise ValueError(f"no clip shared by {base!r} and {candidate!r} at cfg {cfg}")
    rng = random.Random(seed)
    pairs = []
    for i, k in enumerate(sorted(rng.sample(common, min(n, len(common))))):
        candidate_first = rng.random() < 0.5
        first, second = (cand_rows[k], base_rows[k]) if candidate_first else (base_rows[k], cand_rows[k])
        pairs.append({"pair": f"{i + 1:02d}", "text": first["text"], "a_wav": first["wav"], "b_wav": second["wav"],
                      "candidate": "A" if candidate_first else "B"})
    return pairs


def score_answers(answers: list[dict[str, str]], key: dict[str, str]) -> tuple[int, int]:
    """Return (pairs won by the candidate, pairs answered)."""
    wins = answered = 0
    for row in answers:
        choice = row.get("choice", "").strip().upper()
        if choice not in ("A", "B"):
            continue
        answered += 1
        wins += choice == key[row["pair"]]
    return wins, answered


def cmd_make(args) -> int:
    with (args.battery / "manifest.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    pairs = make_pairs(rows, args.base, args.candidate, args.cfg, args.voice, args.n, args.seed)
    out = args.battery / "abx"
    out.mkdir(exist_ok=True)
    with (out / "pairs.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        for p in pairs:
            a, b = f"pair_{p['pair']}_A.wav", f"pair_{p['pair']}_B.wav"
            shutil.copyfile(args.battery / p["a_wav"], out / a)
            shutil.copyfile(args.battery / p["b_wav"], out / b)
            writer.writerow({"pair": p["pair"], "text": p["text"], "a": a, "b": b, "choice": ""})
    meta = {"base": args.base, "candidate": args.candidate, "cfg": args.cfg, "voice": args.voice}
    (out / "key.json").write_text(json.dumps({"meta": meta, "key": {p["pair"]: p["candidate"] for p in pairs}},
                                             indent=2), encoding="utf-8")
    print(f"{len(pairs)} pairs in {out}: fill the 'choice' column of pairs.csv (A/B) without opening key.json")
    return 0


def read_abx(battery: Path) -> tuple[int, int] | None:
    out = battery / "abx"
    if not (out / "pairs.csv").is_file() or not (out / "key.json").is_file():
        return None
    with (out / "pairs.csv").open(encoding="utf-8", newline="") as fh:
        answers = list(csv.DictReader(fh))
    return score_answers(answers, json.loads((out / "key.json").read_text(encoding="utf-8"))["key"])


def cmd_score(args) -> int:
    result = read_abx(args.battery)
    if result is None or result[1] == 0:
        print("error: no answered pairs (fill abx/pairs.csv first)", file=sys.stderr)
        return 1
    wins, answered = result
    print(f"fine-tune preferred in {wins}/{answered} pairs ({wins / answered:.0%})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make")
    make.add_argument("--battery", type=Path, required=True)
    make.add_argument("--base", default="base")
    make.add_argument("--candidate", required=True)
    make.add_argument("--cfg", type=float, default=0.8)
    make.add_argument("--voice", help="restrict to one voice (e.g. yours)")
    make.add_argument("-n", type=int, default=10)
    make.add_argument("--seed", type=int, default=0)
    score = sub.add_parser("score")
    score.add_argument("--battery", type=Path, required=True)
    args = parser.parse_args(argv)
    return cmd_make(args) if args.command == "make" else cmd_score(args)


if __name__ == "__main__":
    sys.exit(main())

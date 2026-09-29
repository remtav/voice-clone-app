"""Aggregate ``scores.csv`` into a go / no-go report and pick the best checkpoint.

Each candidate (checkpoint × CFG weight) is compared with the baseline
checkpoint at the same CFG weight.  A candidate passes when:

* WER ≤ baseline + 0.02 and CER ≤ baseline + 0.02 (intelligibility kept);
* speaker similarity ≥ baseline − 0.02 (voice kept);
* mean P(Quebec) ≥ 2 × baseline (accent moved), when the probe was used;
* the blind ABX test, when answered, prefers it in ≥ 70 % of pairs.

The recommended candidate is the passing one with the highest P(Quebec)
(then lowest WER).  Writes ``report.md`` and exits 1 when nothing passes.

    python -m scripts.finetune.eval.report --battery data/finetune/eval/fr_ca_r16 --baseline base
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

WER_MARGIN = 0.02
SIM_MARGIN = 0.02
P_QC_FACTOR = 2.0
ABX_MIN = 0.7
METRICS = ("wer", "cer", "speaker_sim", "p_qc", "seconds")


@dataclass
class Aggregate:
    checkpoint: str
    cfg: float
    n: int
    means: dict[str, float]


@dataclass
class Decision:
    candidate: Aggregate
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


def _mean(values: list[str]) -> float:
    numbers = [float(v) for v in values if v not in ("", None)]
    return sum(numbers) / len(numbers) if numbers else math.nan


def aggregate(rows: list[dict[str, str]]) -> dict[tuple[str, float], Aggregate]:
    groups: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["checkpoint"], round(float(row["cfg_weight"]), 4))].append(row)
    return {k: Aggregate(k[0], k[1], len(g), {m: _mean([r.get(m, "") for r in g]) for m in METRICS})
            for k, g in groups.items()}


def decide(base: Aggregate, cand: Aggregate, abx: tuple[int, int] | None = None) -> Decision:
    b, c = base.means, cand.means
    decision = Decision(cand)
    for metric in ("wer", "cer"):
        if not math.isnan(c[metric]):
            decision.checks[f"{metric} ≤ base + {WER_MARGIN}"] = c[metric] <= b[metric] + WER_MARGIN
    if not math.isnan(c["speaker_sim"]):
        decision.checks[f"speaker_sim ≥ base − {SIM_MARGIN}"] = c["speaker_sim"] >= b["speaker_sim"] - SIM_MARGIN
    if not math.isnan(c["p_qc"]):
        decision.checks[f"p_qc ≥ {P_QC_FACTOR:g} × base"] = c["p_qc"] >= P_QC_FACTOR * b["p_qc"]
    if abx is not None and abx[1] > 0:
        decision.checks[f"ABX ≥ {ABX_MIN:.0%}"] = abx[0] / abx[1] >= ABX_MIN
    if not decision.checks:
        decision.checks["at least one metric scored"] = False
    return decision


def evaluate(aggregates: dict[tuple[str, float], Aggregate], baseline: str,
             abx: dict[tuple[str, float], tuple[int, int]] | None = None) -> list[Decision]:
    abx = abx or {}
    decisions = []
    for (name, cfg), agg in sorted(aggregates.items()):
        if name == baseline:
            continue
        if (baseline, cfg) not in aggregates:
            raise ValueError(f"baseline {baseline!r} has no clips at cfg {cfg}")
        decisions.append(decide(aggregates[(baseline, cfg)], agg, abx.get((name, cfg))))
    return decisions


def best(decisions: list[Decision]) -> Decision | None:
    passing = [d for d in decisions if d.passed]

    def rank(d: Decision):
        p_qc, wer_value = d.candidate.means["p_qc"], d.candidate.means["wer"]
        return (-(p_qc if not math.isnan(p_qc) else -1), wer_value if not math.isnan(wer_value) else math.inf)

    return min(passing, key=rank) if passing else None


def _fmt(value: float) -> str:
    return "–" if math.isnan(value) else f"{value:.3f}"


def render(aggregates, decisions: list[Decision], baseline: str, chosen: Decision | None) -> str:
    lines = ["# fr-CA fine-tune evaluation", "", f"Baseline: `{baseline}`", "",
             "| checkpoint | cfg | clips | WER | CER | speaker sim | P(Quebec) | mean s |",
             "|---|---|---|---|---|---|---|---|"]
    for (name, cfg), agg in sorted(aggregates.items()):
        m = agg.means
        lines.append(f"| {name} | {cfg:g} | {agg.n} | {_fmt(m['wer'])} | {_fmt(m['cer'])} | "
                     f"{_fmt(m['speaker_sim'])} | {_fmt(m['p_qc'])} | {_fmt(m['seconds'])} |")
    lines += ["", "## Go / no-go", ""]
    for d in decisions:
        status = "PASS" if d.passed else "FAIL"
        checks = ", ".join(f"{name} {'✓' if ok else '✗'}" for name, ok in d.checks.items())
        lines.append(f"- **{d.candidate.checkpoint} @ cfg {d.candidate.cfg:g}: {status}** — {checks}")
    lines += ["", "## Recommendation", ""]
    if chosen:
        lines.append(f"Use `{chosen.candidate.checkpoint}` with CFG weight {chosen.candidate.cfg:g}.")
    else:
        lines.append("No candidate passes. Try another epoch, rank 32 / more epochs if P(Quebec) is flat, "
                     "or a lower learning rate if WER degrades.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--battery", type=Path, required=True)
    parser.add_argument("--baseline", default="base")
    args = parser.parse_args(argv)

    scores = args.battery / "scores.csv"
    if not scores.is_file():
        print(f"error: {scores} not found (run eval.score first)", file=sys.stderr)
        return 1
    with scores.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    aggregates = aggregate(rows)

    from scripts.finetune.eval.abx import read_abx

    abx = {}
    result = read_abx(args.battery)
    if result is not None and result[1] > 0:
        import json

        meta = json.loads((args.battery / "abx" / "key.json").read_text(encoding="utf-8"))["meta"]
        abx[(meta["candidate"], round(float(meta["cfg"]), 4))] = result
    try:
        decisions = evaluate(aggregates, args.baseline, abx)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    chosen = best(decisions)
    text = render(aggregates, decisions, args.baseline, chosen)
    (args.battery / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0 if chosen else 1


if __name__ == "__main__":
    sys.exit(main())

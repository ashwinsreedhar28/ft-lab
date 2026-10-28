"""Severity calibration for the static tool on the TRAIN split (never the eval).

    python -m review.tools.calibrate review/agent/results/tools_static_train.json [--tasks review/bench/public_train.jsonl]

Input: a --tools-only results file (per-task findings). Prints, per rule: bug PRs and clean PRs it fires on, the
likelihood ratio P(fire|bug)/P(fire|clean), gold-hunk hits; the mutant-vs-clean-twin view (findings a mutant has that
its clean original does not); gold hits vs the chance baseline; and the CALIBRATED_MEDIUM set the rule produces.
Paste that set into review/tools/static.py and bump SEVERITY_VERSION. Rule (fixed Oct 28): raw-high stays high;
raw-medium stays medium iff >= MIN_BUG bug PRs and LR >= MIN_LR; everything else low.
"""
from __future__ import annotations

import argparse
import collections
import json

from review.tools import Finding, gold_chance, hits_gold
from review.tools.static import raw_severity

MIN_BUG, MIN_LR = 3, 2.0


def rule_table(per: dict[str, list[Finding]], kinds: dict[str, str], gold: dict[str, list]) -> dict[str, dict]:
    """Pure: {rule: {bug, clean, gold, lr, raw}} counting PRs (not findings)."""
    nb = sum(1 for t in per if kinds[t] == "bug"); nc = len(per) - nb
    acc = collections.defaultdict(lambda: {"bug": set(), "clean": set(), "gold": set()})
    for tid, fs in per.items():
        for f in fs:
            acc[f.rule][kinds[tid]].add(tid)
            if kinds[tid] == "bug" and hits_gold([f], gold.get(tid, [])):
                acc[f.rule]["gold"].add(tid)
    out = {}
    for rule, d in acc.items():
        b, c = len(d["bug"]), len(d["clean"])
        rb, rc = (b / nb if nb else 0.0), (c / nc if nc else 0.0)
        out[rule] = {"bug": b, "clean": c, "gold": len(d["gold"]), "raw": raw_severity(rule),
                     "lr": (rb / rc) if rc else (float("inf") if b else 0.0)}
    return out


def calibrated_medium(table: dict[str, dict], min_bug: int = MIN_BUG, min_lr: float = MIN_LR) -> set[str]:
    return {r for r, d in table.items() if d["raw"] == "medium" and d["bug"] >= min_bug and d["lr"] >= min_lr}


def twin_of(tid: str) -> str | None:
    p = tid.split("__")
    return f"clean__{p[1]}__{p[2]}" if tid.startswith("mut__") and len(p) >= 3 else None


def twin_delta(per: dict[str, list[Finding]]) -> dict[str, list[Finding]]:
    """Pure: for each mutant whose clean twin is present, the findings the twin does not have (multiset on
    file/rule/message, so line shifts from the mutation don't count as new)."""
    out = {}
    for tid, fs in per.items():
        tw = twin_of(tid)
        if not tw or tw not in per:
            continue
        base = collections.Counter((f.file, f.rule, f.message) for f in per[tw])
        new = []
        for f in fs:
            k = (f.file, f.rule, f.message)
            if base[k]:
                base[k] -= 1
            else:
                new.append(f)
        out[tid] = new
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--tasks", default="review/bench/public_train.jsonl")
    a = ap.parse_args()
    r = json.load(open(a.results))
    tasks = {t["id"]: t for t in map(json.loads, open(a.tasks))}
    per = {tid: [Finding(**f) for run in runs if run["tool"] == "static" for f in run["findings"]]
           for tid, runs in r["per_task"].items()}
    kinds = {t: tasks[t]["kind"] for t in per}
    gold = {t: tasks[t].get("gold_hunks", []) for t in per}
    nb = sum(1 for k in kinds.values() if k == "bug"); nc = len(per) - nb
    table = rule_table(per, kinds, gold)
    print(f"{a.results}: {nb} bug / {nc} clean PRs\n{'rule':40s}{'raw':>7}{'bug':>5}{'clean':>6}{'gold':>5}{'LR':>7}")
    for rule, d in sorted(table.items(), key=lambda x: (-x[1]["bug"], x[0])):
        lr = "inf" if d["lr"] == float("inf") else f"{d['lr']:.2f}"
        print(f"{rule:40s}{d['raw']:>7}{d['bug']:>5}{d['clean']:>6}{d['gold']:>5}{lr:>7}")
    keep = calibrated_medium(table)
    print(f"\nCALIBRATED_MEDIUM (raw medium, >= {MIN_BUG} bug PRs, LR >= {MIN_LR}): {sorted(keep)}")
    delta = twin_delta(per)
    with_new = {t: fs for t, fs in delta.items() if fs}
    on_gold = sum(hits_gold(fs, gold[t]) for t, fs in with_new.items())
    print(f"mutants vs clean twin: {len(delta)} paired, {len(with_new)} with a finding the twin lacks, {on_gold} on the gold hunk")
    obs = sum(hits_gold(fs, gold[t]) for t, fs in per.items() if kinds[t] == "bug")
    exp = sum(gold_chance(fs, tasks[t]) for t, fs in per.items() if kinds[t] == "bug")
    print(f"gold-hunk hits on bug PRs: observed {obs}, chance {exp:.1f}")


if __name__ == "__main__":
    main()

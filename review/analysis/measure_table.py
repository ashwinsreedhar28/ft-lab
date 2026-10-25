"""Table for the measurement arms: decisions by kind with the scorer's parser, length-capped counts, and the
PR-weighted paired metric (mutant blocked AND its clean twin approved), mean ± half-range over seeds.

    python -m review.analysis.measure_table runs/measure_4b_*/eval_s*.json
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

from review.agent.run import parse_review

_MUT = re.compile(r"^mut__(.+?)__([0-9a-f]+)__")
_CLEAN = re.compile(r"^clean__(.+?)__([0-9a-f]+)$")


def pr_of(task_id: str):
    m = _MUT.match(task_id) or _CLEAN.match(task_id)
    return (m.group(1), m.group(2)) if m else None


def summarize(path: str) -> dict:
    d = json.loads(Path(path).read_text())
    c = collections.Counter(); capped = 0
    dec = {}
    for r in d["rows"]:
        rev, ok = parse_review(r["completion"], (0, 0), 0.0)
        x = rev.decision if ok else "invalid"
        c[(r["kind"], x)] += 1
        capped += r.get("finish_reason") == "length"
        dec[r["task_id"]] = x
    # paired: for each PR with >=1 mutant and a clean twin, weight each mutant 1/n_mutants
    by_pr = collections.defaultdict(lambda: {"mut": [], "clean": None})
    for tid, x in dec.items():
        pr = pr_of(tid)
        if not pr:
            continue
        if tid.startswith("mut__"):
            by_pr[pr]["mut"].append(x)
        else:
            by_pr[pr]["clean"] = x
    pairs = {pr: v for pr, v in by_pr.items() if v["mut"] and v["clean"] is not None}
    paired = sum(sum(m == "block" and v["clean"] == "approve" for m in v["mut"]) / len(v["mut"]) for v in pairs.values())
    return {"label": Path(path).parent.name + "/" + d["label"], "n": d["n"], "acc": d["acc"],
            "bug": [c[("bug", k)] for k in ("approve", "block", "escalate", "invalid")],
            "clean": [c[("clean", k)] for k in ("approve", "block", "escalate", "invalid")],
            "capped": capped, "paired": paired, "n_pairs": len(pairs)}


def main(paths):
    rows = [summarize(p) for p in paths]
    print(f"{'run':24s} {'acc':>5s} | bug  appr blk esc inv | clean appr blk esc inv | capped | paired/PRs")
    for r in rows:
        b, cl = r["bug"], r["clean"]
        print(f"{r['label']:24s} {r['acc']:5.3f} | {b[0]:9d} {b[1]:3d} {b[2]:3d} {b[3]:3d} | {cl[0]:10d} {cl[1]:3d} {cl[2]:3d} {cl[3]:3d} | {r['capped']:6d} | {r['paired']:5.2f}/{r['n_pairs']}")
    # per arm mean ± half-range
    arms = collections.defaultdict(list)
    for r in rows:
        arms[r["label"].split("/")[0]].append(r)
    print("\nper arm (mean ± half-range over seeds): bug approve | clean approve | invalid | paired")
    for arm, rs in sorted(arms.items()):
        def mr(f):
            v = [f(r) for r in rs]; return f"{sum(v)/len(v):5.2f} ±{(max(v)-min(v))/2:4.2f}"
        print(f"{arm:14s} {mr(lambda r: r['bug'][0])} | {mr(lambda r: r['clean'][0])} | {mr(lambda r: r['bug'][3]+r['clean'][3])} | {mr(lambda r: r['paired'])}")


if __name__ == "__main__":
    main(sys.argv[1:])

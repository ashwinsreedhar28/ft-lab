"""Gate x tools matrix from harness v6 result files (review/agent/results/h6_*.json).

    python -m review.analysis.h6_table review/agent/results/h6_*_shwin.json
    python -m review.analysis.h6_table review/agent/results/h6_*_eval.json --by-source

Per row: bug PRs approved / blocked / escalated, clean PRs approved / blocked / escalated, unparsable, the PR-weighted
paired metric of the measurement job (a mutant blocked AND its clean twin approved, each mutant weighted
1/n_mutants for its PR; review/analysis/measure_table.py), bounces, and $/PR. --by-source splits the bug columns
into LLM-made mutants, mechanical mutants and real bugs, because the reviewer tool is the same model family that
wrote the LLM-made mutants.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from review.analysis.measure_table import pr_of


def source(task_id: str) -> str:
    if task_id.startswith("mut__"):
        return "llm" if task_id.endswith("__llm") else "mechanical"
    if task_id.startswith("clean__"):
        return "clean"
    return "real"


def summarize(d: dict) -> dict:
    """Pure: one results dict -> counts."""
    rows = d["reviews"]
    dec = {r["task_id"]: (r["decision"] if r.get("parsed", True) else "invalid") for r in rows}
    kind = {r["task_id"]: r["kind"] for r in rows}
    c = collections.Counter((kind[t], x) for t, x in dec.items())
    by_src = collections.Counter((source(t), x) for t, x in dec.items() if kind[t] == "bug")
    by_pr = collections.defaultdict(lambda: {"mut": [], "clean": None})
    for t, x in dec.items():
        pr = pr_of(t)
        if not pr:
            continue
        if t.startswith("mut__"):
            by_pr[pr]["mut"].append(x)
        else:
            by_pr[pr]["clean"] = x
    pairs = {pr: v for pr, v in by_pr.items() if v["mut"] and v["clean"] is not None}
    paired = sum(sum(m == "block" and v["clean"] == "approve" for m in v["mut"]) / len(v["mut"]) for v in pairs.values())
    n = len(rows)
    return {"label": d["label"], "gate": d.get("model", ""), "tools": "+".join(d.get("tools") or []) or "none",
            "policy": "+".join(d.get("policy") or ["free"]), "notes": d.get("tool_notes", "plain"),
            "bug": [c[("bug", k)] for k in ("approve", "block", "escalate", "invalid")],
            "clean": [c[("clean", k)] for k in ("approve", "block", "escalate", "invalid")],
            "n_bug": sum(1 for k in kind.values() if k == "bug"), "n_clean": sum(1 for k in kind.values() if k == "clean"),
            "src": {s: [by_src[(s, k)] for k in ("approve", "block", "escalate", "invalid")] for s in ("llm", "mechanical", "real")
                    if any(by_src[(s, k)] for k in ("approve", "block", "escalate", "invalid"))},
            "paired": paired, "n_pairs": len(pairs), "bounces": d.get("bounces", 0),
            "usd_per_pr": (d.get("gate_usd", 0.0) / n) if n else 0.0}


def gate_name(g: str) -> str:
    return g.split("/")[-1] if g.startswith("mlx:") else g.replace("claude-", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--by-source", action="store_true")
    a = ap.parse_args()
    rows = [summarize(json.loads(Path(p).read_text())) for p in a.results]
    rows.sort(key=lambda r: (r["gate"], r["tools"], r["policy"], r["notes"]))
    print(f"{'gate':18s} {'tools':22s} {'policy':28s} {'notes':10s} | bug appr blk esc inv | clean appr blk esc inv | paired | bnc | $/PR")
    for r in rows:
        b, cl = r["bug"], r["clean"]
        pa = f"{r['paired']:.1f}/{r['n_pairs']}" if r["n_pairs"] else "-"
        print(f"{gate_name(r['gate']):18s} {r['tools']:22s} {r['policy']:28s} {r['notes']:10s} | "
              f"{b[0]:5d}/{r['n_bug']:<3d}{b[1]:3d} {b[2]:3d} {b[3]:3d} | {cl[0]:7d}/{r['n_clean']:<3d}{cl[1]:3d} {cl[2]:3d} {cl[3]:3d} | "
              f"{pa:>6s} | {r['bounces']:3d} | {r['usd_per_pr']:.3f}")
        if a.by_source:
            for s, v in r["src"].items():
                print(f"{'':18s}   bugs, {s:10s} approve {v[0]:2d}  block {v[1]:2d}  escalate {v[2]:2d}  invalid {v[3]:2d}")


if __name__ == "__main__":
    main()

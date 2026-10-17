"""Score a set of reviews against review-bench tasks. No model calls here."""
from __future__ import annotations

import json
import math
from pathlib import Path

from review.bench.schema import Comment, Hunk, Review, Task, overlaps


def score(tasks: list[Task], reviews: dict[str, Review], strong: dict[str, bool] | None = None) -> dict:
    """strong: task_id -> whether a proposed test reproduced the oracle (computed by the runner)."""
    bugs = [t for t in tasks if t.kind == "bug"]
    clean = [t for t in tasks if t.kind == "clean"]
    r = lambda t: reviews[t.id]  # noqa: E731
    bugs_approved = sum(1 for t in bugs if r(t).decision == "approve")
    clean_approved = sum(1 for t in clean if r(t).decision == "approve")
    escalated = sum(1 for t in tasks if r(t).decision == "escalate")
    blocked = [t for t in tasks if r(t).decision == "block"]
    blocked_bugs = [t for t in blocked if t.kind == "bug"]
    localized = sum(
        1 for t in blocked_bugs if any(overlaps(c, h) for c in r(t).comments for h in t.gold_hunks)
    )
    strong_n = sum(1 for t in blocked_bugs if strong and strong.get(t.id))
    usd = sum(r(t).usd for t in tasks)
    n_b, n_c = len(bugs), len(clean)
    return {
        "n_bug": n_b, "n_clean": n_c,
        "bugs_approved": bugs_approved,
        "bugs_approved_rate_ub95": 1 - (0.05) ** (1 / n_b) if bugs_approved == 0 and n_b else None,  # rule of three, exact
        "auto_ship_rate": clean_approved / n_c if n_c else None,
        "escalation_rate": escalated / len(tasks) if tasks else None,
        "block_precision": len(blocked_bugs) / len(blocked) if blocked else None,
        "localization": localized / len(blocked_bugs) if blocked_bugs else None,
        "strong_localization": strong_n / len(blocked_bugs) if blocked_bugs else None,
        "usd_per_pr": usd / len(tasks) if tasks else None,
    }


def by_split(tasks: list[Task], reviews: dict[str, Review], strong=None) -> dict[str, dict]:
    out = {"all": score(tasks, reviews, strong)}
    for s in sorted({t.split for t in tasks}):
        out[s] = score([t for t in tasks if t.split == s], reviews, strong)
    return out


def load_tasks(path: str | Path) -> list[Task]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    for r in rows:
        r["gold_hunks"] = [Hunk(**h) for h in r.get("gold_hunks", [])]
    return [Task(**r) for r in rows]


def load_reviews(path: str | Path) -> dict[str, Review]:
    out = {}
    for l in Path(path).read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        tid = r.pop("task_id")
        r["comments"] = [Comment(**c) for c in r.get("comments", [])]
        out[tid] = Review(**r)
    return out

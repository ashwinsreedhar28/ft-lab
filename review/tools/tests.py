"""Test-runner stand-in: the suite delta vs the parent commit, as findings.

Reuses `review.agent.loop.suite_outcomes` and its cache (review/agent/runs.json, keyed outcomes:<sha>), so every
task the v5 loop already ran costs no Docker time here. Findings:
  high  a test that passed at the parent fails at the PR head
  high  a test the PR added fails at the PR head
  low   tests the PR removed (one finding, counted)
Pre-existing failures (failing at the parent too) are reported in the note as NOT evidence -- the v4b failure sheet
found 5 of 7 false blocks came from exactly those.

Known limit on review-bench: mutants were selected to survive the suite, so this tool is empty on every synthetic
bug by construction. It matters for real PRs and for suppressing environment-noise blocks.
"""
from __future__ import annotations

import json

from review.tools import Finding, ToolRun

VERSION = "suite-delta v1"


def version(opts: dict | None = None) -> str:
    return VERSION


def test_file(test_id: str) -> str:
    """junit 'tests.test_core.TestX::test_y' -> 'tests/test_core.py' (best effort; line is unknown)."""
    cls = test_id.split("::", 1)[0]
    parts = [p for p in cls.split(".") if p]
    while parts and parts[-1][:1].isupper():
        parts.pop()
    return "/".join(parts) + ".py" if parts else cls


def delta_findings(parent: dict[str, str], head: dict[str, str]) -> tuple[list[Finding], str]:
    """Pure: outcomes {test_id: passed|failed|error|skipped} at parent and head -> (findings, note)."""
    bad = ("failed", "error")
    new_fail = sorted(t for t, st in head.items() if st in bad and parent.get(t) == "passed")
    added = sorted(t for t in head if t not in parent)
    added_fail = [t for t in added if head[t] in bad]
    gone = sorted(t for t in parent if t not in head)
    pre = sum(1 for t, st in head.items() if st in bad and parent.get(t) in bad)
    out = [Finding("tests", "high", test_file(t), 0, 0, "passed at the parent commit, fails at the PR head", rule=t)
           for t in new_fail]
    out += [Finding("tests", "high", test_file(t), 0, 0, "test added by this PR fails at the PR head", rule=t)
            for t in added_fail]
    if gone:
        out.append(Finding("tests", "low", "", 0, 0, f"{len(gone)} test(s) removed by this PR",
                           rule="removed", evidence=", ".join(gone[:8])))
    n_pass = sum(1 for s in head.values() if s == "passed")
    n_bad = sum(1 for s in head.values() if s in bad)
    note = (f"suite at head: {n_pass} passed, {n_bad} failing, of which {pre} also fail at the parent "
            f"(environment or pre-existing, not evidence about this PR); {len(added)} test(s) added")
    return out, note


def run(task: dict, opts: dict | None = None) -> ToolRun:
    from review.agent.loop import RUNS, suite_outcomes
    if not task.get("image"):
        return ToolRun("tests", VERSION, [], "no task image", ok=False)
    subdir = task.get("subdir") or "."
    runs = json.loads(RUNS.read_text()) if RUNS.exists() else {}
    for sha in (task["base_sha"], task["head_sha"]):
        key = f"outcomes:{sha}"
        if key not in runs:
            runs[key] = suite_outcomes(task, sha, subdir)
            RUNS.write_text(json.dumps(runs, indent=1))
    parent, head = runs[f"outcomes:{task['base_sha']}"], runs[f"outcomes:{task['head_sha']}"]
    if not head:
        return ToolRun("tests", VERSION, [], "suite could not be collected at the PR head", ok=False)
    findings, note = delta_findings(parent, head)
    return ToolRun("tests", VERSION, findings, note)

"""Review tools for the harness (Oct 28): every tool emits Findings, the gate reads them.

The point of the layer is the adapter boundary. Runpod's repos already have a review tool; until we know which
one, three stand-ins emit the same `Finding` shape, so the real tool is one more adapter in January:

    static    ruff (pyflakes/bugbear/pylint-errors, isolated config) + pyright (basic), on lines the PR added,
              minus anything already reported at the parent              -> review/tools/static.py
    tests     the suite delta vs the parent (junit at both commits)      -> review/tools/tests.py
    reviewer  an independent model asked to list defects, never to decide -> review/tools/reviewer.py

Tools are run once per task and cached in review/tools/findings.json keyed by tool + version + task + head sha,
so every gate model in a matrix reads identical findings (the reviewer is stochastic; caching makes the
comparison between gates fair).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

CACHE = Path(__file__).parent / "findings.json"
SEVERITIES = ("high", "medium", "low", "info")


@dataclass
class Finding:
    tool: str                 # static | tests | reviewer | ...
    severity: str             # high | medium | low | info
    file: str
    line_start: int           # 0 = not tied to a line (e.g. a failing test)
    line_end: int
    message: str
    rule: str = ""            # ruff/pyright rule, test id, ...
    evidence: str = ""        # short raw excerpt (tool output, failing assertion)
    id: str = ""              # F1, F2, ... assigned per task when rendered


@dataclass
class ToolRun:
    tool: str
    version: str
    findings: list[Finding] = field(default_factory=list)
    note: str = ""            # one-line context the gate should see even with zero findings
    ok: bool = True           # False = the tool could not run (image missing, timeout); says so in the prompt
    wall_s: float = 0.0
    usd: float = 0.0          # API cost of the tool itself (reviewer); never shown to the gate


# ---------------------------------------------------------------- diffs

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def added_lines(diff: str) -> dict[str, set[int]]:
    """{file: new-side line numbers the diff adds}. Deleted files and pure renames contribute nothing."""
    out: dict[str, set[int]] = {}
    cur, ln = None, 0
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            cur = None
        elif line.startswith("+++ "):
            p = line[4:].strip()
            cur = None if p == "/dev/null" else (p[2:] if p.startswith("b/") else p)
            if cur:
                out.setdefault(cur, set())
        elif cur and (m := _HUNK.match(line)):
            ln = int(m.group(1))
        elif cur and line.startswith("+") and not line.startswith("+++"):
            out[cur].add(ln); ln += 1
        elif cur and line.startswith(" "):
            ln += 1
        # "-" lines and "\ No newline" don't advance the new side
    return {f: s for f, s in out.items() if s}


def py_files(diff: str) -> list[str]:
    return sorted(f for f in added_lines(diff) if f.endswith(".py"))


# ---------------------------------------------------------------- registry + cache

def _adapters():
    from review.tools import reviewer, static, tests
    return {"static": static, "tests": tests, "reviewer": reviewer}


def gather(task: dict, tools: list[str], opts: dict | None = None, refresh: bool = False) -> list[ToolRun]:
    """Run (or load cached) tool results for one task. opts: per-tool options, e.g. {"reviewer": {"model": ...}}."""
    opts = opts or {}
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    out = []
    for name in tools:
        mod = _adapters()[name]
        o = opts.get(name, {})
        key = f"{name}:{mod.version(o)}:{task['id']}:{task['head_sha']}"
        if key in cache and not refresh:
            d = cache[key]
            out.append(ToolRun(d["tool"], d["version"], [Finding(**f) for f in d["findings"]], d["note"], d["ok"],
                               d["wall_s"], d.get("usd", 0.0)))
            continue
        t0 = time.time()
        try:
            run = mod.run(task, o)
        except Exception as e:  # a tool failing must not kill a matrix; the gate is told the tool did not run
            run = ToolRun(name, mod.version(o), [], f"{name} failed to run: {type(e).__name__}: {str(e)[:200]}", ok=False)
        run.wall_s = round(time.time() - t0, 1)
        if run.ok:   # a failure (docker down, image missing) is retried next time, never cached
            cache[key] = {**asdict(run), "findings": [asdict(f) for f in run.findings]}
            CACHE.write_text(json.dumps(cache, indent=1))
        out.append(run)
    return out


# ---------------------------------------------------------------- rendering for the gate

def number(runs: list[ToolRun]) -> list[Finding]:
    """Assign F1..Fn across all tools in a stable order (tool order, then severity, then location)."""
    rank = {s: i for i, s in enumerate(SEVERITIES)}
    flat = []
    for r in runs:
        r.findings.sort(key=lambda f: (rank.get(f.severity, 9), f.file, f.line_start, f.rule))   # in place: render order = id order
        flat += r.findings
    for i, f in enumerate(flat, 1):
        f.id = f"F{i}"
    return flat


def render(runs: list[ToolRun], max_per_tool: int = 25) -> str:
    """Text block appended to the gate prompt. Numbering must have been done (number())."""
    if not runs:
        return ""
    lines = ["Review tools ran on this PR. Their findings are evidence, not verdicts: tools miss real bugs and "
             "flag correct code. Cite a finding by its id (F#) when it supports your decision."]
    for r in runs:
        head = f"[{r.tool}: {r.version}]"
        if not r.ok:
            lines.append(f"{head} did not run ({r.note}). Treat this tool as absent, not as clean.")
            continue
        lines.append(f"{head} {len(r.findings)} finding{'s' if len(r.findings) != 1 else ''}" + (f". {r.note}" if r.note else ""))
        for f in r.findings[:max_per_tool]:
            loc = f"{f.file}:{f.line_start}" + (f"-{f.line_end}" if f.line_end > f.line_start else "") if f.line_start else f.file
            rule = f" {f.rule}" if f.rule else ""
            lines.append(f"  {f.id} {f.severity:<6} {loc}{rule}  {f.message}")
        if len(r.findings) > max_per_tool:
            lines.append(f"  … {len(r.findings) - max_per_tool} more not shown")
    return "\n".join(lines)


# ---------------------------------------------------------------- tool-level metrics (independent of the gate)

def hits_gold(findings: list[Finding], gold_hunks, slack: int = 3) -> bool:
    for f in findings:
        if not f.line_start:
            continue
        for h in gold_hunks:
            hf, hs, he = (h.file, h.line_start, h.line_end) if hasattr(h, "file") else (h["file"], h["line_start"], h["line_end"])
            if f.file == hf and f.line_start <= he + slack and f.line_end >= hs - slack:
                return True
    return False


def tool_stats(per_task: dict[str, list[ToolRun]], tasks: list[dict], min_severity: str = "medium") -> dict:
    """Per tool: how often it fires on bug vs clean PRs (any finding at >= min_severity), and how often a finding
    lands on the gold hunk of a bug PR. This is the tool's own recall/precision, before any gate reads it."""
    rank = {s: i for i, s in enumerate(SEVERITIES)}
    cut = rank[min_severity]
    by_id = {t["id"]: t for t in tasks}
    out = {}
    names = sorted({r.tool for runs in per_task.values() for r in runs})
    for name in names:
        s = {"n_bug": 0, "n_clean": 0, "fired_bug": 0, "fired_clean": 0, "gold_hit": 0, "did_not_run": 0,
             "findings": 0, "wall_s": 0.0}
        for tid, runs in per_task.items():
            t = by_id[tid]
            for r in runs:
                if r.tool != name:
                    continue
                s["wall_s"] += r.wall_s
                if not r.ok:
                    s["did_not_run"] += 1
                    continue
                strong = [f for f in r.findings if rank.get(f.severity, 9) <= cut]
                s["findings"] += len(r.findings)
                if t["kind"] == "bug":
                    s["n_bug"] += 1
                    s["fired_bug"] += bool(strong)
                    s["gold_hit"] += hits_gold(r.findings, t.get("gold_hunks", []))
                else:
                    s["n_clean"] += 1
                    s["fired_clean"] += bool(strong)
        s["wall_s"] = round(s["wall_s"], 1)
        out[name] = s
    return out

"""AI-reviewer stand-in: an independent model lists defects in the diff and never decides.

This is the closest stand-in for what a "built-in review tool" usually is on GitHub (CodeRabbit, Copilot review,
a Claude Code review action): comments with file/line/severity. Keeping it separate from the gate is the point --
the results can then say "the tool found it" (gold-hunk hit rate, below any gate) apart from "the gate decided".

Same input the gate sees minus the tools block: diff + mechanically extracted context. Default model Haiku 5.5.
Cached per (model, task, head sha) by review.tools.gather, so every gate in a matrix reads the same comments.
"""
from __future__ import annotations

import json

from review.tools import Finding, ToolRun

DEFAULT_MODEL = "claude-haiku-5-5"
PROMPT_VERSION = "v1"

SYSTEM = (
    "You are an automated code review tool that comments on pull requests. List concrete defects this change "
    "introduces: wrong logic, broken edge cases, wrong conditions or operators, mismatched callers, errors that "
    "would raise at runtime. Do not comment on style, naming, docs, or tests that are merely missing. Do not "
    "decide whether the PR should merge -- another system does that.\n"
    "Respond with exactly one JSON object and nothing else:\n"
    '{"findings": [{"file": "path", "line_start": N, "line_end": N, "severity": "high|medium|low", '
    '"message": "one sentence: what is wrong and what it breaks"}]}\n'
    "Use new-file line numbers from the diff. At most 8 findings, most severe first. If you find no concrete "
    'defect, return {"findings": []}. You have no tools.'
)


def version(opts: dict | None = None) -> str:
    return f"reviewer {(opts or {}).get('model', DEFAULT_MODEL)} prompt-{PROMPT_VERSION}"


def parse(text: str) -> list[Finding] | None:
    """Pure: model text -> findings, or None when no findings object could be parsed."""
    dec = json.JSONDecoder()
    i = 0
    while (j := text.find("{", i)) >= 0:
        try:
            o, end = dec.raw_decode(text[j:])
        except json.JSONDecodeError:
            i = j + 1
            continue
        if isinstance(o, dict) and isinstance(o.get("findings"), list):
            out = []
            for f in o["findings"][:8]:
                if not isinstance(f, dict):
                    continue
                try:
                    ls = int(f.get("line_start") or 0)
                    le = int(f.get("line_end") or ls)
                except (TypeError, ValueError):
                    ls = le = 0
                sev = str(f.get("severity", "medium")).lower()
                out.append(Finding("reviewer", sev if sev in ("high", "medium", "low") else "medium",
                                   str(f.get("file", "")), ls, max(ls, le), " ".join(str(f.get("message", "")).split())[:300]))
            return out
        i = j + end
    return None


def run(task: dict, opts: dict | None = None) -> ToolRun:
    from review.agent.loop import call
    from review.agent.run import build_prompt
    model = (opts or {}).get("model", DEFAULT_MODEL)
    prompt = build_prompt(task, "diff+graph", None)
    text, (ti, to), usd = call(model, SYSTEM, [{"role": "user", "content": prompt}], 8000)
    findings = parse(text)
    if findings is None:
        return ToolRun("reviewer", version(opts), [], f"reviewer output unparsable: {text[:120]!r}", ok=False, usd=usd)
    return ToolRun("reviewer", version(opts), findings, "", usd=usd)

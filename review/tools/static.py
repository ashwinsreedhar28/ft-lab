"""Static stand-in for a repo's review tool: ruff + pyright, on the lines a PR adds.

    python -m review.tools.static --build review/bench/tasks.jsonl     # one-time: <image>-static per task image

Each task image gets a derived image with ruff and pyright (node bundled via `pyright[nodejs]`) installed and
pyright's npm package pre-fetched, so the analysis itself runs with the network off like every other harness step.

Per task, inside the static image: check out the PR head, run ruff (isolated config: pyflakes F, syntax E9,
bugbear B, pylint errors PLE -- defects, not style) and pyright (basic mode, missing-import noise off) on the
changed .py files; check out the parent and run both again on those files that exist there. A finding is kept
when it sits on a line the PR added AND the same (file, rule, message) was not already reported at the parent --
that second filter drops findings on code the PR merely moved.

Expected on review-bench: little. A single-line `and -> or` is lint- and type-clean. That is the point of
measuring it: the "tool says nothing" baseline, at zero cost, and the shape every other adapter fills.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from review.tools import Finding, ToolRun, added_lines, py_files

RUFF = "0.17.3"
PYRIGHT = "1.1.414"
RUFF_SELECT = "F,E9,B,PLE"
PYRIGHT_IGNORE_RULES = {"reportMissingImports", "reportMissingModuleSource", "reportMissingTypeStubs"}

RUFF_HIGH = {"F821", "F822", "F823", "F701", "F702", "F704", "F706"}
RUFF_LOW = {"F401", "F403", "F405", "F541", "F811", "F841"}
PYRIGHT_HIGH = {"reportUndefinedVariable"}


def version(opts: dict | None = None) -> str:
    return f"ruff {RUFF} [{RUFF_SELECT}] + pyright {PYRIGHT} basic"


def static_image(image: str) -> str:
    last = image.rsplit("/", 1)[-1]
    if ":" in last:
        repo, tag = image.rsplit(":", 1)
        return f"{repo}:{tag}-static"
    return f"{image}:static"


DOCKERFILE = f"""FROM {{base}}
RUN pip install -q "ruff=={RUFF}" "pyright[nodejs]=={PYRIGHT}" && python -m ruff --version && python -m pyright --version
"""


def ensure_image(image: str, build: bool = True) -> str | None:
    tag = static_image(image)
    if subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0:
        return tag
    if not build:
        return None
    p = subprocess.run(["docker", "build", "-q", "-t", tag, "-"], input=DOCKERFILE.format(base=image),
                       capture_output=True, text=True, timeout=1800)
    if p.returncode != 0:
        raise RuntimeError(f"building {tag} failed: {p.stderr[-400:]}")
    return tag


CFG_DIR = "/testbed/.review-pyright"   # untracked dir inside the checkout: survives `git checkout -f <parent>`


def _pyright_config(subdir: str, files: list[str]) -> str:
    """Pyright ignores absolute `include` entries ("not relative") and then analyses nothing, so every path is
    relative to CFG_DIR. Files go in `include` rather than on the command line so the settings certainly apply."""
    root = ".." if subdir in (".", "") else f"../{subdir}"
    return json.dumps({"include": [f"../{f}" for f in files], "extraPaths": [root, f"{root}/src"],
                       "typeCheckingMode": "basic", **{r: "none" for r in sorted(PYRIGHT_IGNORE_RULES)}})


def script(task: dict, files: list[str]) -> str:
    """Shell run in the static image: head analysis, then parent analysis, each between markers."""
    from review.agent.loop import prelude
    subdir = task.get("subdir") or "."
    fl = " ".join(shlex.quote(f) for f in files)
    analyse = (
        "FILES=''; for f in {fl}; do [ -f \"$f\" ] && FILES=\"$FILES $f\"; done; "
        "echo RUFF_{side}_BEGIN; [ -n \"$FILES\" ] && python -m ruff check --isolated --select " + RUFF_SELECT +
        " --output-format json --exit-zero $FILES 2>/dev/null; echo RUFF_{side}_END; "
        "echo PYR_{side}_BEGIN; [ -n \"$FILES\" ] && python -m pyright -p " + CFG_DIR + "/pyrightconfig.json --outputjson 2>/dev/null; "
        "echo PYR_{side}_END; "
    )
    return (f"{prelude(task, task['head_sha'])} && cd /testbed && mkdir -p {CFG_DIR} && "
            f"echo {shlex.quote(_pyright_config(subdir, files))} > {CFG_DIR}/pyrightconfig.json && "
            + analyse.format(fl=fl, side="HEAD")
            + f"git checkout -q -f {task['base_sha']} && "
            + analyse.format(fl=fl, side="BASE"))


def _between(out: str, tag: str) -> str:
    if f"{tag}_BEGIN" not in out:
        return ""
    return out.split(f"{tag}_BEGIN", 1)[1].split(f"{tag}_END", 1)[0].strip()


def _rel(path: str, subdir: str) -> str:
    p = path
    for pre in ("/testbed/",):
        if p.startswith(pre):
            p = p[len(pre):]
    return p


def ruff_items(text: str, subdir: str) -> list[dict]:
    try:
        rows = json.loads(text) if text else []
    except json.JSONDecodeError:
        return []
    out = []
    for r in rows:
        code = r.get("code") or "E999"
        out.append({"file": _rel(r.get("filename", ""), subdir), "line": int(r["location"]["row"]),
                    "end": int((r.get("end_location") or r["location"])["row"]), "rule": code,
                    "message": r.get("message", "").strip(), "tool": "ruff"})
    return out


def pyright_items(text: str, subdir: str) -> list[dict]:
    try:
        j = json.loads(text) if text else {}
    except json.JSONDecodeError:
        return []
    out = []
    for d in j.get("generalDiagnostics", []):
        if d.get("severity") != "error" or d.get("rule", "") in PYRIGHT_IGNORE_RULES:
            continue
        rng = d.get("range", {})
        out.append({"file": _rel(d.get("file", ""), subdir), "line": int(rng.get("start", {}).get("line", -1)) + 1,
                    "end": int(rng.get("end", {}).get("line", -1)) + 1, "rule": d.get("rule", "") or "error",
                    "message": " ".join(d.get("message", "").split())[:240], "tool": "pyright"})
    return out


def _severity(item: dict) -> str:
    if item["tool"] == "ruff":
        r = item["rule"]
        if r in RUFF_HIGH or r.startswith(("E9", "PLE")):
            return "high"
        if r in RUFF_LOW:
            return "low"
        return "medium"
    return "high" if item["rule"] in PYRIGHT_HIGH else "medium"


def parse_findings(head_items: list[dict], base_items: list[dict], added: dict[str, set[int]]) -> list[Finding]:
    """Pure: keep head items on added lines whose (file, tool, rule, message) the parent did not already have."""
    seen_at_base = {(b["file"], b["tool"], b["rule"], b["message"]) for b in base_items}
    out, dedupe = [], set()
    for it in head_items:
        lines = added.get(it["file"], set())
        span = set(range(it["line"], max(it["line"], it["end"]) + 1))
        if not (span & lines):
            continue
        if (it["file"], it["tool"], it["rule"], it["message"]) in seen_at_base:
            continue
        k = (it["file"], it["line"], it["rule"], it["message"])
        if k in dedupe:
            continue
        dedupe.add(k)
        out.append(Finding(tool="static", severity=_severity(it), file=it["file"], line_start=it["line"],
                           line_end=max(it["line"], it["end"]), message=it["message"], rule=f"{it['tool']} {it['rule']}"))
    return out


def run(task: dict, opts: dict | None = None) -> ToolRun:
    from review.agent.loop import container_kw, docker_run
    files = py_files(task["diff"])
    if not files:
        return ToolRun("static", version(), [], "no Python files changed")
    if not task.get("image"):
        return ToolRun("static", version(), [], "no task image", ok=False)
    img = ensure_image(task["image"])
    subdir = task.get("subdir") or "."
    rc, out = docker_run(img, script(task, files), timeout=900, keep=None, **container_kw(task))
    if "RUFF_HEAD_BEGIN" not in out:
        return ToolRun("static", version(), [], f"analysis did not start (rc={rc}): {out[-200:]}", ok=False)
    head = ruff_items(_between(out, "RUFF_HEAD"), subdir) + pyright_items(_between(out, "PYR_HEAD"), subdir)
    base = ruff_items(_between(out, "RUFF_BASE"), subdir) + pyright_items(_between(out, "PYR_BASE"), subdir)
    findings = parse_findings(head, base, added_lines(task["diff"]))
    n_head = len(head)
    return ToolRun("static", version(), findings,
                   f"{len(files)} changed .py file(s); {n_head} diagnostics at head, {len(findings)} on added lines and new vs parent")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True, help="tasks jsonl: build <image>-static for every image it uses")
    a = ap.parse_args()
    imgs = sorted({json.loads(l).get("image") for l in Path(a.build).read_text().splitlines() if l.strip()} - {None, ""})
    for img in imgs:
        tag = ensure_image(img)
        print(f"{img:45s} -> {tag}", flush=True)


if __name__ == "__main__":
    main()

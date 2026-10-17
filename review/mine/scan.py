"""Scan public repos for candidate fix commits (git only; no Docker, no tests run).

    python -m review.mine.scan --repos review/mine/repos.yaml --since 2026-06-01 --out review/mine/work

For each repo in repos.yaml: shallow-ish clone (full history after --since), list non-merge
commits that touch >= 1 non-test .py file AND >= 1 test file, whose subject looks like a fix,
and record the test files + the code files + diff size. Output candidates.jsonl, one line per
commit, ready for the Docker validation stage (does the new test fail at parent / pass at commit)
and then build.py's bisect/match/graph on a per-repo image.

Repo selection rules and per-repo install lines live in repos.yaml (why these: README.md here): Python, pytest, CPU-only suite under
~5 min, active after --since, permissive license. The list is hand-picked; this script just mines it.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

FIX_RE = re.compile(r"\b(fix|fixes|fixed|bug|regress|crash|incorrect|wrong|broken|hotfix|repair)\b", re.I)
SKIP_RE = re.compile(r"\b(revert|merge|bump|release|typo|docs?|changelog|lint|format|ci)\b", re.I)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def is_test(f: str) -> bool:
    return bool(re.search(r"(^|/)tests?/|test_[^/]*\.py$|_test\.py$|conftest\.py$", f))


def clone(url: str, dest: Path, since: str) -> Path:
    name = url.rstrip("/").split("/")[-1].removesuffix(".git")
    repo = dest / name
    if not (repo / ".git").exists():
        r = subprocess.run(["git", "clone", "-q", f"--shallow-since={since}", url, str(repo)], capture_output=True, text=True)
        if r.returncode != 0:   # "error processing shallow info": nothing since `since`, or server quirk
            subprocess.run(["rm", "-rf", str(repo)])
            subprocess.run(["git", "clone", "-q", "--depth=400", url, str(repo)], check=True)
    return repo


def scan_repo(repo: Path, since: str, max_lines: int) -> list[dict]:
    out = []
    log = git(repo, "log", "--no-merges", f"--since={since}", "--format=%H%x00%ad%x00%s", "--date=short")
    for line in log.splitlines():
        sha, date, subj = line.split("\x00", 2)
        if SKIP_RE.search(subj) or not FIX_RE.search(subj):
            continue
        numstat = git(repo, "show", "--numstat", "--format=", sha)
        files, lines = [], 0
        for ln in numstat.splitlines():
            parts = ln.split("\t")
            if len(parts) == 3:
                a, d, f = parts
                if a.isdigit() and d.isdigit():
                    lines += int(a) + int(d)
                files.append(f)
        tests = [f for f in files if is_test(f) and f.endswith(".py")]
        code = [f for f in files if not is_test(f) and f.endswith(".py")]
        if not tests or not code or lines > max_lines:
            continue
        out.append({"repo": repo.name, "sha": sha, "date": date, "subject": subj, "lines": lines,
                    "test_files": tests, "code_files": code,
                    "parent": git(repo, "rev-parse", f"{sha}^").strip()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", default="review/mine/repos.yaml")
    ap.add_argument("--since", default="2026-06-01")
    ap.add_argument("--out", default="review/mine/work")
    ap.add_argument("--max-lines", type=int, default=400, help="fix commits bigger than this are rarely one bug")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import yaml
    urls = [r["url"] for r in yaml.safe_load(Path(a.repos).read_text())]
    allc = []
    for url in urls:
        try:
            repo = clone(url, out / "repos", a.since)
            c = scan_repo(repo, a.since, a.max_lines)
            print(f"{repo.name:30s} {len(c):4d} candidate fix commits since {a.since}", flush=True)
            allc += c
        except subprocess.CalledProcessError as e:
            print(f"{url}: git failed: {e.stderr[-200:] if e.stderr else e}", flush=True)
    (out / "candidates.jsonl").write_text("\n".join(json.dumps(c) for c in allc) + "\n")
    print(f"{len(allc)} candidates -> {out / 'candidates.jsonl'}")


if __name__ == "__main__":
    main()

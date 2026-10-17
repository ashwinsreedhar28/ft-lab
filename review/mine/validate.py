"""Validate mined fix-commit candidates in Docker (Mac stage).

    python -m review.mine.validate images            # build rb-mine-<repo> per repo in repos.yaml
    python -m review.mine.validate run [--repo R] [--limit N] [--full-suite]

For each candidate in work/candidates.jsonl: check out the PARENT, overlay the commit's test files,
run exactly those test files; then check out the COMMIT and run them again. A candidate is

    valid             >= 1 test passes at the commit and FAILS (assertion) at the parent  -> oracle tests
    interface_leak    tests only ERROR at the parent (import of a name the fix adds): no bisect signal
    not_reproduced    the new tests pass at the parent too (refactor labelled "fix", or test is vacuous)
    not_green         some new test does not pass at the commit itself
    install_fail      pip install -e . failed at either sha (deps drifted from the image's HEAD)
    timeout           either run exceeded --timeout
    no_image          `make mine-images` failed for that repo; fix its install line

Writes work/validated.jsonl (every candidate, with status, oracle_tests, and pytest tails) and prints
a per-repo table. Only `valid` rows go on to the introducing-commit bisect (review/bench/build.py
logic, per-repo image), then match/graph/coverage as for shwin-bench.

Network is off inside containers; the image carries the deps. Each candidate costs two pytest runs of
the touched test files only (seconds), not the full suite; --full-suite adds a suite run at the commit
(minutes per candidate on pydantic/fastapi) and records it as `suite_green`.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

from review.bench.build import _parse_results

HERE = Path(__file__).parent
WORK = HERE / "work"
IMG = "rb-mine-{name}"

DOCKERFILE = """FROM python:3.12-slim
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends git build-essential >/dev/null && rm -rf /var/lib/apt/lists/*
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
# build backends up front: per-candidate installs run with the network off and --no-build-isolation
RUN pip install -q --upgrade pip && pip install -q setuptools wheel setuptools-scm flit_core hatchling hatch-vcs hatch-fancy-pypi-readme pdm-backend poetry-core editables
COPY . /seed
WORKDIR /seed
RUN {install}
"""


def repos() -> dict[str, dict]:
    return {r["name"]: r for r in yaml.safe_load((HERE / "repos.yaml").read_text())}


def build_images(only: str | None):
    for name, r in repos().items():
        if only and name != only:
            continue
        clone = WORK / "repos" / name
        if not (clone / ".git").exists():
            print(f"{name:10s} no clone at {clone} (run scan first)"); continue
        df = WORK / "images" / name
        df.mkdir(parents=True, exist_ok=True)
        (df / "Dockerfile").write_text(DOCKERFILE.format(install=r["install"]))
        print(f"{name:10s} building {IMG.format(name=name)} …", flush=True)
        p = subprocess.run(["docker", "build", "-q", "-t", IMG.format(name=name), "-f", str(df / "Dockerfile"), str(clone)],
                           capture_output=True, text=True)
        print(f"{name:10s} {'ok' if p.returncode == 0 else 'FAILED: ' + p.stderr[-600:]}", flush=True)


def docker(name: str, script: str, timeout: int, network: bool = False) -> tuple[int, str]:
    clone = WORK / "repos" / name
    try:
        p = subprocess.run(["docker", "run", "--rm", "--network", "bridge" if network else "none", "-v", f"{clone}:/repo:ro",
                            IMG.format(name=name), "bash", "-lc", script],
                           capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        return 124, (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")


DEFAULT_CANDIDATE_INSTALL = "pip install -q -e . --no-deps --no-build-isolation"


def run_tests_script(sha: str, test_files: list[str], overlay_from: str | None, full_suite: bool,
                     install: str = DEFAULT_CANDIDATE_INSTALL) -> str:
    """Clone /repo -> /w, check out `sha`, optionally overlay test files from `overlay_from`, install, run."""
    files = " ".join(shlex.quote(f) for f in test_files)
    overlay = ""
    if overlay_from:
        overlay = " && ".join(f"mkdir -p $(dirname {shlex.quote(f)}) && (git show {overlay_from}:{shlex.quote(f)} > {shlex.quote(f)} || true)"
                              for f in test_files)
        overlay = f" && {overlay}"
    suite = ""
    if full_suite:
        suite = (" ; echo '=== SUITE'; timeout 1500 python -m pytest -q -x -p no:cacheprovider --no-header -rN 2>&1 | tail -n 5; "
                 "echo SUITE_EXIT=${PIPESTATUS[0]}")
    return (f"git clone -q /repo /w && cd /w && git checkout -q {sha}{overlay}"
            f" && ({install} >/tmp/pip.log 2>&1 || {{ echo INSTALL_FAIL; tail -n 20 /tmp/pip.log; exit 3; }})"
            f" && {{ python -m pytest -rA -q -p no:cacheprovider --no-header {files} 2>&1 | tail -n 400; echo EXIT=${{PIPESTATUS[0]}}; }}{suite}")


def classify(parent: dict[str, str], commit: dict[str, str], p_out: str, c_out: str) -> tuple[str, list[str]]:
    if "INSTALL_FAIL" in p_out or "INSTALL_FAIL" in c_out:
        return "install_fail", []
    if not commit:
        return "not_green", []           # nothing collected at the commit (collection error / no tests)
    if any(s != "passed" for s in commit.values()):
        return "not_green", []
    oracle = [t for t, s in commit.items() if parent.get(t) == "failed"]
    if oracle:
        return "valid", oracle
    if any(s == "error" for s in parent.values()) or not parent:
        return "interface_leak", []
    return "not_reproduced", []


def run(only: str | None, limit: int, timeout: int, full_suite: bool):
    cands = [json.loads(l) for l in (WORK / "candidates.jsonl").read_text().splitlines() if l.strip()]
    if only:
        cands = [c for c in cands if c["repo"] == only]
    if limit:
        cands = cands[:limit]
    done = {}
    out_p = WORK / "validated.jsonl"
    if out_p.exists():
        for l in out_p.read_text().splitlines():
            if l.strip():
                d = json.loads(l); done[(d["repo"], d["sha"])] = d
    cfg = repos()
    with out_p.open("a") as fh:
        for c in cands:
            key = (c["repo"], c["sha"])
            if key in done:
                continue
            if c["repo"] not in cfg:
                print(f"{c['repo']:10s} {c['sha'][:10]} skipped: not in repos.yaml"); continue
            r = cfg[c["repo"]]
            net, inst = bool(r.get("network")), r.get("candidate_install", DEFAULT_CANDIDATE_INSTALL)
            if any(f.startswith(tuple(r.get("exclude_paths", []))) for f in c["test_files"] if r.get("exclude_paths")):
                status, oracle, p_out, c_out = "excluded", [], "", ""
                row = {**c, "status": status, "oracle_tests": oracle, "suite_green": None, "parent_tail": "", "commit_tail": ""}
                fh.write(json.dumps(row) + "\n"); fh.flush(); done[key] = row
                print(f"{c['repo']:10s} {c['sha'][:10]} {status:15s} oracle= 0 lines={c['lines']:4d}  {c['subject'][:60]}", flush=True)
                continue
            rc_p, p_out = docker(c["repo"], run_tests_script(c["parent"], c["test_files"], c["sha"], False, inst), timeout, net)
            rc_c, c_out = docker(c["repo"], run_tests_script(c["sha"], c["test_files"], None, full_suite, inst), timeout, net)
            if "Unable to find image" in p_out or "Unable to find image" in c_out:
                status, oracle = "no_image", []
            elif rc_p == 124 or rc_c == 124:
                status, oracle = "timeout", []
            else:
                status, oracle = classify(_parse_results(p_out), _parse_results(c_out), p_out, c_out)
            suite_green = None
            if full_suite and "SUITE_EXIT=" in c_out:
                suite_green = c_out.rsplit("SUITE_EXIT=", 1)[1].strip().split()[0] == "0"
            row = {**c, "status": status, "oracle_tests": oracle, "suite_green": suite_green,
                   "parent_tail": p_out[-1500:], "commit_tail": c_out[-1500:]}
            fh.write(json.dumps(row) + "\n"); fh.flush()
            done[key] = row
            print(f"{c['repo']:10s} {c['sha'][:10]} {status:15s} oracle={len(oracle):2d} lines={c['lines']:4d}  {c['subject'][:60]}", flush=True)
    table(done.values())


def table(rows):
    by = {}
    for r in rows:
        by.setdefault(r["repo"], Counter())[r["status"]] += 1
    cols = ["valid", "interface_leak", "not_reproduced", "not_green", "install_fail", "no_image", "timeout", "excluded"]
    print("\nrepo        " + " ".join(f"{c[:10]:>10s}" for c in cols))
    tot = Counter()
    for repo, cnt in sorted(by.items()):
        print(f"{repo:11s} " + " ".join(f"{cnt.get(c, 0):10d}" for c in cols)); tot += cnt
    print(f"{'total':11s} " + " ".join(f"{tot.get(c, 0):10d}" for c in cols))
    print(f"\nvalid candidates -> next: bisect each to its introducing commit (review/mine/bisect, to write).")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("images"); a.add_argument("--repo")
    b = sub.add_parser("run"); b.add_argument("--repo"); b.add_argument("--limit", type=int, default=0)
    b.add_argument("--timeout", type=int, default=900); b.add_argument("--full-suite", action="store_true")
    t = sub.add_parser("table")
    args = ap.parse_args()
    if args.cmd == "images":
        build_images(args.repo)
    elif args.cmd == "run":
        run(args.repo, args.limit, args.timeout, args.full_suite)
    else:
        p = WORK / "validated.jsonl"
        table([json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else [])


if __name__ == "__main__":
    main()

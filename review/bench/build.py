"""Build review-bench from shwin-bench.

Stages (run in order; each writes under review/bench/work/):
  prepare  git only   : per fix commit, the ancestor chain of base_commit with diff stats,
                        and the pool of clean-commit candidates.          -> candidates.json
  bisect   docker     : find the commit where the fix's tests first fail. -> positives.json
  match    git only   : one size/file-matched clean commit per positive.  -> negatives.json
  verify   docker     : suite green at parent and commit for each negative. -> negatives.verified.json
  graph    git only   : two-hop neighborhood for every PR.                -> tasks.jsonl (final)

    python -m review.bench.build prepare --shwin ~/shwin-bench --repos ~
    python -m review.bench.build bisect  --shwin ~/shwin-bench            # needs docker (Mac terminal)
    python -m review.bench.build match
    python -m review.bench.build verify  --shwin ~/shwin-bench            # needs docker
    python -m review.bench.build graph   --repos ~

Docker stages use shwin-bench's task images, shwin/<repo>:<base_commit[:12]>, which are
full clones checked out at base_commit (so any ancestor can be checked out inside).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

WORK = Path(__file__).parent / "work"
MAX_ANCESTORS = 400        # walk to the repo root if needed; the error boundary stops early
MAX_POS_LINES = 1500       # aether commits are squash-merged PRs; size is recorded and sliced at report time
SIZE_TOL = 0.30


# ---------- git helpers ----------
def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def diff_stat(repo: Path, sha: str) -> tuple[int, list[str]]:
    """(changed lines, files) for a commit vs its first parent."""
    out = git(repo, "show", "--numstat", "--format=", "--first-parent", sha)
    lines, files = 0, []
    for ln in out.splitlines():
        parts = ln.split("\t")
        if len(parts) == 3:
            a, d, f = parts
            if a.isdigit() and d.isdigit():
                lines += int(a) + int(d)
            files.append(f)
    return lines, files


def is_test_file(f: str) -> bool:
    return bool(re.search(r"(^|/)tests?/|test_[^/]*\.py$|_test\.py$|conftest\.py$", f))


def first_parent_chain(repo: Path, sha: str, n: int) -> list[str]:
    return git(repo, "rev-list", "--first-parent", f"--max-count={n}", sha).split()


def commit_meta(repo: Path, sha: str) -> dict:
    subj, date, parents = git(repo, "show", "-s", "--format=%s%n%ad%n%P", "--date=short", sha).split("\n")[:3]
    return {"sha": sha, "subject": subj, "date": date, "merge": len(parents.split()) > 1}


# ---------- prepare ----------
def load_shwin(shwin: Path) -> list[dict]:
    return [json.loads(l) for l in (shwin / "dataset/tasks.jsonl").read_text().splitlines() if l.strip()]


def prepare(shwin: Path, repos: Path):
    rows = load_shwin(shwin)
    by_fix: dict[str, dict] = {}
    for r in rows:
        fx = by_fix.setdefault(r["fix_commit"], {
            "repo_name": r["repo_name"], "repo_url": r["repo"], "fix_commit": r["fix_commit"],
            "base_commit": r["base_commit"], "split": r["split"], "environment": r["environment"],
            "fail_to_pass": set(), "test_files": set(), "task_ids": [],
        })
        if r.get("meta", {}).get("interface_leak"):
            fx.setdefault("leaked_task_ids", []).append(r["instance_id"])
            continue  # test imports a name the fix adds: errors at every pre-fix commit, no bisect signal
        fx["fail_to_pass"].update(r["FAIL_TO_PASS"])
        fx["test_files"].update(t.split("::")[0] for t in r["FAIL_TO_PASS"])
        fx["task_ids"].append(r["instance_id"])
    fixes, pools = [], {}
    for fx in by_fix.values():
        if not fx["fail_to_pass"]:
            print(f"{fx['repo_name']} {fx['fix_commit'][:7]}: all {len(fx.get('leaked_task_ids', []))} tasks are interface leaks; skipped")
            continue
        repo = repos / fx["repo_name"]
        subdir = fx["environment"].get("subdir") or ""
        chain = first_parent_chain(repo, fx["base_commit"], MAX_ANCESTORS)
        fx["ancestors"] = [{**commit_meta(repo, s), "lines": diff_stat(repo, s)[0]} for s in chain]
        fx["fail_to_pass"] = sorted(fx["fail_to_pass"])
        fx["test_files"] = sorted((Path(subdir) / t).as_posix() if subdir else t for t in fx["test_files"])
        fixes.append(fx)
        if fx["repo_name"] not in pools:
            pools[fx["repo_name"]] = clean_pool(repo)
    WORK.mkdir(exist_ok=True)
    (WORK / "candidates.json").write_text(json.dumps({"fixes": fixes, "pools": pools}, indent=1))
    print(f"{len(fixes)} fix commits, pools: " + ", ".join(f"{k}={len(v)}" for k, v in pools.items()))


def clean_pool(repo: Path) -> list[dict]:
    """Non-merge commits touching ≥1 non-test .py file, with stats; candidates for negatives."""
    out = []
    for sha in git(repo, "rev-list", "--no-merges", "HEAD").split():
        lines, files = diff_stat(repo, sha)
        code = [f for f in files if f.endswith(".py") and not is_test_file(f)]
        if not code or lines == 0:
            continue
        m = commit_meta(repo, sha)
        m.update(lines=lines, files=files, code_files=code,
                 fixlike=bool(re.search(r"\bfix|bug|regress|crash|revert|hotfix", m["subject"], re.I)))
        out.append(m)
    return out


# ---------- bisect (docker) ----------
def docker_run(image: str, script: str, timeout: int = 900) -> tuple[int, str]:
    p = subprocess.run(["docker", "run", "--rm", "--network", "none", image, "bash", "-lc", script],
                       capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


_RESULT_RE = re.compile(r"^(PASSED|FAILED|ERROR) (\S+)", re.M)


def _parse_results(out: str) -> dict[str, str]:
    """pytest -rA summary lines -> {test_id: passed|failed|error}."""
    res = {}
    for status, tid in _RESULT_RE.findall(out):
        tid = tid.split(" - ")[0]
        res[tid] = {"PASSED": "passed", "FAILED": "failed", "ERROR": "error"}[status]
    return res


def bisect(shwin: Path):
    """Per-test bisect: walk ancestors back from base; a test's introducing commit is the child
    of the first ancestor where it passes. Tests that never pass inside the window are dropped.
    One positive per distinct introducing commit (several tests may pin the same one)."""
    cand = json.loads((WORK / "candidates.json").read_text())
    cache_p = WORK / "bisect.cache.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    positives, log = [], []
    for fx in cand["fixes"]:
        image = f"shwin/{fx['repo_name']}:{fx['base_commit'][:12]}"
        subdir = fx["environment"].get("subdir") or "."
        test_cmd = fx["environment"]["test_cmd"]
        tests = " ".join(f"'{t}'" for t in fx["fail_to_pass"])
        tfiles = " ".join(fx["test_files"])
        pending = set(fx["fail_to_pass"])          # tests whose introducing commit is still unknown
        last_failed_at: dict[str, int] = {}        # test -> index of the last ancestor where it failed
        introduced: dict[str, int] = {}            # test -> index of its introducing ancestor
        for i, anc in enumerate(fx["ancestors"]):
            if not pending:
                break
            key = f"{fx['fix_commit'][:12]}:{anc['sha'][:12]}"
            if key in cache and "tests" in cache[key]:
                results = cache[key]["tests"]
            else:
                script = (f"{{ cd /testbed && git checkout -q {anc['sha']} && git checkout -q {fx['fix_commit']} -- {tfiles} "
                          f"&& cd {subdir} && {test_cmd} {tests}; echo EXIT=$?; }} 2>&1 | tail -n 60")
                rc, out = docker_run(image, script)
                if "Unable to find image" in out or "docker: " in out:
                    print(f"     image/docker problem for {image}: build it in shwin-bench first", flush=True)
                    break
                results = _parse_results(out)
                if not results:   # collection error / checkout failure: every pending test is unknown here
                    print("     " + "\n     ".join(out.strip().splitlines()[-6:]), flush=True)
                cache[key] = {"tests": results, "tail": "\n".join(out.strip().splitlines()[-6:])}
                cache_p.write_text(json.dumps(cache, indent=1))
            summary = {"passed": 0, "failed": 0, "error": 0}
            for t in list(pending):
                st = results.get(t, "error")
                summary[st] += 1
                if st == "failed":
                    last_failed_at[t] = i
                elif st == "passed":
                    if t in last_failed_at:
                        introduced[t] = last_failed_at[t]
                    pending.discard(t)     # passes here: introducing commit is the last failing ancestor (if any)
                elif st == "error" and i > 0:
                    # the test can't even import here: the module it exercises doesn't exist yet. The bug was
                    # born with the module, in the last failing ancestor (its child). That is the introducing PR.
                    if t in last_failed_at:
                        introduced[t] = last_failed_at[t]
                    pending.discard(t)
            log.append({"fix": fx["fix_commit"][:7], "ancestor": anc["sha"][:7], "i": i, **summary})
            print(f"{fx['repo_name']} {fx['fix_commit'][:7]} anc[{i}] {anc['sha'][:7]} "
                  f"pass={summary['passed']} fail={summary['failed']} err={summary['error']} pending={len(pending)}", flush=True)
            if results and all(results.get(t) == "error" for t in fx["fail_to_pass"]) and i == 0:
                print("     every oracle test errors at base; skipping this fix", flush=True)
                break
        by_commit: dict[int, list[str]] = {}
        for t, idx in introduced.items():
            by_commit.setdefault(idx, []).append(t)
        if not by_commit:
            print(f"  -> {fx['repo_name']} {fx['fix_commit'][:7]}: no test found its introducing commit in the window", flush=True)
            continue
        repo = Path(shwin).parent / fx["repo_name"] if (Path(shwin).parent / fx["repo_name"]).exists() else None
        for idx, ts in sorted(by_commit.items()):
            anc = fx["ancestors"][idx]
            if anc["lines"] > MAX_POS_LINES:
                print(f"  -> dropped: introducing commit {anc['sha'][:7]} has {anc['lines']} changed lines", flush=True)
                continue
            # The fail->error boundary can also mean "the oracle test's imports became satisfiable", not
            # "the buggy code appeared". Require the introducing commit to touch a code file the fix touched.
            if repo is not None:
                _, fi = diff_stat(repo, anc["sha"]); _, ff = diff_stat(repo, fx["fix_commit"])
                code = lambda fs: {f for f in fs if not is_test_file(f) and f.endswith((".py", ".ts", ".tsx", ".js"))}  # noqa: E731
                if not (code(fi) & code(ff)):
                    print(f"  -> dropped: {anc['sha'][:7]} shares no code file with the fix {fx['fix_commit'][:7]} (import boundary, not the bug)", flush=True)
                    continue
            positives.append({**{k: fx[k] for k in ("repo_name", "repo_url", "fix_commit", "base_commit", "split",
                                                      "test_files", "task_ids", "environment")},
                              "fail_to_pass": sorted(ts), "introducing": anc["sha"], "introducing_lines": anc["lines"],
                              "introducing_subject": anc["subject"]})
            print(f"  -> positive: {anc['sha'][:7]} ({anc['lines']} lines, {len(ts)} tests) {anc['subject'][:60]}", flush=True)
    (WORK / "positives.json").write_text(json.dumps(positives, indent=1))
    (WORK / "bisect.log.json").write_text(json.dumps(log, indent=1))
    print(f"{len(positives)} positives from {len(cand['fixes'])} fix commits")


# ---------- match ----------
def match(repos: Path):
    cand = json.loads((WORK / "candidates.json").read_text())
    positives = json.loads((WORK / "positives.json").read_text())
    excluded = {p["introducing"] for p in positives} | {p["fix_commit"] for p in positives}
    for fx in cand["fixes"]:
        excluded.update(a["sha"] for a in fx["ancestors"][:1])  # base commits (contain the bug)
    used, negatives = set(), []
    for p in positives:
        repo = repos / p["repo_name"]
        plines, pfiles = diff_stat(repo, p["introducing"])
        pcode = {f for f in pfiles if not is_test_file(f)}
        pool = [c for c in cand["pools"][p["repo_name"]]
                if c["sha"] not in excluded and c["sha"] not in used and not c["fixlike"]
                and abs(c["lines"] - plines) <= SIZE_TOL * max(plines, 20)]
        if not pool:
            print(f"no size match for {p['introducing'][:7]} ({plines} lines); widening")
            pool = sorted((c for c in cand["pools"][p["repo_name"]] if c["sha"] not in excluded | used and not c["fixlike"]),
                          key=lambda c: abs(c["lines"] - plines))[:5]
        if not pool:
            continue
        best = max(pool, key=lambda c: (len(pcode & set(c["code_files"])), -abs(c["lines"] - plines)))
        used.add(best["sha"])
        negatives.append({"repo_name": p["repo_name"], "repo_url": p["repo_url"], "split": p["split"],
                          "environment": p["environment"], "sha": best["sha"], "lines": best["lines"],
                          "paired_with": p["introducing"], "file_overlap": len(pcode & set(best["code_files"]))})
    (WORK / "negatives.json").write_text(json.dumps(negatives, indent=1))
    print(f"{len(negatives)} negatives for {len(positives)} positives; "
          f"{sum(1 for n in negatives if n['file_overlap'])} share files with their positive")


# ---------- verify (docker) ----------
def verify(shwin: Path):
    negatives = json.loads((WORK / "negatives.json").read_text())
    cand = json.loads((WORK / "candidates.json").read_text())
    base_by_repo = {fx["repo_name"]: fx["base_commit"] for fx in cand["fixes"]}
    ok = []
    for n in negatives:
        image = f"shwin/{n['repo_name']}:{base_by_repo[n['repo_name']][:12]}"
        subdir = n["environment"].get("subdir") or "."
        cmd = n["environment"]["test_cmd"]
        green = []
        for sha in (f"{n['sha']}^", n["sha"]):
            rc, out = docker_run(image, f"cd /testbed && git checkout -q {sha} && cd {subdir} && {cmd} -q 2>&1 | tail -3; echo EXIT=${{PIPESTATUS[0]}}")
            m = re.search(r"EXIT=(\d+)", out)
            green.append(bool(m) and m.group(1) == "0")
        n["suite_green_parent"], n["suite_green_commit"] = green
        status = "ok" if all(green) else "DROP"
        print(f"{n['repo_name']} {n['sha'][:7]} parent={green[0]} commit={green[1]} {status}")
        if all(green):
            ok.append(n)
    (WORK / "negatives.verified.json").write_text(json.dumps(ok, indent=1))
    print(f"{len(ok)}/{len(negatives)} negatives verified")


# ---------- graph + emit ----------
def pr_diff(repo: Path, sha: str) -> str:
    return git(repo, "show", "--format=", "--first-parent", sha)


def changed_defs(repo: Path, sha: str) -> list[tuple[str, str]]:
    """(file, function/class name) pairs whose body contains a changed line, at `sha`."""
    import ast
    out = []
    numstat = git(repo, "show", "--numstat", "--format=", "--first-parent", sha)
    files = [l.split("\t")[2] for l in numstat.splitlines() if l.count("\t") == 2 and l.split("\t")[2].endswith(".py")]
    for f in files:
        try:
            src = git(repo, "show", f"{sha}:{f}")
        except subprocess.CalledProcessError:
            continue  # deleted file
        hunks = re.findall(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", git(repo, "show", "--format=", "--first-parent", sha, "--", f), re.M)
        ranges = [(int(a), int(a) + int(b or 1)) for a, b in hunks]
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                lo, hi = node.lineno, getattr(node, "end_lineno", node.lineno)
                if any(lo <= e and hi >= s for s, e in ranges):
                    out.append((f, node.name))
    return out


def neighborhood(repo: Path, sha: str, max_chars: int = 6000) -> str:
    """Two hops out from the diff, as text: changed defs, their callers, tests importing the
    modules, recent history of each file. Mechanical, no model."""
    defs = changed_defs(repo, sha)
    files = sorted({f for f, _ in defs})
    lines = []
    if defs:
        lines.append("Changed definitions: " + ", ".join(f"{f}:{n}" for f, n in defs[:20]))
    names = sorted({n for _, n in defs if not n.startswith("_") and len(n) > 3})
    if names:
        pattern = "|".join(re.escape(n) for n in names[:12])
        try:
            callers = git(repo, "grep", "-n", "-E", rf"\b({pattern})\(", sha, "--", "*.py").splitlines()
        except subprocess.CalledProcessError:
            callers = []
        callers = [c.split(":", 1)[1] for c in callers if not any(c.split(":", 1)[1].startswith(f) for f in files)]
        if callers:
            lines.append("Callers elsewhere (file:line):\n  " + "\n  ".join(callers[:25]))
    mods = sorted({Path(f).with_suffix("").as_posix().replace("/", ".") for f in files})
    tests = []
    for m in mods:
        short = m.split(".")[-1]
        try:
            tests += git(repo, "grep", "-l", "-E", rf"(import|from)\s+\S*{re.escape(short)}\b", sha, "--", "tests", "test").splitlines()
        except subprocess.CalledProcessError:
            pass
    tests = sorted({t.split(":", 1)[1] for t in tests})
    if tests:
        lines.append("Tests importing the changed modules: " + ", ".join(tests[:15]))
    for f in files[:6]:
        hist = git(repo, "log", "--format=%h %ad %s", "--date=short", "-3", f"{sha}^", "--", f).strip()
        if hist:
            lines.append(f"Recent history of {f}:\n  " + hist.replace("\n", "\n  "))
    txt = "\n".join(lines)
    return txt[:max_chars]


def graph(repos: Path):
    positives = json.loads((WORK / "positives.json").read_text())
    negs_path = WORK / "negatives.verified.json"
    negatives = json.loads(negs_path.read_text()) if negs_path.exists() else json.loads((WORK / "negatives.json").read_text())
    base_for_repo = {p["repo_name"]: p["base_commit"] for p in positives}
    tasks = []
    for p in positives:
        repo = repos / p["repo_name"]
        sha = p["introducing"]
        lines, files = diff_stat(repo, sha)
        tasks.append({
            "id": f"bug__{p['repo_name']}__{sha[:7]}", "repo": p["repo_name"], "kind": "bug",
            "base_sha": git(repo, "rev-parse", f"{sha}^").strip(), "head_sha": sha,
            "diff": pr_diff(repo, sha), "files": files, "split": "fresh" if p["split"] != "own-older" else "own-older",
            "oracle": p["fail_to_pass"], "gold_hunks": gold_hunks(repo, sha, p["fix_commit"]),
            "neighborhood": neighborhood(repo, sha),
            "image": f"shwin/{p['repo_name']}:{p['base_commit'][:12]}", "test_cmd": p["environment"]["test_cmd"],
            "subdir": p["environment"].get("subdir") or ".", "fix_sha": p["fix_commit"],
        })
    for n in negatives:
        repo = repos / n["repo_name"]
        sha = n["sha"]
        lines, files = diff_stat(repo, sha)
        tasks.append({
            "id": f"clean__{n['repo_name']}__{sha[:7]}", "repo": n["repo_name"], "kind": "clean",
            "base_sha": git(repo, "rev-parse", f"{sha}^").strip(), "head_sha": sha,
            "diff": pr_diff(repo, sha), "files": files, "split": "fresh" if n["split"] != "own-older" else "own-older",
            "oracle": [], "gold_hunks": [], "neighborhood": neighborhood(repo, sha),
            "image": f"shwin/{n['repo_name']}:{base_for_repo[n['repo_name']][:12]}", "test_cmd": n["environment"]["test_cmd"],
            "subdir": n["environment"].get("subdir") or ".", "fix_sha": "",
        })
    out = Path(__file__).parent / "tasks.jsonl"
    out.write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
    print(f"{len(tasks)} tasks -> {out} ({sum(t['kind']=='bug' for t in tasks)} bug, {sum(t['kind']=='clean' for t in tasks)} clean)")


def gold_hunks(repo: Path, sha: str, fix_sha: str | None = None) -> list[dict]:
    """Where the defect lives in the introducing commit: its added-line ranges in non-test files,
    narrowed to the functions/classes the eventual fix commit modified (by file + def name, so line
    drift between the two commits doesn't matter). Falls back to all ranges if nothing intersects."""
    all_h = _added_ranges(repo, sha)
    if not fix_sha:
        return all_h
    fixed = set(changed_defs(repo, fix_sha))
    if not fixed:
        return all_h
    fixed_files = {f for f, _ in fixed}
    narrowed = []
    for h in all_h:
        if h["file"] not in fixed_files:
            continue
        defs_here = _defs_covering(repo, sha, h["file"], h["line_start"], h["line_end"])
        if defs_here & {n for f, n in fixed if f == h["file"]}:
            narrowed.append(h)
    return narrowed or [h for h in all_h if h["file"] in fixed_files] or all_h


def _defs_covering(repo: Path, sha: str, f: str, lo: int, hi: int) -> set[str]:
    import ast
    try:
        tree = ast.parse(git(repo, "show", f"{sha}:{f}"))
    except (subprocess.CalledProcessError, SyntaxError):
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            a, b = node.lineno, getattr(node, "end_lineno", node.lineno)
            if a <= hi and b >= lo:
                out.add(node.name)
    return out


def _added_ranges(repo: Path, sha: str) -> list[dict]:
    hunks = []
    cur = None
    for ln in git(repo, "show", "--format=", "--first-parent", "--unified=0", sha).splitlines():
        if ln.startswith("+++ b/"):
            cur = ln[6:]
        elif ln.startswith("@@") and cur and not is_test_file(cur):
            m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", ln)
            if m:
                s = int(m.group(1)); n = int(m.group(2) or 1)
                hunks.append({"file": cur, "line_start": s, "line_end": max(s, s + n - 1)})
    return hunks


# ---------- coverage (docker): changed lines no existing test executes ----------
def coverage(shwin: Path):
    """For every task in tasks.jsonl: run the suite under coverage at head_sha inside the task image,
    intersect executed lines with the diff's added lines, store `uncovered` = [{file, lines:[...]}]
    (added lines no test executed) and `covered_frac`. Network is enabled for this stage only, to
    pip install coverage inside the container."""
    import tempfile
    tasks_p = Path(__file__).parent / "tasks.jsonl"
    rows = [json.loads(l) for l in tasks_p.read_text().splitlines() if l.strip()]
    for t in rows:
        if not t.get("image"):
            continue
        subdir = t.get("subdir") or "."
        # added lines per file from the diff
        added: dict[str, set[int]] = {}
        cur = None
        for ln in t["diff"].splitlines():
            if ln.startswith("+++ b/"):
                cur = ln[6:]
            elif ln.startswith("@@") and cur and cur.endswith(".py") and not is_test_file(cur):
                m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", ln)
                if m:
                    s, n = int(m.group(1)), int(m.group(2) or 1)
                    added.setdefault(cur, set()).update(range(s, s + n))
        if not added:
            t["uncovered"], t["covered_frac"] = [], None
            continue
        script = (f"cd /testbed && git checkout -q {t['head_sha']} && pip -q install coverage >/dev/null 2>&1; "
                  f"cd {subdir} && python -m coverage run --branch -m pytest -q -x --no-header -p no:cacheprovider -o addopts= >/dev/null 2>&1; "
                  f"python -m coverage json -o /tmp/cov.json >/dev/null 2>&1; cat /tmp/cov.json")
        name = f"rbcov-{int(time.time() * 1000) % 10**9}"
        p = subprocess.run(["docker", "run", "--rm", "--name", name, "--network", "bridge", t["image"], "bash", "-lc", script],
                           capture_output=True, text=True, timeout=1800)
        try:
            cov = json.loads(p.stdout[p.stdout.index("{"):])
        except (ValueError, json.JSONDecodeError):
            print(f"{t['id']}: coverage failed: {p.stdout[-300:]} {p.stderr[-300:]}", flush=True)
            t["uncovered"], t["covered_frac"] = [], None
            continue
        files = cov.get("files", {})
        executed: dict[str, set[int]] = {}
        for fpath, fdata in files.items():
            rel = fpath
            for prefix in ("/testbed/", f"{subdir}/"):
                rel = rel[len(prefix):] if rel.startswith(prefix) else rel
            rel = (Path(subdir) / rel).as_posix() if subdir != "." and not rel.startswith(subdir) else rel
            executed[rel] = set(fdata.get("executed_lines", []))
        unc, n_added, n_cov = [], 0, 0
        for f, lines in added.items():
            ex = executed.get(f, set())
            missing = sorted(l for l in lines if l not in ex)
            n_added += len(lines); n_cov += len(lines) - len(missing)
            if missing:
                unc.append({"file": f, "lines": missing})
        t["uncovered"] = unc
        t["covered_frac"] = round(n_cov / n_added, 3) if n_added else None
        print(f"{t['id']:45s} added {n_added:4d} lines, {n_added - n_cov:4d} never executed by the suite ({t['covered_frac']})", flush=True)
    tasks_p.write_text("\n".join(json.dumps(t) for t in rows) + "\n")
    print(f"wrote {tasks_p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "bisect", "match", "verify", "graph", "coverage"])
    ap.add_argument("--shwin", default=str(Path.home() / "shwin-bench"))
    ap.add_argument("--repos", default=str(Path.home()), help="directory containing emberserve/, aether/, serverless-lakehouse/")
    a = ap.parse_args()
    shwin, repos = Path(a.shwin).expanduser(), Path(a.repos).expanduser()
    {"prepare": lambda: prepare(shwin, repos), "bisect": lambda: bisect(shwin), "match": lambda: match(repos),
     "verify": lambda: verify(shwin), "graph": lambda: graph(repos), "coverage": lambda: coverage(shwin)}[a.stage]()


if __name__ == "__main__":
    main()

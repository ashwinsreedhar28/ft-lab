"""Find the commit that introduced each validated bug (Mac stage, Docker).

    python -m review.mine.bisect [--repo R] [--jobs 3] [--max-ancestors 400]

Input: work/validated.jsonl rows with status=valid (from validate.py). For each, walk the first-parent
chain of the fix's parent; at a commit C, check out C, overlay the fix's test files (so the oracle
tests exist), run only the oracle tests. The oracle FAILS at the parent by construction. Binary-search
for the oldest commit where it still fails; the commit after the boundary is where the oracle first
passes, or first ERRORS (the test cannot import: the module did not exist yet, so the failing commit
created it and is the introducing commit, as in shwin-bench).

Assumes the oracle is monotone along the chain (fails from the introducing commit to the fix). Each
probed sha and its status are recorded so a non-monotone case is visible in the output.

Then two rules from the shwin-bench build:
  shared code file   the introducing commit must change at least one non-test .py file the fix also
                     changes; otherwise it is an import-boundary artifact (dropped, status=no_shared_file)
  size cap           introducing commits above --max-lines are recorded but flagged too_big

Output: work/positives.jsonl, one row per valid fix: introducing sha, boundary kind (pass|error),
probes, status in {positive, too_big, no_shared_file, older_than_window, non_monotone, docker_error}.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from review.bench.build import _parse_results
from review.mine.scan import git, is_test
from review.mine.validate import DEFAULT_CANDIDATE_INSTALL, WORK, docker, repos, run_tests_script


def oracle_status(name: str, sha: str, fix_sha: str, test_files: list[str], oracle: list[str],
                  inst: str, net: bool, timeout: int) -> tuple[str, str]:
    """pass | fail | error | docker_error for the oracle tests at `sha` with the fix's tests overlaid."""
    # run the whole test files (node ids need the file to be collectable), then read the oracle ids
    rc, out = docker(name, run_tests_script(sha, test_files, fix_sha, False, inst), timeout, net)
    if rc == 124 or "Unable to find image" in out or "INSTALL_FAIL" in out:
        return "docker_error", out[-800:]
    res = _parse_results(out)
    st = [res.get(t) for t in oracle]
    if all(s == "passed" for s in st):
        return "pass", out[-400:]
    if any(s == "failed" for s in st):
        return "fail", out[-400:]
    return "error", out[-800:]       # missing ids / collection error / import error


def code_files(repo: Path, sha: str) -> set[str]:
    out = git(repo, "show", "--name-only", "--format=", sha)
    return {f for f in out.split() if f.endswith(".py") and not is_test(f)}


def lines_changed(repo: Path, sha: str) -> int:
    n = 0
    for ln in git(repo, "show", "--numstat", "--format=", sha).splitlines():
        p = ln.split("\t")
        if len(p) == 3 and p[0].isdigit() and p[1].isdigit():
            n += int(p[0]) + int(p[1])
    return n


def search(n: int, status) -> tuple[str, int, int]:
    """Binary search over chain indices 0..n-1 (0 = fix's parent, must fail). Returns
    ("ok", lo, hi) with status(lo)=fail and status(hi) in {pass, error}, hi = lo+1; or an error verdict."""
    s0 = status(0)
    if s0 == "docker_error":
        return "docker_error", 0, 0
    if s0 != "fail":
        return "non_monotone", 0, 0
    s_old = status(n - 1)
    if s_old == "fail":
        return "older_than_window", 0, n - 1
    if s_old == "docker_error":
        return "docker_error", 0, n - 1
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        s = status(mid)
        if s == "fail":
            lo = mid
        elif s == "docker_error":
            return "docker_error", lo, hi
        else:
            hi = mid
    return "ok", lo, hi


def unshallow(repo: Path):
    """scan.py clones --shallow-since; mature repos' bugs are older than that window."""
    if not (repo / ".git" / "shallow").exists():
        return
    print(f"{repo.name:10s} unshallowing clone …", flush=True)
    import subprocess
    for args in (["fetch", "-q", "--unshallow"], ["fetch", "-q", "--depth=1000000"]):
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        if r.returncode == 0 and not (repo / ".git" / "shallow").exists():
            return
        print(f"{repo.name:10s} {' '.join(args)}: {r.stderr.strip()[-200:]}", flush=True)
    n = len(git(repo, "rev-list", "--count", "HEAD").split()) and git(repo, "rev-list", "--count", "HEAD").strip()
    print(f"{repo.name:10s} still shallow ({n} commits reachable); bisect window limited to that", flush=True)


def bisect_one(c: dict, cfg: dict, max_anc: int, max_lines: int, timeout: int) -> dict:
    name = c["repo"]; repo = WORK / "repos" / name
    r = cfg[name]
    inst, net = r.get("candidate_install", DEFAULT_CANDIDATE_INSTALL), bool(r.get("network"))
    chain = git(repo, "rev-list", "--first-parent", f"--max-count={max_anc}", c["parent"]).split()
    probes = {}

    def status(i: int) -> str:
        sha = chain[i]
        if sha not in probes:
            probes[sha] = oracle_status(name, sha, c["sha"], c["test_files"], c["oracle_tests"], inst, net, timeout)[0]
        return probes[sha]

    row = {**{k: c[k] for k in ("repo", "sha", "parent", "subject", "lines", "test_files", "code_files", "oracle_tests")},
           "chain_len": len(chain)}
    verdict, lo, hi = search(len(chain), status)
    if verdict != "ok":
        row.update(status=verdict, probes=probes, note=f"parent={probes.get(chain[0])}")
        return row
    intro = chain[lo]
    row.update(introducing=intro, boundary=status(hi), introducing_parent=chain[hi], probes=probes,
               bug_age_commits=lo + 1,   # first-parent commits from introduction to the fix
               introducing_lines=lines_changed(repo, intro),
               introducing_subject=git(repo, "log", "-1", "--format=%s", intro).strip(),
               shared_code_files=sorted(code_files(repo, intro) & set(c["code_files"])))
    if not row["shared_code_files"]:
        row["status"] = "no_shared_file"
    elif row["introducing_lines"] > max_lines:
        row["status"] = "too_big"
    else:
        row["status"] = "positive"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo"); ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--max-ancestors", type=int, default=5000); ap.add_argument("--max-lines", type=int, default=1500)
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    cfg = repos()
    valid = [json.loads(l) for l in (WORK / "validated.jsonl").read_text().splitlines() if l.strip()]
    valid = [v for v in valid if v["status"] == "valid" and (not a.repo or v["repo"] == a.repo)]
    out_p = WORK / "positives.jsonl"
    done = {}
    if out_p.exists():
        for l in out_p.read_text().splitlines():
            if l.strip():
                d = json.loads(l); done[(d["repo"], d["sha"])] = d
    todo = [v for v in valid if (v["repo"], v["sha"]) not in done]
    for name in sorted({v["repo"] for v in todo}):
        unshallow(WORK / "repos" / name)
    print(f"{len(valid)} valid fixes, {len(todo)} to bisect, {a.jobs} parallel")
    with out_p.open("a") as fh, ThreadPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(bisect_one, v, cfg, a.max_ancestors, a.max_lines, a.timeout): v for v in todo}
        for f in as_completed(futs):
            row = f.result()
            fh.write(json.dumps(row) + "\n"); fh.flush(); done[(row["repo"], row["sha"])] = row
            intro = row.get("introducing", "")[:10]
            print(f"{row['repo']:10s} {row['sha'][:10]} -> {intro:10s} {row['status']:17s} probes={len(row['probes']):2d} "
                  f"boundary={row.get('boundary', '-'):5s} lines={row.get('introducing_lines', 0):5d}  {row.get('introducing_subject', '')[:50]}", flush=True)
    from collections import Counter
    cnt = Counter(r["status"] for r in done.values())
    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(cnt.items())))
    pos = [r for r in done.values() if r["status"] == "positive"]
    if pos:
        import statistics
        print(f"positives: {len(pos)}, introducing-commit size median {statistics.median(r['introducing_lines'] for r in pos):.0f} lines, "
              f"max {max(r['introducing_lines'] for r in pos)}; error-boundary {sum(r['boundary']=='error' for r in pos)}; "
              f"bug age median {statistics.median(r['bug_age_commits'] for r in pos):.0f} commits")
    print("next: python -m review.mine.assemble (match negatives, graph, coverage -> review/bench/public_tasks.jsonl)")


if __name__ == "__main__":
    main()

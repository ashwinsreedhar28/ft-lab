# Mining fresh bug-introducing PRs from public repos (phase 3 training data)

Why public repos: shwin-bench gives 42 positives, enough to design the scorer and bound
"bugs approved" at ~7 %. Training a decision model needs hundreds, and a reliability claim
tighter than 1-in-14 needs hundreds in eval too. Public repos supply volume; a date filter
keeps them fresh.

## Recipe (generalizes shwin-bench's tools/mine.py)
1. **Candidate repos.** Python, pytest, CI green on main, active after 2026-06-01, permissive
   license, suite runs on CPU in < 5 min inside a slim Docker image. Start list: ~30 repos
   from PyPI top downloads and GitHub trending filtered by those rules; drop any whose
   suite needs network or GPUs.
2. **Fix commits.** `git log --since=2026-06-01` for commits that (a) touch at least one
   test file and one non-test file, (b) whose message or linked PR mentions fix/bug/regress/
   crash, (c) are not merges or reverts. Expected yield: 5–15 % of commits.
3. **Validate the test as oracle** (shwin-bench's rule): the commit's new/changed tests
   fail at `parent` 3/3 and pass at `commit` 3/3, and the full suite is green at `commit`.
   Keep the failing set as FAIL_TO_PASS.
4. **Bisect to the introducing commit.** Walk back from `parent` applying the test files
   from `commit` at each ancestor; the first ancestor where FAIL_TO_PASS passes is the
   last-good commit; its child is the **introducing commit** `C_bug`. Stop if the tests
   can't import (interface predates the test) or after 50 ancestors; those tasks are dropped.
   `C_bug`'s diff is the positive PR; `gold_hunks` = the lines of `C_bug`'s diff that
   overlap the fix's hunks (file:line), falling back to all of `C_bug`'s hunks.
5. **Matched negative.** From the same repo, a non-merge commit after 2026-06-01 with the
   suite green at both parent and commit, diff size within ±30 % of `C_bug`'s, preferring
   file-set overlap, not itself a fix commit. One per positive.
6. **Neighborhood.** For both: callers/callees of changed functions (tree-sitter or jedi),
   tests that import the changed modules (coverage.py over the suite once per repo),
   last 3 commits touching each changed file, CODEOWNERS if present. Serialized to text
   for the `diff+graph` condition and stored in the task row.
7. **Split by repo**, not by task: eval repos never appear in train. Target 400 train /
   100 eval positives plus matched negatives.

## Cost
CPU, Docker, and time. One repo ≈ 30–90 min of machine time for steps 2–6 on a laptop or a
cheap CPU pod ($0.10–0.20/h on Runpod). Budget ~20 repo-hours → ~$4 of pod time or a
weekend of the Mac. No GPU until the GRPO run.

## Leakage checklist (why each step exists)
- reverse(fix) shape → **bisect**: positives are the diffs that actually shipped the bug
- positives smaller than negatives → **size match**
- positives in bug-prone files only → **file-set preference** for negatives
- model has seen the repo's history → **date filter** (fresh commits), and report the
  own-older vs fresh split like shwin-bench does
- same repo in train and eval → **split by repo**
- test file in the PR gives the answer away → positives and negatives both include their
  test changes as they were; the oracle test from the *fix* is never shown

## Selection bias to remember
`scan.py` keeps commits whose subject matches a fix vocabulary and that touch both a test and a
code file, ≤400 lines. That selects for small bugs the author recognized and tested as bugs. The
bench will under-represent silent regressions, bugs fixed without a new test, and bugs fixed inside
feature commits. Fine for a first bench of real positives; say so in any write-up, and do not treat
a rate on it as a rate on "PRs in general".

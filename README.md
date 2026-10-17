# ft-lab

GRPO fine-tuning experiments on Runpod GPUs: one script, swappable tasks, measured
against a frozen held-out split and frontier-model baselines.

End goal: a code review agent whose approval is the gate to prod, built eval-first on my
own repos (shwin-bench), then a small model fine-tuned into the roles where it holds the number.

| phase | what | signal | status |
|---|---|---|---|
| 0 | GRPO mechanics: GSM8K, Qwen3-1.7B, 1×H100 | answer match | done Oct 17; lessons below |
| 1 | review-bench from shwin-bench: bisected bug PRs + matched clean PRs, 4 context conditions, frontier baselines | bugs approved (target 0), auto-ship rate | v0 built: 4 bug + 4 clean; see below |
| 2 | the review agent: neighborhood → run suite → probe test → decision | same scorer | v1–v4 run on the 8-PR bench; frozen until the bench is bigger |
| 3a | public bench: mined fix commits → validate → bisect (real positives, eval); mutation-injected bugs on clean PRs with a probe guard (training volume) | same scorer | validate+bisect run on 8 repos; mutate written |
| 3b | SFT warm-start on frontier review traces (TEMP-selected); GRPO a 4B/8B LoRA into the triage gate with variance-guided rollouts (VIGOR) | asymmetric decision reward | recipe written; waits on 3a |
| 4 | topology: 1-colocate / 2-server / 4-colocate | $/1k completions | configs ready |
| backlog | Spider / BIRD / lakehouse text-to-SQL | execution match | adapters kept |

```
pip install -r requirements.txt          # or bash pod/setup.sh on a pod
make test                                # 27 pure-python tests
python -m ftlab.train configs/gsm8k_h100.yaml
python -m ftlab.eval  configs/gsm8k_h100.yaml --label tuned --model runs/gsm8k_h100/final
python -m ftlab.plot  runs/gsm8k_h100
```

Layout: `ftlab/config.py` (YAML → dataclass, `extends:` chains), `ftlab/data/*`
(task adapters: load + `score`), `ftlab/rewards/*` (registry of reward functions),
`ftlab/train.py` (TRL `GRPOTrainer`, vLLM colocate/server, optional LoRA),
`ftlab/eval.py` (greedy vLLM or Anthropic backend, Wilson CI), `ftlab/plot.py`.
Operational steps in `RUNBOOK.md`; design notes and a GRPO primer in `docs/NOTES.md`.

## Phase 0: what the GSM8K run taught (Oct 17, 2026, 1× H100, ~$3)

Not a results table; the accuracy numbers below are not a claim. Qwen3-1.7B, thinking off, full
fine-tune, GRPO 300 steps × 64 completions (4 prompts × 16), lr 1e-6, β 0.04, 2.9 s/step.

- **The harness works end to end** on a real pod: TRL 1.16 + vLLM 0.31 colocated, self-driving pod
  with a results branch, eval with Wilson CIs. Seven script fixes on first contact; now stable.
- **The reward curve was flat** (correct ≈ 0.85 at T=1 throughout, KL < 0.01). The base model is at
  79 % on GSM8K, so a quarter to three quarters of each step's prompts had zero within-group
  variance and contributed no gradient. A task the base already does is the wrong place to look for
  a curve; the hard-pool config and variance-guided rollout allocation (VIGOR) exist for this reason.
- **The policy bought reward with length.** Under a 768-token decode budget the tuned model scored
  75.4 % (base 79.0) because 211 completions hit the cap. At 1,536 tokens it scored 84.7 %
  (CI 82.6–86.5) at 5.4× the output tokens of the base, bimodally: completions under 2k chars are
  96–98 % right, those over 4k chars 32 %. It also broke 79 answers the base had right by looping.
  +5.7 points at 5× decode cost loses to serving a bigger model at 1×. The lesson for every later
  reward: price tokens and abstentions in the reward, or the policy pays for reward with them.
- **LoRA (r=32) matched full fine-tune** within CI (76.2 vs 75.4 at 768 tokens). One task, one seed;
  weak support for the one-base-many-adapters pattern, not proof.
- **Eval protocol is part of the method.** Decode budget and temperature must be fixed before
  training, not chosen after seeing which setting looks best (this run chose 1,536 after the fact;
  the hard-pool run reports at both budgets, decided in advance; it died at step 150 when a second
  17 GB full-FT checkpoint filled the disk, so `save_total_limit: 1` now). Report tokens per answer next to
  accuracy. One seed is an anecdote; `eval.py --table` collapses `<label>_s<seed>` for a reason.
- **What this changes downstream:** model size for the review agent is chosen by $/PR at the target
  reliability (start at 4B/8B), not inherited from here; the review reward gets a token-cost term;
  the 1.7B stays as the smoke-test model.

Raw artifacts: `runs/gsm8k_h100/{eval_*.json, curves.png, log_history.json}` on branch
`results/phase0` and `results/phase0b`.

## Phases 1–2: review-bench v0 and the first harness (Oct 17, 2026, Mac + Docker + API, ~$15)

**Bench.** From shwin-bench's 15 fix commits: 4 were entirely interface-leak tasks (the hidden test
imports a name the fix adds, so it errors at every pre-fix commit; no bisect signal). Per-test bisect
over the full history found the commit where each oracle test first fails; a fail→error boundary
(test can't import any earlier) means the bug was born with the module, and that creating commit is
the PR. Most of emberserve's bugs trace to its 4,427-line initial dump and aether's to a 57k-line
one; dropped. One positive (aether `f0a823d`) was an import-boundary artifact: the test's *imports*
became satisfiable there, but the commit touched none of the files the fix touched; every reviewer
approved it and was right to. Rule added: an introducing commit must share a code file with its fix.
Result: **4 bug PRs** (285–1,155 changed lines; two are themselves fix commits that introduced a
different bug) and **4 size/file-matched clean PRs**, all suite-green at parent and commit. Gold
hunks are narrowed to the functions the fix touched. 0/4 bugs approved bounds the rate at ~1 in 2
(rule of three), so everything below is a design validation, not a reliability claim.

**Diff-only reviewers ship bugs.** Same 5+5 PRs (before the artifact was removed), single call,
adaptive thinking: Sonnet 5.5 approved 3/5 bug PRs reading the diff, 3/5 with the two-hop
neighborhood added; Haiku 5.5 approved 2/5 and 1/5. When they blocked, they blocked the right
hunk (localization 1.0). The frontier model's failure is confidence, not comprehension.

**A green suite made it worse.** Harness v1 (suite result at the PR head + optional probe test):
Sonnet approved 3/5 bugs and auto-shipped 60 % of clean PRs; probes were written for 1 PR in 10
and were trivial (`test_flags_exist`). The suite at a bug-introducing commit is green by
construction (the catching test doesn't exist yet), which is the state of every bug that ever
shipped, and the reviewer read "tests pass" as evidence.

**Harness v2: evidence rules.** The suite is presented as "pre-existing tests still pass, which says
nothing about the change"; changed definitions with no referencing test are listed; an approve is
accepted only with a passing probe that calls a changed function by name, otherwise it is bounced
for one more round and then escalated. First run of Sonnet on the 8-PR bench: 0/4 bugs approved,
2/4 clean auto-shipped. **Three repeats say that was the lucky draw**: Sonnet v2 approved 2/4, 1/4,
0/4 (mean 0.25 ± 0.25), auto-ship 0.17 ± 0.12, escalate 0.75 ± 0.12, $0.10/PR; Haiku v2 1/4, 0/4,
0/4 with escalation 0.83 ± 0.25 and almost nothing shipped. Versus diff-only Sonnet at 3/5, v2 lowers
bugs-approved, mostly by pushing decisions toward escalate. The strong-probe rate (probe fails on
the bug PR, passes on the fix) is **0.00 across all 24 runs**: not one probe found a bug. The probes
that licensed approvals called a changed function by name without testing the behavior that was
wrong, which is the gaming hole in a syntactic evidence rule. Two lessons that survive the noise:
run-to-run variance at n=8 is ±25 points, so nothing here is a number until the bench is 5× larger;
and the harness lever that matters next is probe quality (falsifiable hypotheses), not more rules
about when to approve. This is the "harness engineering" loop: the scoreboard says what the agent
needs to see, what it is forbidden to do, and what it is still not doing.

**Harness v3 (probe as hypothesis) and v4 (uncovered-lines cue).** v3 made the agent state what
would be wrong before writing the probe: Sonnet 1/4 bugs approved in all three repeats, Haiku 0/4
×3, strong-probe rate still 0.00. v4 added a coverage stage to the bench (run the suite under
`coverage` at the PR head; list the changed lines no test executes, e.g. 131 of the lines in
`36f4e93` including 21 of its 54 gold lines) and required an approve-probe to execute one of them.
Sonnet v4, three repeats: **0/4 bugs approved every time**, auto-ship 0.42 ± 0.12, escalate
0.79 ± 0.06, $0.12/PR, and no probe ran on any bug PR. The post-run audit split that last fact in two.
In 10 of 12 bug-PR runs the model escalated in round 0 without asking for evidence: the number is
reached by abstaining, and the agent spends probes only where it believes approval is reachable (the
clean PRs). In the other 2 (both `36f4e93`) the model *did* write a probe, with a hypothesis, but
emitted it inside hallucinated `<invoke>` tool-call syntax; the harness's JSON regex missed it and
scored the review unparsable → escalate. Replaying the fixed parser over the stored replies recovers
17/17; 4 probes (2 bug, 2 clean) were dropped in all. The result record now carries per-round
decisions, a `no_probe_reason`, and API vs Docker wall-clock, so this question is answerable from the
JSON next time instead of by reading raw text.

**v4b (parser fixed, rerun ×3): the approve rule is satisfiable without testing the bug.** Sonnet
approved `36f4e93` in 3/3 runs, each licensed by one passing probe: it calls a changed function by
name and executes one of the PR's 131 uncovered lines, so it meets every rule, and it tests a
hypothesis (timestamp clock bases) unrelated to the defect (token counting in `load.py`). The other
three bug PRs were escalated in round 0 in 8 of 9 runs. v4b: 0.25 bugs approved, auto-ship
0.33 ± 0.12, escalate 0.67 ± 0.06, $0.14/PR. In v4 the parse drop was accidentally protecting the
number. An always-escalate agent scores 0 bugs approved at 0 auto-ship; no harness version here
beats that bound while shipping anything.
Across v1–v4b, 48 bug-PR runs, no
probe has ever failed on a bug and passed on its fix; every approve of a bug PR was licensed by a probe
that passed because it tested the wrong hypothesis. Syntactic evidence rules (names a changed def,
touches an uncovered line) bound confidence, not correctness. The reliability number this project is after
cannot be read off a 4-positive bench (rule of three: 0/4 bounds the rate at ~50 %), and the
harness is now out of levers that rules can pull; the lever left is detection, which needs more
and smaller positives. Harness work stops here until the bench is 5–10× larger.

Raw: `review/bench/tasks.jsonl`, `review/agent/results/` (v1 runs under `results/v1/`).

## Phase 3a: what mining public repos taught (Oct 17, 2026, Mac + Docker)

`review/mine/`: scan → validate → bisect, each a `make` target, details in `review/mine/README.md`.
39 fix-commit candidates since June 2026 from 8 repos → **24 validated** (the fix's tests fail at the
parent, pass at the fix; 62 %, every failure was an install line) → bisected to the introducing commit
with a galloping search along first-parent history (Θ(log age) container runs) → **2 positives**.
The other 22: 4 introducing commits over 1,500 lines (feature commits: SSE support, a router refactor,
"drop Python 3.9" at 10,708 lines), 7 import-boundary artifacts (the overlaid test can't run before a
pytest-9 upgrade or a warning removal), 1 bug that arrived in a branch merge, and 10 where the
oracle fails all the way back to the repo's first commit because the test asserts behavior that did
not exist before the feature. Zero of 39 fix messages mentioned a regression.

**The finding:** in a mature repo a test-covered fix almost always fixes a bug born with its feature,
and the feature PR is 200–10,000 lines. That is what shwin-bench showed on young repos too. The small
bug-introducing PR the gate most needs to catch is a *regression* (test passed → broke), 1 of 24 here
(fastapi, 1,239 lines, 592 commits old). Real small positives are rare by the fix-commit route; a
regression-targeted scan over many more repos is the eval route, and training volume has to be
synthetic: `review/mine/mutate.py` injects one bug into a line a clean PR changed, keeps it only if the
suite still passes **and** a Haiku-written probe fails on the mutant and passes on the original
(equivalent-mutant guard; the probe doubles as the strong-localization oracle), tags each mutant
`op` or `llm` so operator-shape learning shows up as a gap between the two. Eval stays real.

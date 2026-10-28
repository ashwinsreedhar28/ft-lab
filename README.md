# ft-lab

GRPO fine-tuning experiments on Runpod GPUs: one script, swappable tasks, measured
against a frozen held-out split and frontier-model baselines.

End goal: a code review agent whose approval is the gate to prod, built eval-first on my
own repos (shwin-bench), then a small model fine-tuned into the roles where it holds the number.

| phase | what | signal | status |
|---|---|---|---|
| 0 | GRPO mechanics: GSM8K, Qwen3-1.7B, 1×H100 | answer match | done Oct 17; lessons below |
| 1 | review-bench from shwin-bench: bisected bug PRs + matched clean PRs, 4 context conditions, frontier baselines | bugs approved (target 0), auto-ship rate | v0 built: 4 bug + 4 clean; see below |
| 2 | the review agent: neighborhood → run suite → probe test → decision | same scorer | frozen at harness v5 (suite delta); Sonnet 2/35, Haiku 0/35 bugs approved on the bench v4 eval (below) |
| 3a | public bench: mined fix commits → validate → bisect (real positives, eval); mutation-injected bugs on clean PRs with a probe guard (training volume) | same scorer | bench v4 frozen Oct 20: 189 tasks (100 mutants + 2 real + 87 clean), eval 61 / train 128; 9 % label noise on hand review |
| 3b | SFT warm-start on frontier review traces (TEMP-selected); GRPO a 4B/8B LoRA into the triage gate with variance-guided rollouts (VIGOR) | asymmetric decision reward | run Oct 20–25 (~$46): no GRPO run beat its start. 4B: approve-all, then block-all once advantages were unscaled; 8B thinking (v5): null, kl 0.002. Single-seed eval noise ±2–4 paired (below) |
| 5 | harness v6: gate model × review tools (static, test-suite delta, AI reviewer) with evidence policies | same scorer + per-tool hit rates | wk1 tools layer Oct 28 (below) |
| 4 | topology: 1-colocate / 2-server / 4-colocate | $/1k completions | configs ready |
| backlog | Spider / BIRD / lakehouse text-to-SQL | execution match | adapters kept |

```
pip install -r requirements.txt          # or bash pod/setup.sh on a pod
make test                                # 69 pure-python tests
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

**Mutation yield (258 clean PRs, Jan–Oct 2026, 5 repos; Oct 17–18):** ~3 mutants per PR; 190 killed by
the suite (stage 1 on tests importing the module, stage 2 on the full suite, both against a baseline,
since `filterwarnings = error` and tty-only tests make "all green" the wrong criterion); 30 suite
survivors dropped by the probe guard (no distinguishing probe in 2 tries: equivalent or untestable);
**56 positives**, 38 operator / 18 Haiku-written, each with a verified probe. 92 PRs produced nothing
(environment: mostly pydantic baselines, under diagnosis). `assemble.py` makes each a real commit and
pairs it with its clean original: **97 tasks** (58 bug incl. 2 real, 39 clean), split 80/20 by PR
hash because click supplies 81 of them; the repo-level split returns when ≥3 repos contribute.

**First frozen-harness numbers on the public eval (34 tasks: 12 op + 8 llm + 2 real bugs, 12 clean).**
Haiku: 0/22 bugs approved, auto-ship 0.25, escalate 0.41, but it blocked 5 of 12 clean PRs
(block precision 0.76). Sonnet: **7/22 bugs approved (32 %)**, auto-ship 0.58, escalate 0.09, strong-
probe 0.00; by source 42 % of operator mutants, 12 % of LLM mutants, 1 of the 2 real positives. All
seven approvals were probe-licensed: Sonnet wrote a test, it passed, it named a changed function and
executed an uncovered changed line, and it did not exercise the mutated line. The `36f4e93` hole from
the 4-positive bench, now on 22 positives: a passing test that satisfies syntactic evidence rules
bounds the reviewer's confidence, not the change's correctness. The two frontier models fail in
opposite directions, Sonnet by shipping and Haiku by blocking, and neither has found a bug with a probe
in 70 bug-PR runs across both benches.

**Leakage probe** (`make leakage`): a bag-of-tokens logistic regression on added diff lines, 5-fold CV
grouped by PR so a mutant and its original never split across folds. At n=97: AUROC 0.56. At n=415
it found the operator signature: 0.68 on operator mutants (`>=`, `or`, `!=` top-weighted), 0.61 on LLM
mutants, 0.56 overall. Two fixes tried. Unpaired token-matched clean decoys made it *worse* (LLM 0.65):
the paired design is the defense, and unpaired PRs add whole-document content confounds. Direction
balance worked: equalizing each symmetric operator pair in train (`and→or` vs `or→and`, …) makes the
introduced token appear in bug lines equally often either way; operator AUROC 0.62, overall 0.53, at
the cost of 34 train mutants. What remains is the mutated line's meaning, which a reviewer is supposed
to notice. The post-training op-vs-llm gap is the control for anything a linear probe cannot see.

**Label check (Oct 20).** A 22-item sheet of mutants reviewed by hand (would you block this line?)
found **66 % label noise** on the first bench: docstring and comment lines mutated (`>>>` → `>=>>`),
`__version__` strings, two equivalent mutants. The mechanical fix (never mutate docstrings, comments,
doctest prompts, version/import/logging lines) removed **140 of 240 mutants**, and the operator
"leak" went with them (AUROC 0.48 / 0.53 / 0.52). A second 22-item sheet on the cleaned bench,
AI-suggested and human-confirmed (21/22 agreement, one override): **9 % noise**. One caveat on
record: a real bisected regression was confirmed "no code changed" by the human following a wrong
AI note; the executable oracle wins for scoring, and that case is the known failure mode of
AI-assisted labeling.

**Bench v4 (Oct 20): 189 tasks.** 100 behavioral mutants (36 op / 62 llm after direction balance)
+ 2 real + clean originals. Eval 61 (35 bugs: 13 op, 20 llm, 2 real; 26 clean), train 128.

**Frozen harness on the cleaned eval (re-scored from the pub2 runs):**

| model | bugs approved | auto-ship | escalate | block precision | strong probe | $/PR |
|---|---|---|---|---|---|---|
| Sonnet 5.5 | **1/35** (op 0/13, llm 1/20, real 0/2) | 0.46 | 0.23 | 0.91 | 0.00 | 0.030 |
| Haiku 5.5 | 2/35 (op 1/13, llm 0/20, real 1/2) | 0.27 | 0.34 | 0.87 | 0.00 | 0.016 |

The earlier "Sonnet approved 32 %" was label noise: the approved items were docstring and version
mutants, and approving them was right. On single-line *behavioral* bugs the frontier harness blocks
85–92 % and ships 1 in 35, which bounds the rate at roughly 3–15 % at this size. Two things did not
change: no probe has reproduced an oracle in any run on either bench, so the blocks come from reading
the diff, not testing it; and the own-repo `36f4e93` case (probe-licensed approval of a 447-line
PR) stands. Single-line mutants are the easy end of the distribution; the real-PR end is where the
gate still fails, and that is the next bench to grow.

**Harness v5 (Oct 20): the agent sees the suite delta.** v4b showed the agent raw suite output at head, and
5 of 7 false blocks were environment failures already present at the parent. v5 runs junit at both commits
and reports only new failures, fixed tests, and tests the PR added (flagging any that fail at head). Same
eval: Sonnet 2/35 bugs approved, auto-ship 0.38, escalate 0.34; Haiku 0/35, auto-ship 0.26, escalate 0.44.
Fewer false blocks, more escalation; the remaining blocks on clean PRs are speculation the delta refutes,
and one candidate (`clean__click__c69643b6`) both models flag. Eval frozen here; the harness is not
iterated further on 61 tasks.

## Phase 3b: SFT + GRPO into the triage gate (Oct 20–21, 2026, 1× A100 80 GB, ~$22)

**Student role.** No tools: diff + context graph in, `{"decision", "comments"}` out. The teacher is Sonnet
under harness v5 on the 128 train tasks (`review/analysis/traces.py`): 79 usable traces, 54 bug→block,
25 clean→approve; 29 clean→escalate dropped because the gate must not learn to punt. Reward
(`ftlab/rewards/review.py`): approve-a-bug −5, correct +1, block-a-clean −0.5, escalate −0.2, localized
block +0.5, unparsable −1, −0.1 per 256 tokens past 300. Pipeline: `pod/phase3.sh` (base eval → SFT →
eval → GRPO → eval), results every 5 min to `results/phase3`.

**Two runs, both on the 61-task eval (35 bugs / 26 clean):**

| run | model | acc | bugs → approve / block | clean → approve / block | note |
|---|---|---|---|---|---|
| — | Qwen3-4B base | 0.475 | 32 / 3 | 26 / 0 | approves 58/61 |
| v1 | SFT, 54:25 traces | 0.230 | 1 / 34 | 0 / 26 | blocks 60/61; 20 outputs fail strict parse |
| v1 | + GRPO 120 steps, lr 1e-5, β 0.04 | 0.393 | 1 / 34 | 0 / 26 | same decisions; gain is JSON validity. kl 0.001 |
| v2 | SFT, 25:25 balanced | 0.311 | 4 / 31 | 5 / 21 | first non-degenerate prior |
| v2 | + GRPO 300 steps, lr 5e-5, β 0.01, T 1.0 | 0.426 | 35 / 0 | 26 / 0 | approves 61/61; 18 drifted to single-quoted dicts. kl 0.14 |

**What the two runs say.** The gate's decision prior comes from SFT and GRPO has not beaten it. v1 started
from a block-everything policy; every group of 8 samples then carried the same reward, the advantage was
zero, and 120 steps moved nothing (kl 0.001). v2 started from a policy that varied, and the reward signal
was visibly correct in training (`review/approve_bug_frac` logged every step; the −5s show up exactly
where it rises), but the policy slid toward approve anyway: 0.09 → 0.3–0.5 approve-on-bug over 300
steps, unparsable 0.2–0.56 throughout, mean reward negative in most steps (10 of 300 positive, max +0.79). The reward is sparse in the one
direction that matters: switching the decision token and writing a localized comment is a two-part
move, block samples get rare once the policy tilts, and within a group of 8 the comparison is mostly
approve-vs-approve. At lr 5e-5 with β 0.01 nothing pulled it back; the single-quote drift (−1 each)
is the same failure in miniature. Importance-sampling ratio mostly 0.9–1.08 but dipping to 0.53 around step 62, so the sampler is not
fully excluded as a contributor; the decision drift is the policy's.

**Infra lessons (one pod-day lost):** the Runpod template now exports `PYTHONPATH` to its dist-packages,
which outranks a venv; `pip --ignore-installed` leaves stale `*.dist-info` so importlib.metadata reports
versions the module isn't; `trl[vllm]` pulled vllm 0.32.0 over the 0.31.0 pin; flashinfer 0.7.2 broke
against the cutlass-dsl vllm pins. `pod/setup.sh` now does one pinned resolve and force-reinstalls
vllm's pins. Qwen3 detection by path left thinking on for a model at `runs/<name>/final` (53/118
truncated); detect by chat template. GRPO does not truncate prompts: a 36,993-token PR killed a run.

**v3 plan:** eval every 50 steps; 16 generations per prompt; lr 2e-5, β 0.02; reward with unparsable −3 and
block-on-bug +1 before localization so the correct move is reachable in one step; SWE-smith instances as
additional positives once the recipe moves the needle on 61.

**v3 and v4 (Oct 21, 1× H100, ~$12).** v3 fixed the optimizer: `scale_rewards: group` had been dividing each
group's advantages by its own std, so a bug group (−5 vs +1) and a clean group (+1 vs −0.5) both pushed with
|adv| ≈ 1 and the −5 never reached the gradient — v3a (group, lr 2e-5) collapsed to an 11-token
approve-everything by step 18. With `scale_rewards: none` and lr 1e-5, v3b held its format and moved both ways
at step 25 (bugs 29 blocked / 6 approved, clean 9 approved / 17 blocked) then settled into block-all by step
50 and stayed there to 200 (eval every 25 steps, `runs/review_grpo_4b_v3/eval_curve.jsonl`). Block-all is the
reward's rational optimum for a policy that cannot discriminate, and the SFT students never could:
P(block | bug) 0.89 vs P(block | clean) 0.81. The suite delta cannot help either — mutants are selected to
survive the suite, so it is empty on every positive by construction.

v4 tried thinking mode. Untrained Qwen3-4B with thinking on, decoded at T 0.6 (greedy loops in thinking mode,
per Qwen's own docs): bugs 9 blocked / 22 approved / 4 truncated, clean 2 blocked / 23 approved —
block *intent* 9/35 vs 2/26; under the strict parser (valid JSON only) 4/35 vs 2/26 — bug approvals
32 → 22 with clean approvals 26 → 23 is the operationally real change (Oct 25 audit correction). SFT on 50
Sonnet rationales (balanced; `review/analysis/rationales.py`, 42 of 128 tasks refused by the API) made it
worse on every column (5 / 20 / 9 and 3 / 15 / 8; 18 never close the think block). GRPO was not run.

**Oct 25 audit (external).** Decision counts above for v1–v4 came from a regex that accepted malformed JSON; the
scorer's parser does not. Strict recounts, bench caveats and the pre-registered measurement job that now
precedes any further training are in `docs/phase3b/MEASURE.md`; artifacts in `docs/phase3b/`.

**Measurement job and v5 (Oct 25, ~$12).** After the audit, an evaluation-only job (3 prompt/thinking arms × 3 seeds,
strict parser, matched decode) found: the compact no-test prompt halves invalid outputs; thinking cuts bug approvals
by 5 on the 4B; the 4B's paired discrimination ceiling is ≈ 6/26 however prompted; **Qwen3-8B, thinking, compact
prompt, 4k cap: bug approve 17.7/35, valid blocks 17/35, clean approve 22.3/26, 0 invalid, paired 9.8 ±1.4** — every
gain over the 2k cap is the 22 truncated outputs finishing. GRPO v5 from that start (50 steps, lr 1e-5, raw
advantages, reward v5, eval every 10 with step 0): kl 0.002, no movement; and the same untrained weights score
paired 9.0 vs 13.5 across two seed-0 evals, so a single-seed 61-task eval cannot resolve the effects being chased.
Details and the pre-registered rules: `docs/phase3b/MEASURE.md`.

**v6 (Oct 27, ~$10): null.** lr 3e-5, 3-seed checkpoints, 223 train rows (eval frozen verbatim), clip-higher. Bug approvals
15–18 and clean approvals 20–23 across all seven checkpoints, kl ≤ 0.007. Six GRPO runs, each diagnosed, none beat its
start. Direction after the Oct 27 check-in: a review **harness** (model × the repo's review tool) evaluated on the same
benches, with fine-tuning as a lever inside it — `docs/phase3b/MEASURE.md` has the v6 curve and the plan pointer.

## Harness v6: gate model × review tools (Oct 28–, Mac + Docker + API, no GPU)

Six GRPO runs left the gate where they found it; prompt, decode settings, token cap and model size are what moved it.
Production repos usually already run review tooling (linters, type checkers, CI, an AI reviewer), so the question
becomes the harness: which model, reading which tool output, under which rules, makes a gate that never approves a
bug and still ships clean PRs. Fine-tuning stays a lever inside the harness, used only if the gate model is the
bottleneck once the tools are in.

- `review/tools/`: every tool emits `Finding(tool, severity, file, lines, message, rule)`, so a real repo's tool is one
  more adapter. Three stand-ins:
  - `static`: ruff (`F,E9,B,PLE`, isolated config) + pyright (basic, missing-import rules off). Keeps only findings
    on lines the PR added that are not already reported at the parent. Runs in `<image>-static`.
  - `tests`: suite delta vs the parent. passed→failed and added-and-failing are high; pre-existing failures go in the
    note as not evidence (5 of 7 false blocks in the v4b failure sheet came from those).
  - `reviewer`: Haiku 5.5 lists defects as JSON and never decides.
- Findings are cached per tool + version + task + head sha (`review/tools/findings.json`). Every gate in a matrix
  reads identical findings; failed tool runs are retried rather than cached.
- Each tool is measured before any gate reads it: fires on bug PRs, fires on clean PRs, gold-hunk hit.
- `review/agent/harness.py`: the gate sees findings as F1..Fn and cites them in `evidence`. Evidence policies
  (`free`, `block-needs-finding`, `approve-needs-clean-tools`, `approve-needs-probe`) bounce a decision that breaks a
  rule once, then escalate it. Probe tests from the v5 loop are off by default (`--probes N`).

Checks so far. On a planted undefined name in a tenacity PR, static returns 3 findings: pyright
`reportUndefinedVariable`, ruff F821, ruff B006. It reports 0 at the parent and takes 1.6 s. On 6 real upstream
tenacity commits it returns 0 findings (10–15 diagnostics at head, all pre-existing or on unchanged lines). Known
limit: mutants were selected to survive the suite, so `tests` is empty on every synthetic bug by construction.

```
make harness-images                       # docker, once per task image
make harness-tools                        # docker, no API: per-tool table on shwin-bench
make harness GATE=claude-sonnet-5-5 TOOLS=static,tests POLICY=block-needs-finding
```

Next: reviewer smoke and a local Qwen3-8B gate on MLX (measurement-job decode settings), the gate × tools matrix on
shwin-bench, then on the 61-task eval, then the evidence-policy ablation.

# ft-lab phase 3b: results so far and where to go (Oct 25, 2026)

Artifacts in this directory (pulled from the `results/phase3` branch):
- `review_grpo_4b*/eval_*.json` — every eval on the frozen 61-task set: `rows[]` with `task_id`, `kind` (bug|clean), `completion`, `score`. Decision = the `"decision"` field in the JSON part of `completion` (strip any `<think>…</think>` first).
- `review_grpo_4b_v2/log_history.json`, `review_grpo_4b_v3/log_history.json` — per-step GRPO metrics (`reward`, `reward_std`, `kl`, `review/approve_bug_frac`, `review/unparsable_frac`, `completions/mean_length`, `sampling/importance_sampling_ratio/mean`, `frac_reward_zero_std`, …).
- `review_grpo_4b_v3/eval_curve.jsonl` — held-out eval every 25 steps during v3b.
Code: `ftlab/rewards/review.py` (reward), `ftlab/train.py` + `ftlab/periodic_eval.py` (GRPO), `ftlab/sft.py`, `configs/review_*_v{1..5}.yaml`, `review/analysis/traces.py` + `rationales.py` (teacher traces), `review/bench/public_*.jsonl` (bench), `review/mine/` (how the bench was built), README.md "Phase 3b".

## Setup
- **Goal:** a small model as a PR "triage gate": diff + context graph in → `approve | block | escalate` + comments out. Approval merges to prod, so approving a bug is the worst outcome.
- **Bench v4 (frozen):** 189 Python tasks from 11 repos (click, pydantic, fastapi, starlette, more-itertools, …). Positives = single-line mutants of real clean PRs, kept only if the test suite still passes AND a generated probe test catches them (so the suite itself cannot reveal them); 2 real bisected regressions. Negatives = the paired clean PRs. Eval 61 (35 bug / 26 clean), train 128. Hand-labelled noise 9 %.
- **Frontier with tools (harness v5, can run tests):** Sonnet 5.5 approves 2/35 bugs, auto-ships 38 % of clean; Haiku 0/35, 26 %. Neither has ever caught a bug with a probe — they block by reading the diff.
- **Student:** Qwen3-4B, LoRA r32, no tools. Reward: approve-a-bug −5, correct +1, block-a-clean −0.5, escalate −0.2, localized block +0.5, unparsable −1 (−3 from v3), token price past a budget.
- **Metric that matters:** bugs approved (must be ≈0) at the highest clean auto-ship rate. Accuracy alone is misleading (block-all scores 0.57, approve-all 0.43).

## Runs (all evaluated on the same 61)

| run | start | GRPO | bugs approve/block | clean approve/block | what happened |
|---|---|---|---|---|---|
| base | Qwen3-4B, no thinking | — | 32 / 3 | 26 / 0 | approves everything |
| v1 | SFT on 54:25 Sonnet decisions | 120 steps, lr 1e-5, β .04, group-scaled | 1 / 34 | 0 / 26 | SFT = block-all; GRPO moved nothing (all 8 samples per group identical → zero advantage, kl 0.001) |
| v2 | SFT on 25:25 | 300 steps, lr 5e-5, β .01, group-scaled | 35 / 0 | 26 / 0 | SFT start varied (31/35, 5/26); GRPO slid to approve-all despite −5 being logged every step |
| v3a | same SFT | lr 2e-5, group-scaled | collapsed by step 18 | | 11-token approve-everything |
| v3b | same SFT | 200 steps, lr 1e-5, β .02, **scale_rewards none**, G 16, eval/25 | 0 / 35 | 0 / 26 | step 25: 6/29 and 9/17 (moving both ways); block-all from step 50 onward |
| v4 base | Qwen3-4B **thinking on**, T 0.6 | — | 22 / 9 (4 cut off) | 23 / 2 | **first start that discriminates**: P(block\|bug) .29 vs P(block\|clean) .08 |
| v4 SFT | + SFT on 50 Sonnet rationales | not run | 20 / 5 (9 cut off) | 15 / 3 (8 cut off) | worse than base on every column; 18/61 never close `<think>` |

## Diagnoses (each from logs, not guessed)
1. **Group std-scaling erased the asymmetry.** `(r − mean)/std` per group makes a bug group (−5 vs +1) and a clean group (+1 vs −0.5) both push with |adv| ≈ 1. The −5 never reached the optimizer in v1–v3a. Fixed in v3b (`scale_rewards: none`).
2. **Non-discriminating starts collapse to a corner.** SFT'd students block bugs and clean PRs at nearly the same rate (0.89 vs 0.81). With nothing to sharpen, GRPO picks whichever corner the advantages favor: approve-all under scaling (shortest output wins), block-all under raw advantages (−5 dominates). Importance-sampling ratio ≈1 throughout, so this is the policy, not a sampler bug.
3. **The suite delta can't help by construction.** Positives were selected to survive the suite.
4. **Thinking mode is the only thing that produced discrimination**, and distilling Sonnet's rationales *destroyed* it (50 samples overwrote the base's own reasoning). Greedy decoding in thinking mode loops (Qwen's own warning) — evals must use T 0.6.
5. **Small-data SFT is the wrong warm start here** three times out of three.

## Planned next (v5, ready, ~$6): GRPO straight from the thinking base, no SFT, raw advantages, unparsable −3, rationale free to ~1200 tokens, lr 1e-5, β 0.02, G 16, 75 steps, T 0.6 eval every 25. `configs/review_grpo_4b_v5.yaml`.

## Variations worth debating
**Reward shape**
- Block-all is the rational optimum when P(bug) ≈ 0.5 and approve-bug = −5. Options: raise block_clean to −1.5/−2; or reweight so expected reward of block-all ≈ approve-all ≈ 0 and only discrimination pays; or a *pairwise* reward — score the mutant and its clean twin together (same PR!) and pay only when the two decisions differ in the right direction. The paired bench makes this possible and it directly targets discrimination.
- Reward the think block: a cheap verifier that checks whether the rationale names the mutated line (we know it) → partial credit for localization in reasoning, not just in comments.
- Length: is 1200 free tokens right? v4 base used ~900 median.

**Optimizer**
- G 16 → 32 on fewer prompts per step (more block/approve contrast within a group).
- β: 0.02 vs 0 (Dr. GRPO) vs 0.1; lr 1e-5 vs 3e-6 with more steps.
- Curriculum: start on the pairs where the base already differs (easy), add the rest.
- Filter prompts where all 16 samples agree (zero signal) — TRL logs `frac_reward_zero_std`.

**Data**
- 128 train prompts is tiny. Mutation run 5 has 296 positives ready to assemble (eval stays frozen). SWE-smith has 50k of the same construction.
- Are single-line mutants the right difficulty? Ablation: operator vs LLM mutants by split (bench carries `source`).
- Add "easy" negatives/positives for signal density (e.g. mutants the suite *does* catch, with the failing test output in the prompt) as a curriculum stage.

**Model**
- Qwen3-8B thinking base: check its untrained P(block|bug) vs P(block|clean) first (one eval, $0.20). If 8B discriminates far better than 4B, train that.
- Qwen3-4B with a thinking *budget* of 4k instead of 2k.
- Non-Qwen: a code-specialized model.

**Framing**
- Is "gate" the right first task? A detector ("which line is wrong?") is an easier RL target with a verifiable reward (we know the mutated line), and a detector that works turns into a gate trivially. Localization accuracy as the primary metric for one run.
- Two-stage: student proposes a hypothesis + probe test, harness runs it, reward from execution. Tool-using RL, much more expensive, but it's what the eventual agent does.

**Measurement**
- 35 bugs bounds any "0 approved" claim at ~8 %. Enough for direction, not for reliability claims.
- Report op / llm / real splits separately every time (bench has `source`).
- Each run so far is one seed.

## Costs
Pods ≈ $40 so far (A100 $1.80/h, H100 SXM ≈ $2.70/h). Setup is pinned and takes 5 min; a 75-step thinking-mode GRPO on an H100 ≈ 2 h.

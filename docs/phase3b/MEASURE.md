# Measurement job (Oct 25) — pre-registered reading rule

Why: the Oct 25 audit (external, Codex) showed the decision counts behind v1–v4 used a regex that accepted
malformed JSON; the scorer's parser does not. Under the strict parser the thinking base separates 4/35 vs 2/26,
not 9/35 vs 2/26. The periodic-eval histogram disagreed with its own accuracy for the same reason and saved no
completions. Sampling differed between standalone evals (top_p 0.95, seed 0) and the training curve (top_p 1.0,
no seed, no top_k). And the no-tools student was still asked for `proposed_test`. Fix measurement before training.

What runs (`V= bash -s measure`, ~$2): Qwen3-4B untrained, 61 tasks, T 0.6 / top_p 0.95 / top_k 20 / cap 2048,
seeds 0,1,2:
- A: current SYSTEM prompt, thinking on
- B: compact no-test prompt (`SYSTEM_COMPACT`), thinking on
- C: compact prompt, thinking off

Outputs: `runs/measure_4b_{a,b,c}/eval_s{0,1,2}.json` (per task: decision via `parse_review`, finish_reason,
out_tokens, completion) and `runs/measure_table.txt`.

Reading rule (decided before the run):
- Formatting hypothesis (A vs B): B cuts invalid outputs to ≤ 2/61 per seed with bug approvals no more than one
  higher and clean approvals no more than one lower. Invalid→valid blocks is a formatting gain, not detection.
- Thinking hypothesis (B vs C): more PR-weighted correct pairs (mutant blocked AND twin approved) without more
  bug approvals; read the newly correct rationales.
- Stability: direction agrees in ≥ 2 of 3 seeds.
- If B fixes parsing but paired discrimination stays ≈ 3/26: next is the Qwen3-8B eval under the same protocol
  (`MODEL=Qwen/Qwen3-8B`) or data work — not another reward run.

Before the next GRPO run (v5 or later): step-0 eval on the same path (now in `PeriodicEval.on_train_begin`),
parser-consistent curves with saved completions (`runs/<name>/eval_curve/stepNNNN.json`). A checkpoint is
"worth expanding evaluation" at 0 approved bugs, ≥ 5/26 approved clean, ≤ 1 invalid, across seeds.

Bench issues the audit found (not fixed; eval stays frozen, report around them): one eval prompt is an exact
duplicate under two sources (`mut__click__c69643b6__508__{false_true,llm}`), two more in train; 33 synthetic
eval bugs come from 26 PRs → weight by PR; `real__fastapi__6ac12207` has boundary=error (bisected introduction,
not a demonstrated pass→fail regression); all 7 eval repos also appear in train.

## Result (Oct 25, 4B a/b/c, 8B b/c, 8B b@4k; `measure/TABLE.txt`) and second audit
- Compact prompt: invalid 9 → 4 on 4B, bug approvals unchanged; keep it.
- Thinking: −5 bug approvals on 4B (all seeds), paired ≈ same (5.9 vs 5.3). 4B paired ceiling ≈ 6/26 however prompted.
- 8B thinking @2k: paired 6.8, 5–10 truncated. **@4k: paired 9.8 ±1.4, 0 invalid, 0 truncated, bug approve 17.7, clean
  approve 22.3.** Every non-truncated @2k output is identical @4k — the gain is the 22 truncated outputs finishing.
- Audit (Codex, partial — ran out of budget before its report): table reproduced exactly; PR-cluster bootstrap 8B@4k vs 4B B
  = +3.8 paired, 95 % CI ≈ 0–8 (direction real, magnitude borderline); rationales mixed — some explain the mutation,
  some earn paired credit for unrelated claims (the duplicate click pair, `concurrent_tee`); reward v4 would have
  penalized 22/51 correct 8B blocks for length (>1200 tokens); escalate had no optimal region. Fixed in reward v5
  (free tokens 2500, escalate 0) + `mask_truncated_completions`.

## v5 (8B) pre-registered reading rule — `configs/review_grpo_8b_v5.yaml`, ~$9
Eval at steps 0, 10, …, 50 on the training path (strict parser, T 0.6 / top_p 0.95 / top_k 20 / seed 0, cap 4096).
Step 0 must land near the measurement job (bug approve ~18, clean approve ~22, paired ~10); if it doesn't, the
training path differs from the eval path and the run is void. Success = paired rising by ≥ 4 over step 0 at any
checkpoint with bug approvals not higher than step 0 and clean approvals ≥ 20, holding for two consecutive
checkpoints. Failure = bug approvals ↓ only via clean approvals ↓ (block drift), or invalid/capped rising, or kl
> 0.3 with paired flat. One run is one seed; a success is a reason to replicate, not a result.

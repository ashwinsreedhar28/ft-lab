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

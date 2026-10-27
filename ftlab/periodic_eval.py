"""Greedy eval on the held-out split every N GRPO steps, through the trainer's own colocated vLLM.

Phase 3b v1/v2 kept one checkpoint and evaluated only the endpoint, so a run that peaked at step 150 and
collapsed by 300 looked identical to one that never learned. This callback re-uses `trainer.vllm_generation`
(weights already synced each step, no second engine on the GPU), decodes greedily, scores with the task's
`score`, and appends one JSON line per eval to `<output_dir>/eval_curve.jsonl` plus a stdout line.
"""
from __future__ import annotations

import collections
import json
import time
from pathlib import Path

from transformers import TrainerCallback

class PeriodicEval(TrainerCallback):
    def __init__(self, cfg, task, every: int, limit: int = 0, seeds=(0,)):
        self.cfg, self.task, self.every, self.limit = cfg, task, every, limit
        self.seeds = tuple(seeds)   # v5 lesson: one seed on 61 tasks is +-2-4 paired of noise; 3 seeds per checkpoint
        self.trainer = None            # set by the caller after GRPOTrainer is built
        self.rows = None
        self.prompt_ids = None

    def _prepare(self):
        from ftlab.prompting import render
        tok = self.trainer.processing_class
        ds = self.task.load_eval(self.cfg.data, self.cfg.system_prompt)
        self.rows = [dict(r) for r in ds][: self.limit or None]
        self.prompt_ids = [tok(render(tok, r["prompt"], self.cfg.enable_thinking))["input_ids"] for r in self.rows]

    def run(self, step: int, seed: int = 0) -> dict:
        if self.rows is None:
            self._prepare()
        vg = self.trainer.vllm_generation
        tok = self.trainer.processing_class
        t0 = time.time()
        saved = (vg.temperature, vg.top_p, vg.top_k, dict(vg.generation_kwargs))
        # same sampling as ftlab.eval's defaults (and a fixed seed), so the curve and the standalone evals are one experiment
        vg.temperature, vg.top_p, vg.top_k = (0.6, 0.95, 20) if self.cfg.enable_thinking else (0.0, 1.0, 0)
        vg.generation_kwargs = {**vg.generation_kwargs, "seed": seed}
        try:
            vg.sync_weights()
            self.trainer._last_loaded_step = self.trainer.state.global_step
            _, completion_ids, _, _ = vg.generate(prompts=self.prompt_ids, images=None, num_generations=1)
        finally:
            vg.temperature, vg.top_p, vg.top_k, vg.generation_kwargs = saved
        texts = [tok.decode(ids, skip_special_tokens=True) for ids in completion_ids]
        scores = [self.task.score(t, r) for t, r in zip(texts, self.rows)]
        cap = int(self.trainer.args.max_completion_length or 0)
        by, recs = collections.Counter(), []
        for t, r, ids, sc in zip(texts, self.rows, completion_ids, scores):
            rev, ok = self.task.decision_of(t)          # the scorer's parser, not a regex (Oct 25 audit)
            dec = rev.decision if ok else "invalid"
            by[f"{r.get('kind')}->{dec}"] += 1
            recs.append({"task_id": r.get("task_id"), "kind": r.get("kind"), "decision": dec, "parsed": ok, "score": sc,
                         "out_tokens": len(ids), "finish_reason": "length" if cap and len(ids) >= cap else "stop",
                         "completion": t})
        rec = {"step": step, "seed": seed, "n": len(scores), "acc": round(sum(scores) / max(len(scores), 1), 4),
               "sampling": {"temperature": vg.temperature if False else (0.6 if self.cfg.enable_thinking else 0.0), "top_p": 0.95 if self.cfg.enable_thinking else 1.0,
                            "top_k": 20 if self.cfg.enable_thinking else 0, "seed": seed, "max_tokens": cap},
               "decisions": dict(sorted(by.items())), "invalid": sum(not x["parsed"] for x in recs),
               "length_capped": sum(x["finish_reason"] == "length" for x in recs), "wall_s": round(time.time() - t0, 1)}
        out_dir = Path(self.trainer.args.output_dir)
        with (out_dir / "eval_curve.jsonl").open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        (out_dir / "eval_curve").mkdir(exist_ok=True)
        (out_dir / "eval_curve" / f"step{step:04d}_s{seed}.json").write_text(json.dumps({**rec, "rows": recs}, indent=1))
        print(f"=== eval@{step} s{seed}: acc={rec['acc']}  {rec['decisions']}  invalid={rec['invalid']} capped={rec['length_capped']}  ({rec['wall_s']}s)", flush=True)
        return rec

    def run_all(self, step: int):
        recs = [self.run(step, s) for s in self.seeds]
        if len(recs) > 1:
            keys = sorted({k for r in recs for k in r["decisions"]})
            mean = {k: round(sum(r["decisions"].get(k, 0) for r in recs) / len(recs), 1) for k in keys}
            acc = sum(r["acc"] for r in recs) / len(recs)
            print(f"=== eval@{step} mean/{len(recs)} seeds: acc={acc:.4f}  {mean}", flush=True)
        return recs

    def on_train_begin(self, args, state, control, **kw):
        if self.every and state.global_step == 0:
            try:
                self.run_all(0)
            except Exception as e:
                print(f"=== eval@0 FAILED: {type(e).__name__}: {e}", flush=True)

    def on_step_end(self, args, state, control, **kw):
        if self.every and state.global_step % self.every == 0 and state.global_step > 0:
            try:
                self.run_all(state.global_step)
            except Exception as e:  # never let an eval kill a 6-hour run
                print(f"=== eval@{state.global_step} FAILED: {type(e).__name__}: {e}", flush=True)

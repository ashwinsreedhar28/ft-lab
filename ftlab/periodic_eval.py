"""Greedy eval on the held-out split every N GRPO steps, through the trainer's own colocated vLLM.

Phase 3b v1/v2 kept one checkpoint and evaluated only the endpoint, so a run that peaked at step 150 and
collapsed by 300 looked identical to one that never learned. This callback re-uses `trainer.vllm_generation`
(weights already synced each step, no second engine on the GPU), decodes greedily, scores with the task's
`score`, and appends one JSON line per eval to `<output_dir>/eval_curve.jsonl` plus a stdout line.
"""
from __future__ import annotations

import collections
import json
import re
import time
from pathlib import Path

from transformers import TrainerCallback

_DEC = re.compile(r'"decision"\s*:\s*"(\w+)"')


class PeriodicEval(TrainerCallback):
    def __init__(self, cfg, task, every: int, limit: int = 0):
        self.cfg, self.task, self.every, self.limit = cfg, task, every, limit
        self.trainer = None            # set by the caller after GRPOTrainer is built
        self.rows = None
        self.prompt_ids = None

    def _prepare(self):
        from ftlab.prompting import render
        tok = self.trainer.processing_class
        ds = self.task.load_eval(self.cfg.data, self.cfg.system_prompt)
        self.rows = [dict(r) for r in ds][: self.limit or None]
        self.prompt_ids = [tok(render(tok, r["prompt"], self.cfg.enable_thinking))["input_ids"] for r in self.rows]

    def run(self, step: int) -> dict:
        if self.rows is None:
            self._prepare()
        vg = self.trainer.vllm_generation
        tok = self.trainer.processing_class
        t0 = time.time()
        saved = vg.temperature
        vg.temperature = 0.0
        try:
            vg.sync_weights()
            self.trainer._last_loaded_step = self.trainer.state.global_step
            _, completion_ids, _, _ = vg.generate(prompts=self.prompt_ids, images=None, num_generations=1)
        finally:
            vg.temperature = saved
        texts = [tok.decode(ids, skip_special_tokens=True) for ids in completion_ids]
        scores = [self.task.score(t, r) for t, r in zip(texts, self.rows)]
        by = collections.Counter()
        for t, r in zip(texts, self.rows):
            m = _DEC.search(t)
            by[f"{r.get('kind')}->{m.group(1) if m else 'unparsable'}"] += 1
        rec = {"step": step, "n": len(scores), "acc": round(sum(scores) / max(len(scores), 1), 4),
               "decisions": dict(sorted(by.items())), "wall_s": round(time.time() - t0, 1)}
        out = Path(self.trainer.args.output_dir) / "eval_curve.jsonl"
        with out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"=== eval@{step}: acc={rec['acc']}  {rec['decisions']}  ({rec['wall_s']}s)", flush=True)
        return rec

    def on_step_end(self, args, state, control, **kw):
        if self.every and state.global_step % self.every == 0 and state.global_step > 0:
            try:
                self.run(state.global_step)
            except Exception as e:  # never let an eval kill a 6-hour run
                print(f"=== eval@{state.global_step} FAILED: {type(e).__name__}: {e}", flush=True)

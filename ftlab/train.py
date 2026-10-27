"""GRPO training entry point.

    python -m ftlab.train configs/gsm8k_h100.yaml [key=value ...]
    accelerate launch --num_processes N -m ftlab.train configs/x.yaml   # multi-GPU

Everything task-specific comes from the config: the data adapter, the reward
names, the system prompt. Everything TRL-specific sits under `grpo:` and is
forwarded to GRPOConfig untouched.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import yaml

from ftlab.config import RunConfig, load_config, to_dict
from ftlab.data import get_task
from ftlab.rewards import get_rewards


def build_grpo_config(cfg: RunConfig, out_dir: Path):
    from trl import GRPOConfig

    grpo = dict(cfg.grpo)
    grpo.setdefault("output_dir", str(out_dir))
    grpo.setdefault("report_to", "none")
    grpo.setdefault("logging_steps", 1)
    grpo.setdefault("bf16", True)
    grpo.setdefault("save_strategy", "steps")
    grpo.setdefault("save_steps", 50)
    grpo.setdefault("save_total_limit", 2)
    grpo.setdefault("log_completions", True)
    grpo.setdefault("num_completions_to_print", 2)
    # Transformers v5 loads in fp32 unless told otherwise; bf16 weights halve memory.
    mik = dict(grpo.get("model_init_kwargs") or {})
    mik.setdefault("dtype", "bfloat16")
    grpo["model_init_kwargs"] = mik
    ctk = dict(grpo.get("chat_template_kwargs") or {})
    from ftlab.prompting import supports_thinking
    if supports_thinking(cfg.model):
        ctk.setdefault("enable_thinking", cfg.enable_thinking)
    grpo["chat_template_kwargs"] = ctk
    if cfg.reward_weights:
        grpo["reward_weights"] = cfg.reward_weights
    return GRPOConfig(**grpo)


def build_peft_config(cfg: RunConfig):
    if not cfg.lora.enabled:
        return None
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        task_type="CAUSAL_LM",
    )


class _LineLog:
    """One compact stdout line per logging step; the full log_history is saved at the end."""

    def __init__(self):
        from transformers import TrainerCallback

        class CB(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kw):
                if not logs or "reward" not in logs:
                    return
                keys = ["reward", "reward_std", "kl", "completions/mean_length",
                        "completions/clipped_ratio", "step_time", "loss"]
                parts = [f"step {state.global_step}"]
                for k in keys:
                    if k in logs:
                        parts.append(f"{k.split('/')[-1]}={logs[k]:.3f}")
                for k, v in logs.items():
                    if k.startswith("rewards/") and k.endswith("/mean"):
                        parts.append(f"{k.split('/')[1]}={v:.3f}")
                print("  ".join(parts), flush=True)

        self.cb = CB()


def drop_long_prompts(ds, cfg):
    """GRPO does not truncate prompts: a review prompt longer than the vLLM window minus the completion
    budget raises VLLMValidationError mid-run (a 36,993-token PR killed phase 3b at step 16). Drop them."""
    from transformers import AutoTokenizer
    from ftlab.prompting import render
    grpo = cfg.grpo
    budget = int(grpo.get("vllm_max_model_length") or 0) - int(grpo.get("max_completion_length") or 0)
    if budget <= 0 or "prompt" not in ds.column_names:
        return ds
    tok = AutoTokenizer.from_pretrained(cfg.model)
    def ok(row):
        return len(tok(render(tok, row["prompt"], cfg.enable_thinking))["input_ids"]) <= budget
    keep = ds.filter(ok)
    print(f"prompt budget {budget} tokens: kept {len(keep)}/{len(ds)} train rows", flush=True)
    return keep


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        sys.exit(__doc__)
    cfg = load_config(argv[0], argv[1:])
    out_dir = cfg.resolved_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.resolved.yaml").write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False))

    task = get_task(cfg.data.task)
    train_ds = task.load_train(cfg.data, cfg.system_prompt)
    train_ds = drop_long_prompts(train_ds, cfg)
    print(f"train rows: {len(train_ds)}  columns: {train_ds.column_names}", flush=True)

    from trl import GRPOTrainer

    callbacks = [_LineLog().cb]
    every = int(cfg.data.extra.get("eval_every") or 0) if isinstance(cfg.data.extra, dict) else 0
    pe = None
    if every and cfg.grpo.get("use_vllm", True) and cfg.grpo.get("vllm_mode", "colocate") == "colocate":
        from ftlab.periodic_eval import PeriodicEval
        pe = PeriodicEval(cfg, task, every, int(cfg.data.extra.get("eval_limit") or 0),
                          seeds=cfg.data.extra.get("eval_seeds") or (0,))
        callbacks.append(pe)

    trainer = GRPOTrainer(
        model=cfg.model,
        reward_funcs=get_rewards(cfg.rewards),
        args=build_grpo_config(cfg, out_dir),
        train_dataset=train_ds,
        peft_config=build_peft_config(cfg),
        callbacks=callbacks,
    )
    if pe is not None:
        pe.trainer = trainer

    t0 = time.time()
    trainer.train(resume_from_checkpoint=os.environ.get("RESUME") or None)
    wall = time.time() - t0

    final = out_dir / "final"
    trainer.save_model(str(final))
    trainer.processing_class.save_pretrained(str(final))
    if cfg.lora.enabled:
        # vLLM eval wants plain weights; keep the adapter too (that's the org artifact).
        merged = out_dir / "merged"
        base = trainer.accelerator.unwrap_model(trainer.model)
        base.merge_and_unload().save_pretrained(str(merged), safe_serialization=True)
        trainer.processing_class.save_pretrained(str(merged))

    hist = trainer.state.log_history
    (out_dir / "log_history.json").write_text(json.dumps(hist, indent=1))
    summary = {
        "name": cfg.name,
        "model": cfg.model,
        "task": cfg.data.task,
        "lora": cfg.lora.enabled,
        "steps": trainer.state.global_step,
        "wall_s": round(wall, 1),
        "num_gpus": trainer.accelerator.num_processes,
        "final_reward": next((h["reward"] for h in reversed(hist) if "reward" in h), None),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

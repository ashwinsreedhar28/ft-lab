"""SFT warm-start for the review gate (phase 3b, step 1).

    python -m ftlab.sft configs/review_sft_4b.yaml [--data data/review_sft.jsonl]

Prompt/completion conversational JSONL (review/analysis/traces.py) -> TRL SFTTrainer with LoRA, loss on the
completion only. Saves the merged model to runs/<name>/final so GRPO (ftlab.train) and eval (ftlab.eval)
load it as a plain HF path. Why warm-start at all: a 4B base will not emit the decision JSON reliably, and
GRPO on a policy that cannot format its answer spends its first hundred steps learning braces (phase 0).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ftlab.prompting import supports_thinking
from ftlab.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--data", default="data/review_sft.jsonl")
    ap.add_argument("--set", action="append", default=[], help="dotted override, e.g. sft.num_train_epochs=2")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    if isinstance(cfg.data.extra, dict) and cfg.data.extra.get("sft_file"):
        a.data = cfg.data.extra["sft_file"]        # e.g. data/review_sft_v4.jsonl (rationale traces)
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    out_dir = Path("runs") / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("json", data_files=a.data, split="train")
    ds = ds.remove_columns([c for c in ds.column_names if c not in ("prompt", "completion")])
    sft = dict(cfg.sft)
    args = SFTConfig(output_dir=str(out_dir), bf16=True, logging_steps=5, save_strategy="no", report_to=[],
                     completion_only_loss=True, **sft)
    tok = AutoTokenizer.from_pretrained(cfg.model)
    if supports_thinking(tok):
        # Render exactly as eval/GRPO do (prompt ends with the empty <think/> block when thinking is off),
        # so the student never learns to open a <think> block of its own. TRL reads this column per example.
        ctk = {"enable_thinking": cfg.enable_thinking}
        ds = ds.map(lambda _: {"chat_template_kwargs": ctk})
    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=torch.bfloat16)
    peft = None
    if cfg.lora.enabled:
        peft = LoraConfig(r=cfg.lora.r, lora_alpha=cfg.lora.alpha, target_modules=cfg.lora.target_modules, task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=peft)
    trainer.train()
    final = out_dir / "final"
    m = trainer.model
    if peft is not None:
        m = m.merge_and_unload()
    m.save_pretrained(final, safe_serialization=True)
    tok.save_pretrained(final)
    (out_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    (out_dir / "summary.json").write_text(json.dumps({"name": cfg.name, "model": cfg.model, "n_train": len(ds), "sft": sft,
                                                       "final_loss": next((h["loss"] for h in reversed(trainer.state.log_history) if "loss" in h), None)}, indent=1))
    print(f"SFT done: {len(ds)} examples -> {final}")


if __name__ == "__main__":
    main()

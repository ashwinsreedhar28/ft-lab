"""Greedy eval on the frozen held-out split. Same prompt, same scorer as training.

    python -m ftlab.eval configs/gsm8k_h100.yaml --label base
    python -m ftlab.eval configs/gsm8k_h100.yaml --label tuned --model runs/gsm8k_h100/final
    python -m ftlab.eval configs/spider_h100.yaml --label haiku --backend anthropic --model claude-haiku-4-5
    python -m ftlab.eval --table runs/spider_h100/eval_*.json

Writes runs/<name>/eval_<label>.json: accuracy, Wilson 95% CI, per-row preds.
Backends: vllm (local HF path or hub id) and anthropic (ANTHROPIC_API_KEY from env).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import os
import time
from pathlib import Path

from ftlab.config import load_config
from ftlab.data import get_task


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def gen_vllm(model: str, prompts: list[str], max_tokens: int, gpu_mem: float, max_model_len: int, temperature: float = 0.0, seed: int = 0) -> list[str]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=gpu_mem,
              max_model_len=max_model_len, enable_prefix_caching=True)
    sp = SamplingParams(temperature=temperature, top_p=0.95 if temperature > 0 else 1.0, max_tokens=max_tokens, seed=seed)
    outs = llm.generate(prompts, sp)
    return [o.outputs[0].text for o in outs]


def gen_anthropic(model: str, messages_list: list[list[dict]], max_tokens: int, workers: int = 8) -> list[str]:
    from concurrent.futures import ThreadPoolExecutor

    import anthropic

    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY from env

    def one(msgs):
        system = next((m["content"] for m in msgs if m["role"] == "system"), None)
        user = [m for m in msgs if m["role"] != "system"]
        for attempt in range(5):
            try:
                kw = {"model": model, "max_tokens": max_tokens, "messages": user}
                if system:
                    kw["system"] = system
                r = client.messages.create(**kw)
                return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            except anthropic.RateLimitError:
                time.sleep(2 ** attempt)
        return ""

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, messages_list))


def run_eval(cfg_path: str, label: str, model: str | None, backend: str, max_tokens: int,
             gpu_mem: float, max_model_len: int, overrides: list[str], temperature: float = 0.0):
    cfg = load_config(cfg_path, overrides)
    task = get_task(cfg.data.task)
    ds = task.load_eval(cfg.data, cfg.system_prompt)
    model = model or cfg.model
    # default the vLLM window and completion cap to what the config trains with
    max_model_len = max_model_len or int(cfg.grpo.get("vllm_max_model_length") or 4096)
    max_tokens = max_tokens or int(cfg.grpo.get("max_completion_length") or 1024)
    rows = [dict(r) for r in ds]
    t0 = time.time()
    if backend == "vllm":
        from transformers import AutoTokenizer

        from ftlab.prompting import render

        tok = AutoTokenizer.from_pretrained(model)
        prompts = [render(tok, r["prompt"], cfg.enable_thinking) for r in rows]
        texts = gen_vllm(model, prompts, max_tokens, gpu_mem, max_model_len, temperature)
    elif backend == "anthropic":
        texts = gen_anthropic(model, [r["prompt"] for r in rows], max_tokens)
    else:
        raise ValueError(backend)
    wall = time.time() - t0

    scores = [task.score(t, r) for t, r in zip(texts, rows)]
    k, n = int(sum(scores)), len(scores)
    lo, hi = wilson(k, n)
    out = {
        "label": label, "model": model, "backend": backend, "task": cfg.data.task,
        "n": n, "correct": k, "acc": k / n, "ci95": [lo, hi], "wall_s": round(wall, 1),
        "max_tokens": max_tokens, "temperature": temperature,
        "no_answer_line": sum(1 for t in texts if "####" not in t),
        "rows": [
            {k2: v for k2, v in r.items() if k2 != "prompt"} | {"completion": t, "score": s}
            for r, t, s in zip(rows, texts, scores)
        ],
    }
    out_dir = cfg.resolved_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"eval_{label}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"{label}: {k}/{n} = {k/n:.3f}  (95% CI {lo:.3f}-{hi:.3f})  {wall:.0f}s  -> {path}")
    return out


_SEED_RE = re.compile(r"^(.*)_s(\d+)$")


def table(paths: list[str]):
    """One row per label; labels `<name>_s<seed>` collapse to mean ± half-range over seeds.

    The CI column is the Wilson interval for a single run (eval-set noise); the ± column
    is spread across training seeds (run-to-run noise). A lift smaller than either is not a result.
    """
    rows = [json.loads(Path(p).read_text()) for p in paths]
    groups: dict[str, list[dict]] = {}
    for r in rows:
        m = _SEED_RE.match(r["label"])
        groups.setdefault(m.group(1) if m else r["label"], []).append(r)
    out = []
    for label, rs in groups.items():
        accs = [r["acc"] for r in rs]
        mean = sum(accs) / len(accs)
        spread = (max(accs) - min(accs)) / 2 if len(accs) > 1 else None
        lo, hi = wilson(round(mean * rs[0]["n"]), rs[0]["n"])
        out.append((label, rs[0]["model"], rs[0]["n"], mean, spread, len(rs), lo, hi))
    out.sort(key=lambda t: -t[3])
    print(f"{'label':<14}{'model':<40}{'n':>6}{'acc':>8}{'±seeds':>9}{'runs':>5}{'wilson95':>16}")
    for label, model, n, mean, spread, k, lo, hi in out:
        sp = f"{spread:.3f}" if spread is not None else "—"
        print(f"{label:<14}{model[:39]:<40}{n:>6}{mean:>8.3f}{sp:>9}{k:>5}{lo:>8.3f}-{hi:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("--label", default="base", help="use <name>_s<seed> for seed runs; --table groups on the prefix")
    ap.add_argument("--model", default=None, help="HF id / local path / anthropic model id")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "anthropic"])
    ap.add_argument("--max-tokens", type=int, default=None, help="default: config max_completion_length")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=None, help="default: config vllm_max_model_length")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy; Qwen3 recommends 0.6")
    ap.add_argument("--table", nargs="*", help="print a comparison table from eval json files")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    a = ap.parse_args()
    if a.table:
        return table(a.table)
    if not a.config:
        ap.error("config required")
    run_eval(a.config, a.label, a.model, a.backend, a.max_tokens, a.gpu_mem, a.max_model_len, a.set, a.temperature)


if __name__ == "__main__":
    main()

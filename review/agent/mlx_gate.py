"""Local gate on Apple silicon via mlx-lm: `--gate mlx:<hf repo>` (default size for a 16 GB Mac: Qwen3-8B 4-bit).

Decode settings are the measurement job's (docs/phase3b/MEASURE.md): thinking on, T 0.6, top_p 0.95, top_k 20,
4096 new tokens. Caveat when comparing to the pod numbers: those were bf16 under vLLM; this is a 4-bit MLX quant.

The model loads once per process. Seeding: `seed(n)` once per run (harness --seed); every later draw follows from it,
so a run is reproducible given the same task order. Returns the same (text, (in, out), usd) shape as loop.call;
`<think>` content is stripped from the text (the harness and parse_review only read the JSON after it).
"""
from __future__ import annotations

import re

DEFAULT_REPO = "mlx-community/Qwen3-8B-4bit"
DECODE = {"temp": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 4096, "enable_thinking": True}

_loaded: dict = {}
_last: dict = {}          # finish_reason / tokens of the last call, for the harness audit

_THINK = re.compile(r"<think>.*?</think>", re.S)


def strip_think(text: str) -> str:
    """Pure: drop closed <think> blocks; an unclosed one (cut off by the token cap) leaves nothing usable."""
    out = _THINK.sub("", text)
    if "<think>" in out:
        out = out.split("<think>", 1)[0]
    return out.strip()


def chat(system: str, messages: list[dict]) -> list[dict]:
    """Pure: the harness's (system, messages) as one chat list for the template."""
    return [{"role": "system", "content": system}] + [{"role": m["role"], "content": m["content"]} for m in messages]


def seed(n: int) -> None:
    import mlx.core as mx
    mx.random.seed(n)


def _model(repo: str):
    if repo not in _loaded:
        from mlx_lm import load
        _loaded.clear()
        _loaded[repo] = load(repo)
    return _loaded[repo]


def call(repo: str, system: str, messages: list[dict], max_tokens: int | None = None):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    model, tok = _model(repo or DEFAULT_REPO)
    prompt = tok.apply_chat_template(chat(system, messages), add_generation_prompt=True,
                                     enable_thinking=DECODE["enable_thinking"])
    sampler = make_sampler(temp=DECODE["temp"], top_p=DECODE["top_p"], top_k=DECODE["top_k"])
    text, last = "", None
    for r in stream_generate(model, tok, prompt, max_tokens=max_tokens or DECODE["max_tokens"], sampler=sampler):
        text += r.text
        last = r
    _last.clear()
    if last is not None:
        _last.update(finish_reason=last.finish_reason, prompt_tokens=last.prompt_tokens,
                     generation_tokens=last.generation_tokens, generation_tps=round(last.generation_tps, 1),
                     peak_memory_gb=round(last.peak_memory, 2))
    n_in = last.prompt_tokens if last else len(prompt)
    n_out = last.generation_tokens if last else 0
    return strip_think(text), (n_in, n_out), 0.0

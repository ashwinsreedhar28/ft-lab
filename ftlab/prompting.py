"""Prompt construction shared by train and eval.

Rows are stored in TRL's conversational format: `prompt` is a list of chat
messages. TRL applies the model's chat template itself (with
`chat_template_kwargs={"enable_thinking": False}` for Qwen3), so train and eval
render the same way as long as eval uses `render()` below with the same tokenizer.
"""
from __future__ import annotations


def make_messages(user: str, system: str | None = None) -> list[dict[str, str]]:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return msgs


def render(tokenizer, messages: list[dict[str, str]], enable_thinking: bool = False) -> str:
    """Chat-template a message list into the generation prompt string."""
    kwargs = {}
    if "qwen3" in getattr(tokenizer, "name_or_path", "").lower():
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs
    )


def completion_text(completion) -> str:
    """TRL hands conversational completions as [{"role": "assistant", "content": ...}]."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "") or ""
    return str(completion)

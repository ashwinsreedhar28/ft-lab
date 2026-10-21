"""Asymmetric review-gate reward (phase 3).

  approve a bug PR        -5.0   the shipped defect
  correct approve/block   +1.0
  block a clean PR        -0.5   wasted engineer time
  escalate (either kind)  -0.2   cheap but useless if always chosen
  block w/ localized hint +0.5   the block was for the right reason
  unparsable output       -1.0   (scored as escalate on top)
  tokens                  -0.1 per 256 completion tokens beyond 300   (phase 0 lesson: price length
                                                                      or the policy pays for reward with it)
"""
from __future__ import annotations

from ftlab.data import review as task
from ftlab.prompting import completion_text

W = {"approve_bug": -5.0, "correct": 1.0, "block_clean": -0.5, "escalate": -0.2, "localized": 0.5, "unparsable": -1.0,
     "token_unit": -0.1, "free_tokens": 300, "token_unit_size": 256}


W_V3 = {**W, "unparsable": -3.0}   # v2 drifted into single-quoted dicts at -1: format drift must never be a wash
W_V4 = {**W_V3, "free_tokens": 1200}   # thinking mode: a ~150-word rationale is free, padding past ~1200 tokens is not


def gate(completions, kind, gold_hunks, completion_ids=None, log_metric=None, log_extra=None, W=W, **_):
    rewards, decisions, approve_bug = [], [], 0
    lens = [len(ids) for ids in completion_ids] if completion_ids is not None else [0] * len(completions)
    for c, k, gh, n_tok in zip(completions, kind, gold_hunks, lens):
        rev, ok = task.decision_of(completion_text(c))
        r = 0.0 if ok else W["unparsable"]
        r += W["token_unit"] * max(0, n_tok - W["free_tokens"]) / W["token_unit_size"]
        if rev.decision == "escalate":
            r += W["escalate"]
        elif rev.decision == "approve":
            if k == "bug":
                r += W["approve_bug"]; approve_bug += 1
            else:
                r += W["correct"]
        else:  # block
            if k == "bug":
                r += W["correct"] + (W["localized"] if task.localized(rev, gh) else 0.0)
            else:
                r += W["block_clean"]
        rewards.append(r); decisions.append(rev.decision if ok else "unparsable")
    if log_extra:
        log_extra("decision", decisions)
    if log_metric:
        n = max(len(rewards), 1)
        log_metric("review/approve_bug_frac", approve_bug / n)
        log_metric("review/escalate_frac", sum(d == "escalate" for d in decisions) / n)
        log_metric("review/unparsable_frac", sum(d == "unparsable" for d in decisions) / n)
        log_metric("review/mean_tokens", sum(lens) / n)
    return rewards


def gate_v3(completions, kind, gold_hunks, **kw):
    return gate(completions, kind, gold_hunks, W=W_V3, **kw)


def gate_v4(completions, kind, gold_hunks, **kw):
    return gate(completions, kind, gold_hunks, W=W_V4, **kw)

"""Asymmetric review-gate reward (phase 3).

  approve a bug PR        -5.0   the shipped defect
  correct approve/block   +1.0
  block a clean PR        -0.5   wasted engineer time
  escalate (either kind)  -0.2   cheap but useless if always chosen
  block w/ localized hint +0.5   the block was for the right reason
  unparsable output       -1.0   (scored as escalate on top)
"""
from __future__ import annotations

from ftlab.data import review as task
from ftlab.prompting import completion_text

W = {"approve_bug": -5.0, "correct": 1.0, "block_clean": -0.5, "escalate": -0.2, "localized": 0.5, "unparsable": -1.0}


def gate(completions, kind, gold_hunks, log_metric=None, log_extra=None, **_):
    rewards, decisions, approve_bug = [], [], 0
    for c, k, gh in zip(completions, kind, gold_hunks):
        rev, ok = task.decision_of(completion_text(c))
        r = 0.0 if ok else W["unparsable"]
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
    return rewards

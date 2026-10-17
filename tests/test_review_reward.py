import json

from ftlab.rewards.review import gate


def _c(text):
    return [{"role": "assistant", "content": text}]


def test_gate_rewards():
    comps = [
        _c('{"decision":"approve","comments":[]}'),                                 # bug approved
        _c('{"decision":"block","comments":[{"file":"a.py","line_start":10,"line_end":12,"body":"x"}]}'),  # localized block
        _c('{"decision":"block","comments":[]}'),                                   # block clean
        _c('{"decision":"escalate","comments":[]}'),
        _c('not json at all'),
        _c('{"decision":"approve","comments":[]}'),                                 # correct approve
    ]
    kind = ["bug", "bug", "clean", "clean", "bug", "clean"]
    gh = [json.dumps([{"file": "a.py", "line_start": 10, "line_end": 11}])] * 2 + ["[]"] * 4
    seen = {}
    r = gate(comps, kind=kind, gold_hunks=gh, log_metric=lambda k, v: seen.__setitem__(k, v))
    assert r == [-5.0, 1.5, -0.5, -0.2, -1.2, 1.0]          # no completion_ids -> no token cost
    assert seen["review/approve_bug_frac"] == 1 / 6
    assert seen["review/unparsable_frac"] == 1 / 6


def test_token_cost():
    comps = [_c('{"decision":"approve","comments":[]}')] * 2
    r = gate(comps, kind=["clean", "clean"], gold_hunks=["[]", "[]"], completion_ids=[[0] * 300, [0] * 812])
    assert r[0] == 1.0 and abs(r[1] - (1.0 - 0.2)) < 1e-9     # 512 tokens past the free 300 = 2 units

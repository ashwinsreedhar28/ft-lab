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
    assert r == [-5.0, 1.5, -0.5, -0.2, -1.2, 1.0]
    assert seen["review/approve_bug_frac"] == 1 / 6
    assert seen["review/unparsable_frac"] == 1 / 6

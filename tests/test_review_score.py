from review.bench.schema import Comment, Hunk, Review, Task, overlaps
from review.bench.score import score


def _t(i, kind, split="fresh", hunks=()):
    return Task(id=f"t{i}", repo="r", kind=kind, base_sha="a", head_sha="b", diff="", files=["f.py"],
                split=split, gold_hunks=[Hunk(*h) for h in hunks])


def test_overlap_slack():
    h = Hunk("f.py", 10, 20)
    assert overlaps(Comment("f.py", 22, 22, "x"), h)
    assert not overlaps(Comment("f.py", 30, 31, "x"), h)
    assert not overlaps(Comment("g.py", 10, 20, "x"), h)


def test_score_metrics():
    tasks = [_t(0, "bug", hunks=[("f.py", 10, 20)]), _t(1, "bug", hunks=[("f.py", 50, 60)]),
             _t(2, "clean"), _t(3, "clean")]
    reviews = {
        "t0": Review("block", [Comment("f.py", 12, 12, "off by one")], usd=0.01),
        "t1": Review("approve", usd=0.01),               # the shipped defect
        "t2": Review("approve", usd=0.01),
        "t3": Review("escalate", usd=0.01),
    }
    s = score(tasks, reviews)
    assert s["bugs_approved"] == 1
    assert s["auto_ship_rate"] == 0.5
    assert s["escalation_rate"] == 0.25
    assert s["block_precision"] == 1.0
    assert s["localization"] == 1.0
    assert abs(s["usd_per_pr"] - 0.01) < 1e-9
    assert s["bugs_approved_rate_ub95"] is None


def test_rule_of_three_bound():
    tasks = [_t(i, "bug") for i in range(42)]
    reviews = {t.id: Review("block") for t in tasks}
    s = score(tasks, reviews)
    assert 0.06 < s["bugs_approved_rate_ub95"] < 0.075


def test_parse_review_ignores_think_block():
    from review.agent.run import parse_review

    rev, ok = parse_review('<think>\nsee {foo: bar} above\n</think>\n\n{"decision": "block", "comments": []}', (0, 0), 0.0)
    assert ok and rev.decision == "block"
    rev, ok = parse_review('<think>\ncut off {"decision": "approve"', (0, 0), 0.0)
    assert not ok

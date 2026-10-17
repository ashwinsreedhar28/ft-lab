import random

from review.mine.mutate import OPS, op_mutants, pytest_ok, script, SKIP_LINE


def test_every_operator_changes_its_match():
    import re
    samples = {"rel_lt_le": "if a < b:", "rel_eq_ne": "if a == b:", "bool_and_or": "if a and b:", "not_drop": "if not x:",
               "not_add": "if x:", "is_none_flip": "if x is None:", "plus1_minus1": "n = k + 1", "true_false": "flag = True",
               "zero_one": "return items[0]", "return_none": "    return value", "raise_pass": "    raise ValueError(x)",
               "range_n_minus1": "for i in range(n):", "slice_drop_last": "xs = ys[:-1]"}
    for name, pat, rep in OPS:
        if name in samples:
            assert re.sub(pat, rep, samples[name], count=1) != samples[name], name


def test_op_mutants_one_per_line_and_skips_noise():
    lines = [(10, "    if a < b and c:"), (11, "    # if x == y"), (12, "    import os"), (13, "    return a + 1")]
    ms = op_mutants(lines, 5, random.Random(0))
    assert {m["line"] for m in ms} <= {10, 13} and len(ms) == 2
    assert all(m["mutated"] != m["original"] for m in ms)


def test_skip_line_filters_defs_and_comments():
    assert SKIP_LINE.match("def f(x):") and SKIP_LINE.match("    # note") and SKIP_LINE.match("") and not SKIP_LINE.match("    x = 1")


def test_script_applies_edit_before_pytest_and_captures_diff():
    s = script("a" * 40, "pip install -e .", {"pkg/m.py": "x = 1\n"}, ("tests/test_p.py", "def test_a(): pass\n"), ["tests/test_p.py"], True, "b" * 40)
    assert s.index("base64 -d > pkg/m.py") < s.index("python -m pytest")
    assert "git diff " + "b" * 40 in s and "PYTHONPATH=/w/src:/w" in s
    assert pytest_ok("...\nEXIT=0") and not pytest_ok("EXIT=1") and not pytest_ok("no marker")

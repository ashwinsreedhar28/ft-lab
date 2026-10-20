import random

from review.mine.mutate import OPS, junit, killed_by, op_mutants, passed, pytest_ok, script, stage_status, SKIP_LINE


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


def test_script_mounts_edits_before_pytest_and_captures_diff():
    s, files = script("a" * 40, "pip install -e .", {"pkg/m.py": "x = 1\n"}, ("tests/test_p.py", "def test_a(): pass\n"), ["tests/test_p.py"], True, "b" * 40)
    assert s.index("cp /inject/edit_0 pkg/m.py") < s.index("python -m pytest") and files == {"edit_0": "x = 1\n", "probe": "def test_a(): pass\n"}
    assert "git diff " + "b" * 40 in s and "PYTHONPATH=/w/src:/w" in s and "base64" not in s
    assert pytest_ok("...\nEXIT=0") and not pytest_ok("EXIT=1") and not pytest_ok("no marker")


def _junit(cases):
    body = "".join(f'<testcase classname="tests.t" name="{n}">{"<failure/>" if s == "failed" else "<error/>" if s == "error" else ""}</testcase>' for n, s in cases)
    return f"...\nEXIT=1\nJUNIT_BEGIN\n<testsuites><testsuite>{body}</testsuite></testsuites>\nJUNIT_END"


def test_kill_criterion_is_relative_to_baseline():
    base = _junit([("a", "passed"), ("b", "passed"), ("tty_only", "failed")])
    mut = _junit([("a", "passed"), ("b", "failed"), ("tty_only", "failed")])
    same = _junit([("a", "passed"), ("b", "passed"), ("tty_only", "error")])
    aborted = _junit([("a", "passed")])
    assert passed(base) == {"tests.t::a", "tests.t::b"} and junit(base)["tests.t::tty_only"] == "failed"
    assert killed_by(passed(base), mut) == {"tests.t::b"} and killed_by(passed(base), same) == set()
    assert stage_status(passed(base), mut) == ("killed", {"tests.t::b"})
    assert stage_status(passed(base), same)[0] == "survived"
    assert stage_status(passed(base), aborted)[0] == "error"
    assert junit("no markers") == {} and junit("JUNIT_BEGIN\n<bad\nJUNIT_END") == {}


def test_statement_ops_skip_continuation_lines():
    src = ["def f():", "    return (", "        x,", "        y)", "    raise ValueError(x)"]
    lines = [(3, "        x,"), (5, "    raise ValueError(x)")]
    ms = op_mutants(lines, 5, random.Random(0), src)
    assert all(not (m["operator"] in ("return_none", "raise_pass") and m["line"] == 3) for m in ms)
    assert any(m["operator"] == "raise_pass" and m["line"] == 5 for m in ms)


def test_operators_skip_docstrings_and_doctest_prompts():
    from review.mine.mutate import docstring_lines
    src = ['def f():', '    """Doc.', '', '        >>> f(1) > 2', '        True', '    """', '    return a > b  # x > y']
    assert docstring_lines(src) == {2, 3, 4, 5, 6}
    ms = op_mutants([(4, src[3]), (7, src[6])], 5, random.Random(0), src)
    assert all(m["line"] == 7 for m in ms) and ms

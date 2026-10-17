from review.mine.validate import classify, run_tests_script


def test_valid_when_new_test_fails_at_parent_passes_at_commit():
    st, oracle = classify({"t.py::a": "failed", "t.py::b": "passed"}, {"t.py::a": "passed", "t.py::b": "passed"}, "", "")
    assert st == "valid" and oracle == ["t.py::a"]


def test_interface_leak_when_parent_only_errors():
    st, _ = classify({"t.py::a": "error"}, {"t.py::a": "passed"}, "", "")
    assert st == "interface_leak"


def test_not_reproduced_and_not_green_and_install_fail():
    assert classify({"t.py::a": "passed"}, {"t.py::a": "passed"}, "", "")[0] == "not_reproduced"
    assert classify({"t.py::a": "failed"}, {"t.py::a": "failed"}, "", "")[0] == "not_green"
    assert classify({}, {}, "", "")[0] == "not_green"
    assert classify({}, {"t.py::a": "passed"}, "INSTALL_FAIL\n", "")[0] == "install_fail"


def test_script_overlays_tests_from_commit_and_turns_off_deps():
    s = run_tests_script("p" * 40, ["tests/test_x.py"], "c" * 40, False)
    assert f"git show {'c' * 40}:tests/test_x.py" in s and "--no-deps" in s and "PIPESTATUS" in s
    assert "git show" not in run_tests_script("c" * 40, ["tests/test_x.py"], None, False)

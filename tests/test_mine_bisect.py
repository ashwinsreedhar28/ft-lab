from review.mine.bisect import search


def mk(states):
    calls = []
    def status(i):
        calls.append(i); return states[i]
    return status, calls


def test_finds_fail_to_pass_boundary_in_log_probes():
    # index 0 = parent (fail) ... bug introduced at index 37 (fails from 0..37, passes 38..)
    states = ["fail"] * 38 + ["pass"] * 362
    status, calls = mk(states)
    v, lo, hi = search(400, status)
    assert (v, lo, hi) == ("ok", 37, 38) and len(calls) <= 9


def test_error_boundary_counts_as_module_creation():
    states = ["fail"] * 5 + ["error"] * 20
    v, lo, hi = search(25, mk(states)[0])
    assert (v, lo, hi) == ("ok", 4, 5)


def test_older_than_window_and_non_monotone():
    assert search(10, mk(["fail"] * 10)[0])[0] == "older_than_window"
    assert search(10, mk(["pass"] * 10)[0])[0] == "non_monotone"
    assert search(10, mk(["docker_error"] * 10)[0])[0] == "docker_error"
    # cannot build at the far end: the install failure bounds the search instead of aborting it
    v, lo, hi = search(3000, mk(["fail"] * 1200 + ["docker_error"] * 1800)[0])
    assert (v, lo, hi) == ("ok", 1199, 1200)

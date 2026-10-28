import json

import pytest

from review.agent import harness
from review.bench.schema import Hunk
from review.tools import Finding, ToolRun, added_lines, gather, hits_gold, number, py_files, render, tool_stats
from review.tools import reviewer, static, tests as tests_tool

DIFF = """diff --git a/pkg/a.py b/pkg/a.py
index 1..2 100644
--- a/pkg/a.py
+++ b/pkg/a.py
@@ -10,4 +10,5 @@ def f():
 x = 1
-y = 2
+y = 3
+z = 4
 return x
@@ -40,2 +41,2 @@ def g():
-    a = b
+    a = c
 pass
diff --git a/pkg/new.py b/pkg/new.py
new file mode 100644
--- /dev/null
+++ b/pkg/new.py
@@ -0,0 +1,2 @@
+import os
+print(os)
diff --git a/pkg/gone.py b/pkg/gone.py
deleted file mode 100644
--- a/pkg/gone.py
+++ /dev/null
@@ -1,1 +0,0 @@
-x = 1
diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1 +1 @@
-old
+new
"""


def test_added_lines():
    a = added_lines(DIFF)
    assert a["pkg/a.py"] == {11, 12, 41}
    assert a["pkg/new.py"] == {1, 2}
    assert "pkg/gone.py" not in a
    assert py_files(DIFF) == ["pkg/a.py", "pkg/new.py"]


def test_static_image_names():
    assert static.static_image("shwin/emberserve:0a59d6") == "shwin/emberserve:0a59d6-static"
    assert static.static_image("rb-mine-click") == "rb-mine-click:static"
    assert static.static_image("localhost:5000/x") == "localhost:5000/x:static"


def test_ruff_and_pyright_items():
    ruff = json.dumps([{"code": "F821", "message": "Undefined name `q`", "filename": "/testbed/pkg/a.py",
                        "location": {"row": 12, "column": 1}, "end_location": {"row": 12, "column": 2}}])
    pyr = json.dumps({"generalDiagnostics": [
        {"file": "/testbed/pkg/a.py", "severity": "error", "message": "Cannot access attribute \"x\"",
         "range": {"start": {"line": 40, "character": 0}, "end": {"line": 40, "character": 5}}, "rule": "reportAttributeAccessIssue"},
        {"file": "/testbed/pkg/a.py", "severity": "warning", "message": "w",
         "range": {"start": {"line": 0}, "end": {"line": 0}}, "rule": "x"},
        {"file": "/testbed/pkg/a.py", "severity": "error", "message": "Import could not be resolved",
         "range": {"start": {"line": 1}, "end": {"line": 1}}, "rule": "reportMissingImports"}]})
    r = static.ruff_items(ruff, ".")
    p = static.pyright_items(pyr, ".")
    assert r == [{"file": "pkg/a.py", "line": 12, "end": 12, "rule": "F821", "message": "Undefined name `q`", "tool": "ruff"}]
    assert len(p) == 1 and p[0]["line"] == 41 and p[0]["rule"] == "reportAttributeAccessIssue"
    assert static.ruff_items("not json", ".") == [] and static.pyright_items("", ".") == []


def test_parse_findings_filters_unchanged_and_preexisting():
    added = added_lines(DIFF)
    head = [
        {"file": "pkg/a.py", "line": 12, "end": 12, "rule": "F821", "message": "Undefined name `q`", "tool": "ruff"},
        {"file": "pkg/a.py", "line": 10, "end": 10, "rule": "F841", "message": "unused x", "tool": "ruff"},      # not added
        {"file": "pkg/a.py", "line": 41, "end": 41, "rule": "B006", "message": "mutable default", "tool": "ruff"},
        {"file": "pkg/a.py", "line": 41, "end": 41, "rule": "reportCallIssue", "message": "moved", "tool": "pyright"},
    ]
    base = [{"file": "pkg/a.py", "line": 38, "end": 38, "rule": "reportCallIssue", "message": "moved", "tool": "pyright"}]
    f = static.parse_findings(head, base, added)
    assert [(x.line_start, x.rule, x.severity) for x in f] == [(12, "ruff F821", "high"), (41, "ruff B006", "medium")]
    assert all(x.tool == "static" for x in f)


def test_tests_delta_findings():
    assert tests_tool.test_file("tests.test_core.TestX::test_y") == "tests/test_core.py"
    assert tests_tool.test_file("tests.test_core::test_y") == "tests/test_core.py"
    parent = {"t.a::x": "passed", "t.a::y": "failed", "t.a::gone": "passed", "t.a::z": "passed"}
    head = {"t.a::x": "failed", "t.a::y": "failed", "t.a::new": "error", "t.a::z": "passed"}
    f, note = tests_tool.delta_findings(parent, head)
    assert [(x.severity, x.rule) for x in f] == [("high", "t.a::x"), ("high", "t.a::new"), ("low", "removed")]
    assert "1 also fail at the parent" in note


def test_reviewer_parse():
    ok = reviewer.parse('noise {"findings": [{"file": "a.py", "line_start": 3, "severity": "HIGH", "message": "off  by one"},'
                        ' {"file": "b.py", "line_start": "x", "severity": "weird", "message": "m"}]} trailing')
    assert [(x.file, x.line_start, x.line_end, x.severity, x.message) for x in ok] == [
        ("a.py", 3, 3, "high", "off by one"), ("b.py", 0, 0, "medium", "m")]
    assert reviewer.parse('{"findings": []}') == []
    assert reviewer.parse("no json here") is None


def _runs():
    return [ToolRun("static", "s", [Finding("static", "low", "a.py", 5, 5, "l", "ruff F841"),
                                    Finding("static", "high", "a.py", 9, 9, "h", "ruff F821")]),
            ToolRun("tests", "t", [], "suite fine"),
            ToolRun("reviewer", "r", [], "boom", ok=False)]


def test_number_and_render():
    runs = _runs()
    fs = number(runs)
    assert [(f.id, f.severity) for f in fs] == [("F1", "high"), ("F2", "low")]
    txt = render(runs)
    assert "F1 high   a.py:9 ruff F821  h" in txt
    assert "[tests: t] 0 findings. suite fine" in txt
    assert "did not run (boom)" in txt


def test_hits_gold_and_stats():
    f = [Finding("reviewer", "high", "a.py", 100, 100, "m")]
    assert hits_gold(f, [Hunk("a.py", 102, 102)]) and not hits_gold(f, [Hunk("a.py", 110, 110)])
    assert hits_gold(f, [{"file": "a.py", "line_start": 97, "line_end": 97}])
    tasks = [{"id": "b", "kind": "bug", "gold_hunks": [Hunk("a.py", 100, 100)]}, {"id": "c", "kind": "clean", "gold_hunks": []}]
    per = {"b": [ToolRun("reviewer", "r", f)], "c": [ToolRun("reviewer", "r", [Finding("reviewer", "low", "z.py", 1, 1, "m")])]}
    s = tool_stats(per, tasks)["reviewer"]
    assert (s["fired_bug"], s["fired_clean"], s["gold_hit"], s["n_bug"], s["n_clean"]) == (1, 0, 1, 1, 1)


def test_gather_caches_and_survives_tool_failure(tmp_path, monkeypatch):
    import review.tools as T
    calls = {"n": 0}

    class Good:
        @staticmethod
        def version(o): return "g1"
        @staticmethod
        def run(task, o):
            calls["n"] += 1
            return ToolRun("good", "g1", [Finding("good", "medium", "a.py", 1, 1, "m")])

    class Bad:
        @staticmethod
        def version(o): return "b1"
        @staticmethod
        def run(task, o): raise RuntimeError("docker down")

    monkeypatch.setattr(T, "CACHE", tmp_path / "f.json")
    monkeypatch.setattr(T, "_adapters", lambda: {"good": Good, "bad": Bad})
    task = {"id": "t1", "head_sha": "abc"}
    r1 = gather(task, ["good", "bad"])
    r2 = gather(task, ["good"])
    assert calls["n"] == 1 and r2[0].findings[0].message == "m"
    assert r1[1].ok is False and "docker down" in r1[1].note
    assert not any(k.startswith("bad:") for k in json.loads((tmp_path / "f.json").read_text()))   # failures not cached


def test_cited_and_policies():
    fs = number([ToolRun("static", "s", [Finding("static", "high", "a.py", 9, 9, "h")]),
                 ToolRun("reviewer", "r", [Finding("reviewer", "medium", "a.py", 3, 3, "m")])])
    valid = {f.id for f in fs}
    obj = {"decision": "block", "evidence": ["F2", "F9"], "comments": [{"body": "see F1 for the undefined name"}]}
    assert harness.cited(obj, valid) == ["F1", "F2"]
    P = harness.apply_policy
    assert P("block", [], fs, ["free"]) is None
    assert "Block rejected" in P("block", [], fs, ["block-needs-finding"])
    assert P("block", ["F1"], fs, ["block-needs-finding"]) is None
    assert P("block", [], fs, ["block-needs-finding"], probe_failed=True, probes_enabled=True) is None
    assert "F1" in P("approve", [], fs, ["approve-needs-clean-tools"])
    assert P("approve", [], fs[1:], ["approve-needs-clean-tools"]) is None
    assert "probe" in P("approve", [], [], ["approve-needs-probe"])
    assert P("escalate", [], fs, list(harness.POLICIES)) is None


def test_system_prompt_variants():
    s0 = harness.system_prompt([], ["free"], 0)
    assert "No review tools ran" in s0 and "Rules of this gate" not in s0 and "probe" not in s0
    s1 = harness.system_prompt(["static"], ["block-needs-finding", "approve-needs-clean-tools"], 0)
    assert "findings F1..Fn" in s1 and "cites at least one finding id." in s1 and "high-severity" in s1
    s2 = harness.system_prompt(["static"], ["block-needs-finding"], 2)
    assert "or a probe you ran that failed" in s2 and '"probe"' in s2


def test_parse_list_rejects_unknown():
    assert harness.parse_list("static, tests") == ["static", "tests"]
    assert harness.parse_list("none") == []
    with pytest.raises(SystemExit):
        harness.parse_list("static,lint", ("static", "tests"))


def test_pyright_config_paths_are_relative():
    c = json.loads(static._pyright_config(".", ["pkg/a.py"]))
    assert c["include"] == ["../pkg/a.py"] and c["extraPaths"] == ["..", "../src"]
    assert c["typeCheckingMode"] == "basic" and c["reportMissingImports"] == "none"
    c = json.loads(static._pyright_config("sub", ["sub/x.py"]))
    assert c["extraPaths"] == ["../sub", "../sub/src"]
    assert not any(p.startswith("/") for p in c["include"] + c["extraPaths"])


def test_static_run_parses_long_output(monkeypatch):
    """Regression: docker_run used to keep the last 6000 chars, which cut the BEGIN markers off big pyright JSON."""
    import review.agent.loop as loop
    diags = [{"file": f"/testbed/pkg/a.py", "severity": "error", "message": f"m{i}", "rule": "reportCallIssue",
              "range": {"start": {"line": 200 + i}, "end": {"line": 200 + i}}} for i in range(400)]
    diags.append({"file": "/testbed/pkg/a.py", "severity": "error", "message": "\"q\" is not defined",
                  "rule": "reportUndefinedVariable", "range": {"start": {"line": 10}, "end": {"line": 10}}})
    pyr = json.dumps({"generalDiagnostics": diags, "summary": {"filesAnalyzed": 1}})
    out = f"RUFF_HEAD_BEGIN\n[]\nRUFF_HEAD_END\nPYR_HEAD_BEGIN\n{pyr}\nPYR_HEAD_END\nRUFF_BASE_BEGIN\n[]\nRUFF_BASE_END\nPYR_BASE_BEGIN\n{pyr[:0] or '{}'}\nPYR_BASE_END\n"
    assert len(out) > 6000
    seen = {}

    def fake_run(img, script, timeout=1200, network="none", mounts=None, keep=6000, files=None):
        seen["keep"] = keep
        return 0, out if keep is None else out[-keep:]

    monkeypatch.setattr(static, "ensure_image", lambda image, build=True: image + "-static")
    monkeypatch.setattr(loop, "docker_run", fake_run)
    monkeypatch.setattr(loop, "prelude", lambda task, sha: "true")
    task = {"id": "t", "image": "img:1", "diff": DIFF, "base_sha": "b", "head_sha": "h"}
    r = static.run(task)
    assert seen["keep"] is None and r.ok
    assert [(f.line_start, f.severity) for f in r.findings] == [(11, "high")]


def _shwin_task():
    from review.bench.score import load_tasks
    return [t.to_json() for t in load_tasks("review/bench/tasks.jsonl")][0]


def _fake_gate(monkeypatch, replies):
    seen = []

    def fake(gate, system, messages, max_tokens):
        seen.append([dict(m) for m in messages])
        return replies[min(len(seen) - 1, len(replies) - 1)], (100, 10), 0.001
    monkeypatch.setattr(harness, "gate_call", fake)
    return seen


def test_review_one_bounce_then_escalate(monkeypatch):
    seen = _fake_gate(monkeypatch, ['{"decision": "block", "evidence": [], "comments": []}'])
    runs = [ToolRun("static", "s", [Finding("static", "medium", "a.py", 3, 3, "m")])]
    rev, ok, text, probes, audit = harness.review_one(_shwin_task(), "fake", runs, ["static"], ["block-needs-finding"], 0, 100)
    assert len(seen) == 2 and json.loads(text)["decision"] == "escalate" and "policy:" in json.loads(text)["harness_note"]
    assert len(audit["bounces"]) == 2 and probes == []
    roles = [m["role"] for m in seen[1]]
    assert roles == ["user", "assistant", "user"] and "Final round" in seen[1][-1]["content"]   # bounce + nudge merged
    assert "F1 medium a.py:3" in seen[0][0]["content"]


def test_review_one_cited_block_stands_and_format_retry(monkeypatch):
    _fake_gate(monkeypatch, ["no json at all", '{"decision": "block", "evidence": ["F1"], "comments": []}'])
    runs = [ToolRun("static", "s", [Finding("static", "high", "a.py", 3, 3, "m")])]
    rev, ok, text, probes, audit = harness.review_one(_shwin_task(), "fake", runs, ["static"], ["block-needs-finding"], 0, 100)
    assert json.loads(text)["decision"] == "block" and audit["cited"] == ["F1"] and audit["bounces"] == []
    assert [r["parsed"] for r in audit["rounds"]] == [False, True]


def test_main_gate_row_end_to_end(monkeypatch, tmp_path, capsys):
    import sys
    _fake_gate(monkeypatch, ['{"decision": "approve", "evidence": [], "comments": []}'])
    monkeypatch.setattr(harness, "RESULTS", tmp_path)
    monkeypatch.setattr(harness, "gather", lambda t, tools, opts, refresh=False: [])
    tid = _shwin_task()["id"]
    monkeypatch.setattr(sys, "argv", ["h", "--tasks", "review/bench/tasks.jsonl", "--gate", "fake", "--tools", "none",
                                      "--only", tid, "--label", "t"])
    harness.main()
    out = json.loads((tmp_path / "t.json").read_text())
    assert out["n"] == 1 and out["reviews"][0]["decision"] == "approve" and out["condition"] == "none"
    assert "bugs_approved=" in capsys.readouterr().out

from review.agent.loop import extract_reply, no_probe_reason

PROBE = '{"path":"tests/test_probe_review.py","source":"def test_x():\\n    assert 1","hypothesis":"h"}'


def test_plain_json():
    o = extract_reply('{"decision":"escalate","comments":[]}')
    assert o["decision"] == "escalate" and "probe" not in o


def test_fenced_with_probe():
    o = extract_reply('```json\n{"decision":"pending","probe":' + PROBE + '}\n```')
    assert o["probe"]["path"] == "tests/test_probe_review.py"


def test_invoke_wrapped_probe_is_recovered():
    # Sonnet 5.5 in v4 emitted the probe inside hallucinated tool-call syntax; v4 scored it unparsable.
    o = extract_reply('<invoke name="bash">\n<parameter name="probe">' + PROBE + '</parameter>\n</invoke>')
    assert "source" in o["probe"] and o.get("decision") is None


def test_invoke_placeholder_then_json():
    o = extract_reply('<invoke name="">\n</invoke>\n{"decision":"escalate","comments":[]}')
    assert o["decision"] == "escalate"


def test_no_probe_reason():
    r = {"round": 0, "decision": "escalate", "probe_requested": False, "probe_runnable": False, "parsed": True}
    assert no_probe_reason([r], True) == "model_decided_round0:escalate"
    assert no_probe_reason([r], False) == "no_image"
    assert no_probe_reason([{**r, "decision": None, "parsed": False}], True) == "unparsable"
    assert no_probe_reason([{**r, "decision": None, "invoke_syntax": True}], True) == "format_failure"

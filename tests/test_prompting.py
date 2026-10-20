

def test_supports_thinking_reads_template_not_path():
    from ftlab.prompting import supports_thinking

    class Tok:
        name_or_path = "runs/review_sft_4b/final"
        chat_template = "{% if enable_thinking is defined and not enable_thinking %}<think>\n\n</think>{% endif %}"

    class Plain:
        name_or_path = "Qwen/Qwen3-4B"
        chat_template = "{{ messages }}"

    assert supports_thinking(Tok())
    assert not supports_thinking(Plain())

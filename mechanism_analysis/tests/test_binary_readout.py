import pytest

from binary_readout import (TEMPLATES, build_answer_contrast, contexts_of, format_prompt,
                            margin_from_token_logprobs, render_chat_prompt, vllm_margin,
                            vllm_requests)


class AnswerTokenizer:
    """Tokenizer with declared tokenizations of the two complete answers."""

    def __init__(self, encodings):
        self.encodings = encodings

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is True
        return list(self.encodings[text])


class ClosedReasoningTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=None):
        self.calls.append(enable_thinking)
        return f"<user>{messages[0]['content']}</user><assistant><think>\n\n</think>\n\n"


class OpenReasoningTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return ("<system><think>quoted system marker</think></system>"
                f"<user>{messages[0]['content']}</user><assistant><think>\n")


def test_seven_templates_and_context_order():
    assert len(TEMPLATES) == 7 and len(set(TEMPLATES)) == 7
    assert contexts_of(["a", "b"]) == [[], ["a", "b"], ["a"], ["b"]]
    assert "(no posts provided)" in format_prompt("E", [], "c", TEMPLATES[0])
    assert "1. a\n2. b" in format_prompt("E", ["a", "b"], "c", TEMPLATES[0])


def test_answers_split_at_their_common_token_prefix():
    base = "<assistant><think></think>\n"
    contrast = build_answer_contrast(AnswerTokenizer({
        base + "Answer: A": [1, 2, 3, 10], base + "Answer: B": [1, 2, 3, 11]}), base)
    assert contrast.prefix_ids == (1, 2, 3)
    assert contrast.true_suffix_ids == (10,) and contrast.false_suffix_ids == (11,)


def test_colon_and_label_fused_in_one_token():
    contrast = build_answer_contrast(AnswerTokenizer({
        "<assistant>Answer: A": [1, 2, 20], "<assistant>Answer: B": [1, 2, 21]}), "<assistant>")
    assert contrast.prefix_ids == (1, 2)
    assert contrast.true_suffix_ids == (20,) and contrast.false_suffix_ids == (21,)


def test_margin_sums_every_suffix_token():
    assert margin_from_token_logprobs((-0.2, -0.3), (-0.1, -0.4, -0.7)) == pytest.approx(0.7)


def test_reasoning_is_disabled_and_a_closed_block_is_kept():
    tok = ClosedReasoningTokenizer()
    chat = render_chat_prompt(tok, "prompt")
    assert tok.calls == [False]
    assert chat.endswith("</think>\n\n")


def test_only_the_block_opened_in_the_assistant_turn_is_closed():
    prompt = "evidence literally contains <think>quoted</think>"
    chat = render_chat_prompt(OpenReasoningTokenizer(), prompt)
    tail = chat[chat.rfind(prompt) + len(prompt):]
    assert tail.count("<think>") == tail.count("</think>") == 1


def test_vllm_margin_reads_the_suffix_positions():
    contrast = build_answer_contrast(AnswerTokenizer({
        "baseAnswer: A": [1, 2, 10, 12], "baseAnswer: B": [1, 2, 11]}), "base")
    assert vllm_requests(contrast) == ((1, 2, 10, 12), (1, 2, 11))

    class LP:
        def __init__(self, value):
            self.logprob = value

    margin = vllm_margin(contrast, [None, None, {10: LP(-0.2)}, {12: LP(-0.3)}],
                         [None, None, {11: LP(-1.1)}])
    assert margin == pytest.approx(0.6)

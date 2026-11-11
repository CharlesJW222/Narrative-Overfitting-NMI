"""Binary belief readout for the natural corpus.

The model reads a prompt with the posts of a feed and one claim. The readout
compares the two complete answers "Answer: A" (claim TRUE) and "Answer: B"
(claim FALSE). The belief margin of one read is

    log P("Answer: A" | prompt) - log P("Answer: B" | prompt),

computed over the tokens after the longest common token prefix of the two
answers. The margin m(C) of a context C is the mean over the seven prompt
templates. Reasoning is disabled.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

LABEL_TRUE = "A"
LABEL_FALSE = "B"
ANSWER_PREFIX = "Answer:"
MAX_EVIDENCE_CHARS = 6000

_FIRST_TEMPLATE = (
    "You are assessing a claim based on a sequence of social-media posts about a news event.\n\n"
    "Event: {event}\n\n"
    "Posts (in the order received):\n{evidence}\n\n"
    "Claim under assessment: \"{claim}\"\n\n"
    "Based ONLY on the posts above, is the claim more likely TRUE or FALSE?\n"
    "Answer with a single letter: A = the claim is TRUE, B = the claim is FALSE.\n"
    "Answer:"
)

# The seven prompt templates. They differ in the instruction text only.
TEMPLATES = [
    _FIRST_TEMPLATE,
    (
        "Read the following posts about a news event and judge one claim.\n\n"
        "Event: {event}\n\n"
        "Posts in order:\n{evidence}\n\n"
        "Claim: \"{claim}\"\n\n"
        "Using only the posts, decide if the claim is TRUE or FALSE.\n"
        "Reply with one letter: A = TRUE, B = FALSE.\n"
        "Answer:"
    ),
    (
        "Below is a timeline of posts concerning a news event, then a claim.\n\n"
        "Event: {event}\n\n"
        "Timeline:\n{evidence}\n\n"
        "Claim to evaluate: \"{claim}\"\n\n"
        "Given only these posts, is the claim TRUE or FALSE?\n"
        "Output a single character: A for TRUE, B for FALSE.\n"
        "Answer:"
    ),
    (
        "Consider these posts about a news event and one claim about it.\n\n"
        "Event: {event}\n\n"
        "Posts received in sequence:\n{evidence}\n\n"
        "Claim: \"{claim}\"\n\n"
        "On the basis of the posts alone, is this claim TRUE or FALSE?\n"
        "Answer with a single letter, A = TRUE or B = FALSE.\n"
        "Answer:"
    ),
    (
        "The following are posts about a developing news event, followed by a claim.\n\n"
        "Event: {event}\n\n"
        "Posts:\n{evidence}\n\n"
        "Statement to judge: \"{claim}\"\n\n"
        "Relying only on what the posts say, decide TRUE or FALSE.\n"
        "Give one character: A means TRUE, B means FALSE.\n"
        "Answer:"
    ),
    (
        "You will see a series of posts about an event, then a single claim.\n\n"
        "Event: {event}\n\n"
        "The posts, in order:\n{evidence}\n\n"
        "Claim: \"{claim}\"\n\n"
        "Judging only from these posts, is the claim TRUE or FALSE?\n"
        "Respond with exactly one letter: A (TRUE) or B (FALSE).\n"
        "Answer:"
    ),
    (
        "Here are posts describing a news event, and a claim to assess.\n\n"
        "Event: {event}\n\n"
        "Posts (chronological):\n{evidence}\n\n"
        "Claim under review: \"{claim}\"\n\n"
        "Using only the posts above, is the claim TRUE or FALSE?\n"
        "Answer with a single letter: A for TRUE, B for FALSE.\n"
        "Answer:"
    ),
]


def format_prompt(event: str, posts: List[str], claim: str, template: str) -> str:
    """Fill one template with the event description, the posts and the claim."""
    if posts:
        evidence = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(posts))
    else:
        evidence = "(no posts provided)"
    if len(evidence) > MAX_EVIDENCE_CHARS:
        evidence = evidence[:MAX_EVIDENCE_CHARS]
    return template.format(event=event, evidence=evidence, claim=claim)


def contexts_of(posts: List[str]) -> List[List[str]]:
    """Contexts read for one feed: empty, the complete feed, then each post alone."""
    return [[]] + [posts] + [[p] for p in posts]


def render_chat_prompt(tokenizer: object, prompt: str) -> str:
    """Apply the chat template of the model with reasoning disabled.

    If the template opens a reasoning block in the assistant turn, the block is
    closed before the answer is scored.
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        chat = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        chat = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    start = chat.rfind(prompt)
    if start < 0:
        raise ValueError("chat template did not preserve the prompt verbatim")
    offset = start + len(prompt)

    def open_blocks(text: str) -> int:
        return text.count("<think>") - text.count("</think>")

    if open_blocks(chat[offset:]) == 1:
        chat += "\n</think>\n\n"
    if open_blocks(chat[offset:]) != 0:
        raise ValueError("assistant reasoning block is unbalanced")
    return chat


@dataclass(frozen=True)
class AnswerContrast:
    """Token ids of the two complete answers, split at their common prefix."""

    prefix_ids: Tuple[int, ...]
    true_suffix_ids: Tuple[int, ...]
    false_suffix_ids: Tuple[int, ...]


def build_answer_contrast(tokenizer: object, chat_prompt: str) -> AnswerContrast:
    """Tokenize both complete answers and split them at their longest common prefix."""
    true_ids = tuple(int(v) for v in tokenizer.encode(
        chat_prompt + f"{ANSWER_PREFIX} {LABEL_TRUE}", add_special_tokens=True))
    false_ids = tuple(int(v) for v in tokenizer.encode(
        chat_prompt + f"{ANSWER_PREFIX} {LABEL_FALSE}", add_special_tokens=True))
    n = 0
    for lhs, rhs in zip(true_ids, false_ids):
        if lhs != rhs:
            break
        n += 1
    prefix_ids, true_suffix_ids, false_suffix_ids = true_ids[:n], true_ids[n:], false_ids[n:]
    if not prefix_ids:
        raise ValueError("the two answers have no common token prefix")
    if not true_suffix_ids or not false_suffix_ids:
        raise ValueError("the two answers must have non-empty divergent suffixes")
    if true_suffix_ids[0] == false_suffix_ids[0]:
        raise ValueError("the answer suffixes do not diverge at the scoring boundary")
    return AnswerContrast(prefix_ids, true_suffix_ids, false_suffix_ids)


def margin_from_token_logprobs(true_token_logprobs: Sequence[float],
                               false_token_logprobs: Sequence[float]) -> float:
    """Belief margin of one read from the log-probabilities of the suffix tokens."""
    values = tuple(float(v) for v in tuple(true_token_logprobs) + tuple(false_token_logprobs))
    if any(not math.isfinite(v) for v in values):
        raise ValueError("answer token log-probabilities must be finite")
    return float(math.fsum(true_token_logprobs)) - float(math.fsum(false_token_logprobs))


def hf_margin(model: object, contrast: AnswerContrast, *, device: str) -> float:
    """Belief margin of one read with a Hugging Face causal language model."""
    import torch

    def score(suffix_ids: Tuple[int, ...]) -> Tuple[float, ...]:
        input_ids = torch.tensor([contrast.prefix_ids + suffix_ids], dtype=torch.long,
                                 device=device)
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits[0]
        start = len(contrast.prefix_ids) - 1
        return tuple(float(torch.log_softmax(logits[start + i], dim=-1)[token_id])
                     for i, token_id in enumerate(suffix_ids))

    return margin_from_token_logprobs(score(contrast.true_suffix_ids),
                                      score(contrast.false_suffix_ids))


def vllm_requests(contrast: AnswerContrast) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Token sequences of the two answers. vLLM returns their prompt log-probabilities."""
    return (contrast.prefix_ids + contrast.true_suffix_ids,
            contrast.prefix_ids + contrast.false_suffix_ids)


def vllm_margin(contrast: AnswerContrast, true_prompt_logprobs: Sequence[object],
                false_prompt_logprobs: Sequence[object]) -> float:
    """Belief margin of one read from two vLLM prompt-logprob responses."""
    start = len(contrast.prefix_ids)

    def extract(prompt_logprobs: Sequence[object], suffix_ids: Tuple[int, ...]):
        if len(prompt_logprobs) != start + len(suffix_ids):
            raise ValueError("vLLM response length does not match the answer token ids")
        values = []
        for i, token_id in enumerate(suffix_ids):
            entry = prompt_logprobs[start + i]
            if entry is None or token_id not in entry:
                raise ValueError(f"vLLM response omitted answer token id {token_id}")
            value = entry[token_id]
            values.append(float(value.logprob if hasattr(value, "logprob") else value))
        return tuple(values)

    return margin_from_token_logprobs(extract(true_prompt_logprobs, contrast.true_suffix_ids),
                                      extract(false_prompt_logprobs, contrast.false_suffix_ids))


def feed_contrasts(tokenizer: object, feed: dict) -> List[AnswerContrast]:
    """Answer contrasts of one feed: every context of `contexts_of` under every template."""
    return [build_answer_contrast(tokenizer, render_chat_prompt(
        tokenizer, format_prompt(feed["event_display"], context, feed["claim"], template)))
        for context in contexts_of(feed["posts"]) for template in TEMPLATES]


def vllm_context_margins(llm: object, tokenizer: object, feeds: List[dict]) -> List[List[float]]:
    """Read a batch of feeds with vLLM.

    Returns, for each feed, the template-mean margin of each context in the
    order of `contexts_of`: empty, complete feed, each post alone.
    """
    import numpy as np
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    contrasts, spans = [], []
    for feed in feeds:
        own = feed_contrasts(tokenizer, feed)
        spans.append((len(contrasts), len(own)))
        contrasts += own
    requests = []
    for contrast in contrasts:
        true_ids, false_ids = vllm_requests(contrast)
        requests.append(TokensPrompt(prompt_token_ids=list(true_ids)))
        requests.append(TokensPrompt(prompt_token_ids=list(false_ids)))
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    outputs = llm.generate(requests, params, use_tqdm=False)
    margins = [vllm_margin(contrast, outputs[k].prompt_logprobs, outputs[k + 1].prompt_logprobs)
               for contrast, k in zip(contrasts, range(0, len(outputs), 2))]
    n = len(TEMPLATES)
    return [[float(np.mean(np.array(margins[start + c * n: start + (c + 1) * n], dtype=np.float64)))
             for c in range(count // n)] for start, count in spans]

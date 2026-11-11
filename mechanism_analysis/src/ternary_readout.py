"""Three-label belief readout for the synthetic testbed.

The model classifies a claim as TRUE, FALSE or INSUFFICIENT from the supplied
evidence. The readout scores each label as a continuation of the prompt.
p_TRUE is the three-way softmax probability of TRUE, and the belief margin is
logit(clip(p_TRUE, 1e-12, 1 - 1e-12)). The three prompt templates differ in the
order of the label descriptions.
"""
from __future__ import annotations

import math

LABELS = ("TRUE", "FALSE", "INSUFFICIENT")

SYSTEM_MESSAGE = (
    "You are an evidence evaluator. Use only the evidence supplied by the "
    "user and follow the output format exactly."
)
USER_STEM = (
    "<EVIDENCE>\n{evidence}\n</EVIDENCE>\n<CLAIM>\n{claim}\n</CLAIM>\n"
    "Using only the evidence above, classify the claim. The labels mean:\n"
    "{label_menu}\nReturn exactly one label and no other text."
)
MENU_ENTRIES = {
    "TRUE": "TRUE — the evidence establishes the claim.",
    "FALSE": "FALSE — the evidence establishes that the claim is false.",
    "INSUFFICIENT": (
        "INSUFFICIENT — the evidence establishes neither the claim nor its "
        "negation."
    ),
}
# Template id -> order of the label descriptions in the prompt.
TEMPLATES = {
    "q1": ("TRUE", "FALSE", "INSUFFICIENT"),
    "q2": ("FALSE", "INSUFFICIENT", "TRUE"),
    "q3": ("INSUFFICIENT", "TRUE", "FALSE"),
}


def render_user_prompt(evidence: str, claim: str, template_id: str) -> str:
    menu = "\n".join(MENU_ENTRIES[label] for label in TEMPLATES[template_id])
    return USER_STEM.format(evidence=evidence, claim=claim, label_menu=menu)


def render_chat_prompt(tokenizer: object, user_prompt: str) -> str:
    """Apply the chat template of the model.

    If the template opens a reasoning block in the assistant turn, the block is
    closed before the labels are scored.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_MESSAGE},
         {"role": "user", "content": user_prompt}],
        tokenize=False, add_generation_prompt=True)
    start = rendered.rfind(user_prompt)
    if start < 0:
        raise ValueError("chat template did not preserve the prompt verbatim")
    tail = rendered[start + len(user_prompt):]
    open_blocks = tail.count("<think>") - tail.count("</think>")
    if open_blocks == 1:
        rendered += "\n</think>\n\n"
    elif open_blocks != 0:
        raise RuntimeError("assistant reasoning block is unbalanced")
    return rendered


def score_label_logprobs(model, tokenizer, prompt_text: str, device: str) -> dict:
    """Log-probability of each label as a continuation of the prompt."""
    import torch
    import torch.nn.functional as F

    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    plen = prompt_ids.shape[1]
    lp = {}
    for label in LABELS:
        lab = tokenizer(label, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        full = torch.cat([prompt_ids, lab], dim=1)
        with torch.no_grad():
            logits = model(full).logits
        logprob = F.log_softmax(logits, dim=-1)
        total = 0.0
        for i in range(lab.shape[1]):
            total += logprob[0, plen + i - 1, lab[0, i]].item()
        lp[label] = total
    return lp


def margin_from_label_logprobs(lp: dict) -> float:
    """Belief margin logit(p_TRUE) from the three label log-probabilities."""
    mx = max(lp.values())
    exp = {k: math.exp(v - mx) for k, v in lp.items()}
    p_true = exp["TRUE"] / sum(exp.values())
    p = min(max(p_true, 1e-12), 1.0 - 1e-12)
    return math.log(p / (1.0 - p))

#!/usr/bin/env python3
"""Belief margin and response dispersion on the synthetic testbed.

Input: a JSONL file with one feed on each line:

    {"world_id": ..., "condition": ..., "claim": ..., "fragments": [...]}

Conditions:
    C     short-fragment control
    C_s   scorer-selected short fragments
    N     random-evidence narrative
    N_s   scorer-selected narrative
    A     attack
    S     supported narrative (reads its own supported claim h'; an optional
          "target_type" field is "direct" or "multi_step")

For each feed and each of the three prompt templates, the model reads the
contexts: no evidence, the complete feed, and each fragment alone. The complete
feed is its fragments joined by line breaks. Output: JSONL with one row for
each (feed, template).
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feeds", required=True)
    ap.add_argument("--model", required=True, help="panel tag (see src/panel_models.py) or local path")
    ap.add_argument("--model-tag", default=None, help="name used in the output (default: --model)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="read only the first N feeds")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dispersion import belief_pulls, response_dispersion
    from panel_models import resolve_model
    from ternary_readout import (TEMPLATES, margin_from_label_logprobs, render_chat_prompt,
                                 render_user_prompt, score_label_logprobs)

    tag = args.model_tag or args.model
    model_dir = resolve_model(args.model)
    feeds = [json.loads(line) for line in Path(args.feeds).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if args.limit:
        feeds = feeds[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16).to(args.device).eval()

    def read_margin(evidence: str, claim: str, template_id: str) -> float:
        prompt = render_chat_prompt(tokenizer, render_user_prompt(evidence, claim, template_id))
        return margin_from_label_logprobs(score_label_logprobs(model, tokenizer, prompt, args.device))

    empty_margins = {}   # the empty-context margin depends on the claim and the template only
    with open(args.out, "w", encoding="utf-8") as out:
        for index, feed in enumerate(feeds):
            for template_id in TEMPLATES:
                key = (feed["claim"], template_id)
                if key not in empty_margins:
                    empty_margins[key] = read_margin("", feed["claim"], template_id)
                margin_empty = empty_margins[key]
                margin = read_margin("\n".join(feed["fragments"]), feed["claim"], template_id)
                pulls = belief_pulls(margin_empty, [read_margin(fragment, feed["claim"], template_id)
                                                    for fragment in feed["fragments"]])
                out.write(json.dumps({
                    "model": tag, "world_id": feed["world_id"], "condition": feed["condition"],
                    "target_type": feed.get("target_type"), "template": template_id,
                    "dispersion": response_dispersion(pulls), "margin": margin,
                    "margin_empty": margin_empty, "n_fragments": len(feed["fragments"])}) + "\n")
            if (index + 1) % 50 == 0:
                print(f"[{tag}] {index + 1}/{len(feeds)} feeds", flush=True)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Select the attack feeds of the natural corpus with a reference model.

The reference model reads each generated attack feed with the binary belief
readout. A feed is accepted if its full-feed belief margin is above zero.
Feeds with a generator artifact (a refusal or a "fictional" label) are marked.

Output: JSON with one row for each generated attack feed.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--attacks-glob", required=True,
                    help="Generative Montage result files, for example 'attacks/*/result_*.json'")
    ap.add_argument("--model", default=None,
                    help="local path or Hugging Face id of the reference model (default: Qwen3-8B)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from binary_readout import TEMPLATES, build_answer_contrast, format_prompt, hf_margin, \
        render_chat_prompt
    from feeds import load_attack_feeds
    from panel_models import REFERENCE_MODEL

    model_name = args.model
    if model_name is None:
        from huggingface_hub import snapshot_download
        model_name = snapshot_download(REFERENCE_MODEL[0], revision=REFERENCE_MODEL[1])
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(args.device).eval()

    def full_feed_margin(feed) -> float:
        reads = []
        for template in TEMPLATES:
            prompt = format_prompt(feed.event_display, feed.posts, feed.claim, template)
            contrast = build_answer_contrast(tokenizer, render_chat_prompt(tokenizer, prompt))
            reads.append(hf_margin(model, contrast, device=args.device))
        return float(np.array(reads, dtype=np.float64).mean())

    feeds = load_attack_feeds(args.attacks_glob)
    print(f"loaded {len(feeds)} generated attack feeds")
    rows = []
    for i, feed in enumerate(feeds):
        with open(feed.meta["source_path"]) as f:
            source = json.load(f).get("metadata", {})
        margin = full_feed_margin(feed)
        rows.append({"feed_id": feed.feed_id, "event": feed.event, "claim": feed.claim,
                     "refutation_class": source.get("refutation_class", "?"),
                     "generator": source.get("gen_provider"),
                     "margin": margin, "accepted": bool(margin > 0),
                     "generator_artifact": feed.meta["generator_artifact"],
                     "artifact_reason": feed.meta["artifact_reason"],
                     "n_posts": len(feed.posts)})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(feeds)}  last margin={margin:+.1f}", flush=True)

    valid = [r for r in rows if not r["generator_artifact"]]
    out = {"reference_model": model_name, "n_generated": len(rows),
           "n_generator_artifacts": len(rows) - len(valid),
           "n_valid": len(valid), "n_accepted": sum(r["accepted"] for r in valid),
           "rows": rows}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"accepted {out['n_accepted']}/{out['n_valid']} valid attack feeds "
          f"({out['n_generator_artifacts']} generator artifacts excluded)")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

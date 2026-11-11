#!/usr/bin/env python3
"""Belief margin, belief pulls and response dispersion on the natural corpus.

For each feed F = (p_1, ..., p_n) and its claim, the model reads these contexts
under seven prompt templates: no posts, the complete feed, and each post alone.

    m(C)      template-mean belief margin of context C
    delta_i   = m({p_i}) - m(empty)            belief pull of post i
    u(F)      = sd(delta_1..delta_n, ddof=1)   response dispersion

Inference uses vLLM. Output: JSON with one row for each feed.
"""
import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))


def load_corpus(attacks_glob, selection_path, ordinary_reports_path, exclude_events, seed):
    """Accepted attack feeds plus ordinary-report feeds, in a seeded random order."""
    from feeds import load_attack_feeds

    excluded = set(exclude_events)
    with open(selection_path) as f:
        accepted_ids = {r["feed_id"] for r in json.load(f)["rows"] if r["accepted"]}
    corpus = []
    for feed in load_attack_feeds(attacks_glob, drop_generator_artifacts=True):
        if feed.event in excluded or feed.feed_id not in accepted_ids:
            continue
        with open(feed.meta["source_path"]) as f:
            source = json.load(f).get("metadata", {})
        corpus.append({"feed_id": feed.feed_id, "event": feed.event,
                       "event_display": feed.event_display, "claim": feed.claim,
                       "feed_type": "attack", "construction": None,
                       "generator": source.get("gen_provider"),
                       "refutation_class": source.get("refutation_class", "?"),
                       "posts": feed.posts, "post_ids": feed.post_ids})
    with open(ordinary_reports_path) as f:
        reports = json.load(f)
    for key in ("ordinary_report_shuffled", "ordinary_report_ranked"):
        for feed in reports[key]:
            if feed["event"] in excluded:
                continue
            corpus.append({"feed_id": feed["feed_id"], "event": feed["event"],
                           "event_display": feed["event_display"], "claim": feed["claim"],
                           "feed_type": "ordinary_report", "construction": feed["construction"],
                           "generator": None, "refutation_class": None,
                           "posts": feed["posts"], "post_ids": feed["post_ids"]})
    random.Random(seed).shuffle(corpus)
    return corpus


def summarize_feed(context_margins):
    """Feed statistics from the margins of [empty, complete feed, post_1, ..., post_n]."""
    from dispersion import belief_pulls, response_dispersion

    margin_empty, margin = context_margins[0], context_margins[1]
    pulls = belief_pulls(margin_empty, context_margins[2:])
    dispersion = response_dispersion(pulls)
    # A feed with fewer than two posts has no spread between posts.
    return {"margin": float(margin), "margin_empty": float(margin_empty), "pulls": pulls,
            "dispersion": 0.0 if dispersion is None else dispersion}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="panel tag (see src/panel_models.py) or local path")
    ap.add_argument("--model-tag", default=None, help="name used in the output (default: --model)")
    ap.add_argument("--attacks-glob", required=True)
    ap.add_argument("--selection", required=True, help="output of select_attack_feeds.py")
    ap.add_argument("--ordinary-reports", required=True,
                    help="output of build_ordinary_report_feeds.py")
    ap.add_argument("--exclude-events", nargs="*", default=["putinmissing"])
    ap.add_argument("--seed", type=int, default=7, help="seed of the feed order")
    ap.add_argument("--limit", type=int, default=0, help="read only the first N feeds")
    ap.add_argument("--batch", type=int, default=32, help="feeds for each vLLM call")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from binary_readout import TEMPLATES, vllm_context_margins
    from panel_models import resolve_model
    from transformers import AutoTokenizer
    from vllm import LLM

    tag = args.model_tag or args.model
    model_dir = resolve_model(args.model)
    corpus = load_corpus(args.attacks_glob, args.selection, args.ordinary_reports,
                         args.exclude_events, args.seed)
    if args.limit:
        corpus = corpus[: args.limit]
    n_attack = sum(1 for r in corpus if r["feed_type"] == "attack")
    print(f"[{tag}] {len(corpus)} feeds ({n_attack} attack, {len(corpus) - n_attack} ordinary-report), "
          f"{len(TEMPLATES)} templates", flush=True)

    partial = args.out + ".partial"
    done = {}
    if args.resume and os.path.exists(partial):
        with open(partial) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    done[row["feed_id"]] = row
        print(f"resume: {len(done)} feeds done", flush=True)
    partial_file = open(partial, "a" if args.resume else "w")
    pending = [feed for feed in corpus if feed["feed_id"] not in done]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    llm = LLM(model=model_dir, dtype="auto", gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_model_len, enable_prefix_caching=True, enforce_eager=False)

    started = time.time()
    for start in range(0, len(pending), args.batch):
        batch = pending[start:start + args.batch]
        for feed, context_margins in zip(batch, vllm_context_margins(llm, tokenizer, batch)):
            row = {**feed, **summarize_feed(context_margins)}
            done[feed["feed_id"]] = row
            partial_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        partial_file.flush()
        n = start + len(batch)
        elapsed = time.time() - started
        print(f"  {n}/{len(pending)} feeds  {elapsed / n:.1f}s/feed", flush=True)

    rows = [done[feed["feed_id"]] for feed in corpus]
    with open(args.out, "w") as f:
        json.dump({"model_tag": tag, "model": model_dir, "n_templates": len(TEMPLATES),
                   "n_attack": n_attack, "n_ordinary_report": len(corpus) - n_attack,
                   "rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

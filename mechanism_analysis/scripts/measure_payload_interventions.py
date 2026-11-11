#!/usr/bin/env python3
"""Payload deletion and transplantation for one model.

The payload of an attack feed is the post with the largest belief pull,
i* = argmax_i delta_i, taken from the natural-corpus measurement of the same
model. The script builds these feed variants and reads each of them again with
the same readout (the original feeds included):

    attack_original           the attack feed
    attack_payload_deleted    the attack feed without post i*
    attack_random_deleted     the attack feed without one random other post
    report_original           an ordinary-report feed
    report_payload_inserted   the same feed with the payload of a same-event
                              attack feed inserted at a random position

Attack feeds and ordinary-report feeds need at least four posts. Ordinary-report
feeds are the shuffled-order feeds of the natural corpus.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
SEED = 7


def build_variants(rows, seed=SEED):
    """Feed variants from the natural-corpus rows of one model. Returns (variants, counts)."""
    rng = random.Random(seed)
    rows = sorted(rows, key=lambda r: r["feed_id"])
    attacks = [r for r in rows if r["feed_type"] == "attack" and len(r["posts"]) >= 4]
    reports = [r for r in rows if r["feed_type"] == "ordinary_report"
               and r["construction"] == "shuffled" and len(r["posts"]) >= 4]

    def variant(kind, r, posts, **meta):
        return {"kind": kind, "feed_id": r["feed_id"], "claim": r["claim"], "event": r["event"],
                "event_display": r["event_display"], "posts": posts, **meta}

    out, payloads_by_event = [], {}
    for r in attacks:
        payload_idx = int(np.argmax(r["pulls"]))             # first index on ties
        payloads_by_event.setdefault(r["event"], []).append(r["posts"][payload_idx])
        random_idx = rng.choice([j for j in range(len(r["posts"])) if j != payload_idx])
        out.append(variant("attack_original", r, list(r["posts"])))
        out.append(variant("attack_payload_deleted", r,
                           [p for j, p in enumerate(r["posts"]) if j != payload_idx],
                           payload_idx=payload_idx))
        out.append(variant("attack_random_deleted", r,
                           [p for j, p in enumerate(r["posts"]) if j != random_idx],
                           deleted_idx=random_idx))
    skipped, n_reports = 0, 0
    for r in reports:
        pool = payloads_by_event.get(r["event"])
        if not pool:
            skipped += 1
            continue
        payload = rng.choice(pool)
        position = rng.randrange(len(r["posts"]) + 1)
        out.append(variant("report_original", r, list(r["posts"])))
        out.append(variant("report_payload_inserted", r,
                           r["posts"][:position] + [payload] + r["posts"][position:],
                           insert_position=position, payload_post=payload))
        n_reports += 1
    return out, {"n_attack": len(attacks), "n_ordinary_report": n_reports,
                 "n_ordinary_report_without_same_event_payload": skipped}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--natural-corpus", required=True,
                    help="output of measure_natural_corpus.py for this model")
    ap.add_argument("--model", default=None,
                    help="panel tag or local path (default: the model of --natural-corpus)")
    ap.add_argument("--limit", type=int, default=0,
                    help="first N attack feeds and first N ordinary-report feeds only")
    ap.add_argument("--batch", type=int, default=32, help="feed variants for each vLLM call")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--kv-cache-gib", type=float, default=None,
                    help="fixed KV cache size; use it on a GPU shared with other jobs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from binary_readout import TEMPLATES, vllm_context_margins
    from measure_natural_corpus import summarize_feed
    from panel_models import resolve_model
    from transformers import AutoTokenizer
    from vllm import LLM

    with open(args.natural_corpus) as f:
        natural = json.load(f)
    tag = natural["model_tag"]
    model_dir = resolve_model(args.model) if args.model else natural["model"]

    variants, counts = build_variants(natural["rows"])
    if args.limit:
        keep = set(sorted({v["feed_id"] for v in variants
                           if v["kind"] == "attack_original"})[: args.limit])
        keep |= set(sorted({v["feed_id"] for v in variants
                            if v["kind"] == "report_original"})[: args.limit])
        variants = [v for v in variants if v["feed_id"] in keep]
    for v in variants:
        v["variant_id"] = f'{v["kind"]}::{v["feed_id"]}'
    print(f"[{tag}] {len(variants)} feed variants  {counts}", flush=True)

    partial = args.out + ".partial"
    done = {}
    if args.resume and os.path.exists(partial):
        with open(partial) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    done[row["variant_id"]] = row
        print(f"resume: {len(done)} variants done", flush=True)
    partial_file = open(partial, "a" if args.resume else "w")
    pending = [v for v in variants if v["variant_id"] not in done]

    engine_extra = {}
    if args.kv_cache_gib:
        engine_extra["kv_cache_memory_bytes"] = int(args.kv_cache_gib * 2**30)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    llm = LLM(model=model_dir, dtype="auto", gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_model_len, enable_prefix_caching=True, enforce_eager=False,
              **engine_extra)

    started = time.time()
    for start in range(0, len(pending), args.batch):
        batch = pending[start:start + args.batch]
        for v, context_margins in zip(batch, vllm_context_margins(llm, tokenizer, batch)):
            row = {**{k: v[k] for k in v if k != "posts"}, "n_posts": len(v["posts"]),
                   **summarize_feed(context_margins)}
            done[v["variant_id"]] = row
            partial_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        partial_file.flush()
        n = start + len(batch)
        elapsed = time.time() - started
        print(f"  {n}/{len(pending)} variants  {elapsed / n:.1f}s/variant", flush=True)

    rows = [done[v["variant_id"]] for v in variants]
    with open(args.out, "w") as f:
        json.dump({"model_tag": tag, "model": model_dir, "natural_corpus": args.natural_corpus,
                   "limit": args.limit, "n_templates": len(TEMPLATES), "counts": counts,
                   "rows": rows}, f, indent=1, ensure_ascii=False)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

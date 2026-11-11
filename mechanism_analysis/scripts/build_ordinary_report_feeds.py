#!/usr/bin/env python3
"""Build the ordinary-report feeds of the natural corpus.

For each event, the script builds feeds of non-rumour and verified-true posts
about a true claim. The number of posts in each feed is sampled from the
accepted attack feeds of the same event. The script needs no model.
"""
import argparse
import json
import os
import random
import sys
from dataclasses import asdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from feeds import EVENT_DISPLAY, build_control_feeds, load_attack_feeds  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--attacks-glob", required=True)
    ap.add_argument("--selection", required=True, help="output of select_attack_feeds.py")
    ap.add_argument("--copheme-dir", default=None,
                    help="default: ../lying_with_truth/CoPHEME in this repository")
    ap.add_argument("--per-event", type=int, default=70,
                    help="requested shuffled-order feeds for each event")
    ap.add_argument("--exclude-events", nargs="*", default=["putinmissing"])
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    with open(args.selection) as f:
        selection = json.load(f)
    accepted_ids = {r["feed_id"] for r in selection["rows"]
                    if r["accepted"] and not r["generator_artifact"]}
    accepted_by_event = {}
    for feed in load_attack_feeds(args.attacks_glob, drop_generator_artifacts=True):
        if feed.feed_id in accepted_ids:
            accepted_by_event.setdefault(feed.event, []).append(feed)

    events = [e for e in (sorted(accepted_by_event) or sorted(EVENT_DISPLAY))
              if e not in set(args.exclude_events)]
    out = {"seed": args.seed, "excluded_events": args.exclude_events,
           "ordinary_report_shuffled": [], "ordinary_report_ranked": [],
           "order_shuffled_attack": []}
    for event in events:
        built = build_control_feeds(event, accepted_by_event.get(event, []), rng,
                                    n_shuffled=args.per_event, copheme_dir=args.copheme_dir)
        for key, feeds in built.items():
            out[key] += [asdict(feed) for feed in feeds]
        print(f"{event:20s}: " + ", ".join(f"{k}={len(v)}" for k, v in built.items()))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"ordinary-report feeds: {len(out['ordinary_report_shuffled'])} shuffled, "
          f"{len(out['ordinary_report_ranked'])} ranked")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

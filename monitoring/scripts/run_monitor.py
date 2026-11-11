"""Run the GC monitor over the ORIG or ANON feed set (needs OPENAI_API_KEY).

    python scripts/run_monitor.py --setting orig   # 158 attack, 80 control, 158 chrono
    python scripts/run_monitor.py --setting anon   # 158 attack, 80 control
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from _common import OUTPUTS
from gcmonitor import feeds, monitor


def run_one(feed):
    from openai import OpenAI
    return monitor.monitor_feed(feed, OpenAI())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["orig", "anon"], required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    fs = feeds.load(args.setting)
    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(run_one, fs))
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(OUTPUTS / f"monitor_{args.setting}.json", "w"), indent=1)
    print(f"{len(rows)} feeds -> {OUTPUTS / f'monitor_{args.setting}.json'}")


if __name__ == "__main__":
    main()

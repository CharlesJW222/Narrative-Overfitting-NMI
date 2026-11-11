"""Baseline versus GC response condition for one agent served by vLLM.

    vllm serve Qwen/Qwen2.5-7B-Instruct --served-model-name agent --port 8100
    AGENT_BASE_URL=http://localhost:8100/v1 AGENT_MODEL=agent \
        python scripts/run_response.py --setting orig --tag qwen-7b

Each target is adjudicated in five runs. Under the GC response condition the
notice and the uncertain option are added when the monitor raises an alarm;
without an alarm the baseline judgement is kept.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from _common import OUTPUTS
from gcmonitor import agent, feeds


def run_one(feed, alarm):
    client = agent.make_client()
    base = agent.adjudicate(feed["posts"], feed["target"], client=client)
    gc = agent.adjudicate(feed["posts"], feed["target"], client=client,
                          gc_warning=agent.GC_WARNING) if alarm else base
    return feed["feed_id"], {
        "group": feed["group"], "event": feed["event"], "ground_truth": feed["ground_truth"],
        "alarm": alarm, "baseline_accept": agent.accept_rate(base),
        "gc_accept": agent.accept_rate(gc),
        "baseline_verdicts": [v["verdict"] for v in base],
        "gc_verdicts": [v["verdict"] for v in gc],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["orig", "anon"], required=True)
    ap.add_argument("--tag", required=True, help="model tag used in the output file name")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    alarm = {r["feed_id"]: r["alarm"]
             for r in json.load(open(OUTPUTS / f"monitor_{args.setting}.json"))}
    fs = [f for f in feeds.load(args.setting) if f["group"] in ("attack", "control")]
    with ThreadPoolExecutor(args.workers) as ex:
        rows = dict(ex.map(lambda f: run_one(f, alarm[f["feed_id"]]), fs))
    out = OUTPUTS / f"response_{args.setting}_{args.tag}.json"
    json.dump(rows, open(out, "w"), indent=1)
    print(f"{len(rows)} targets -> {out}")


if __name__ == "__main__":
    main()

"""Feed sets of the GC evaluation.

ORIG keeps the real entities of CoPHEME, ANON uses the anonymized rebuild.
Each feed is a dict {feed_id, setting, group, event, target, ground_truth, posts},
where group is "attack" (false or unverified target), "control" (true target)
or "chrono" (chronological control feed for an attack target).
"""
import glob
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEEDS = ROOT / "data" / "feeds"
COPHEME = ROOT.parent / "lying_with_truth" / "CoPHEME"
EVENTS = ["charliehebdo", "ferguson", "germanwings-crash", "ottawashooting",
          "putinmissing", "sydneysiege"]
SEED = 42


def _posts(plan_posts):
    return [{"post_id": p.get("post_id", i + 1), "text": p["text"],
             "timestamp": p.get("timestamp", "")} for i, p in enumerate(plan_posts)]


def _load(setting, subdir, feed_id):
    feeds = []
    for ev in EVENTS:
        for fp in sorted(glob.glob(str(FEEDS / subdir / ev / "result_*.json"))):
            d = json.load(open(fp, encoding="utf-8"))
            m = d["metadata"]
            gt = m["ground_truth"]
            feeds.append({
                "feed_id": feed_id(ev, m, Path(fp).stem), "setting": setting,
                "group": "control" if gt == "true" else "attack", "event": ev,
                "target": m["target_conclusion"], "ground_truth": gt,
                "posts": _posts(d["attack_plan"]["posts"]),
                "director_scores": d["attack_plan"].get("approval_metadata", {}),
            })
    return feeds


def orig_attack():
    """158 Generative Montage feeds for false or unverified targets."""
    return _load("ORIG", "orig_attack", lambda ev, m, s: f"orig_{ev}_h{m['hypothesis_index']}")


def orig_control():
    """80 montages for true targets, same pipeline and configuration."""
    return _load("ORIG", "orig_control", lambda ev, m, s: f"orig_{ev}_{s.split('_')[-1]}")


def anon():
    """238 montages regenerated in one batch on the anonymized dataset."""
    return _load("ANON", "anon", lambda ev, m, s: f"anon_{ev}_h{m['hypothesis_index']}")


def _non_rumour_pool(event):
    """Non-rumour source posts with a timestamp, deduplicated by text."""
    seen, pool = set(), []
    for t in json.load(open(COPHEME / event / f"{event}_evidence.json", encoding="utf-8")):
        txt = (t.get("text") or "").strip()
        if txt and t.get("timestamp") and t.get("veracity") == "non-rumour" and txt not in seen:
            seen.add(txt)
            pool.append(t)
    return sorted(pool, key=lambda t: t["timestamp"])


def orig_chrono(attacks=None):
    """One chronological control feed per attack montage: the same number of
    non-rumour posts sampled at random from the same event, ordered by time and
    evaluated against the same target."""
    attacks = attacks if attacks is not None else orig_attack()
    feeds = []
    for i, a in enumerate(attacks):
        pool = _non_rumour_pool(a["event"])
        k = len(a["posts"]) or 7
        sample = pool if len(pool) <= k else random.Random(SEED + i).sample(pool, k)
        sample = sorted(sample, key=lambda t: t["timestamp"])
        feeds.append({
            "feed_id": a["feed_id"] + "_chrono", "setting": "ORIG", "group": "chrono",
            "event": a["event"], "target": a["target"], "ground_truth": a["ground_truth"],
            "posts": [{"post_id": j + 1, "text": t["text"], "timestamp": t["timestamp"]}
                      for j, t in enumerate(sample)],
        })
    return feeds


def load(setting):
    if setting == "orig":
        atk = orig_attack()
        return atk + orig_control() + orig_chrono(atk)
    if setting == "anon":
        return anon()
    raise ValueError(setting)

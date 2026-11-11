"""Reproduce the GC numbers reported in the paper from data/outputs (no API calls).

    python scripts/analyze.py
"""
import json
import statistics

from _common import OUTPUTS
from gcmonitor import feeds

MODELS = ["qwen-7b", "qwen-14b", "ds-qwen-7b", "ds-qwen-14b"]


def auroc(pos, neg):
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def mean(xs):
    return sum(xs) / len(xs)


def load(name):
    return json.load(open(OUTPUTS / name))


def target_status(rows, setting):
    """Non-rumour control montages (positive class) versus attack montages."""
    ctl = [r for r in rows if r["group"] == "control"]
    atk = [r for r in rows if r["group"] == "attack"]
    print(f"\n[{setting}] control (n={len(ctl)}) vs attack (n={len(atk)})")
    for name, f in [("mean salience", lambda r: r["salience"]),
                    ("mean grounding posts per feed", lambda r: r["n_grounds"]),
                    ("feeds with grounding (%)", lambda r: 100 * r["grounded"])]:
        c, a = [f(r) for r in ctl], [f(r) for r in atk]
        print(f"  {name:30s} {mean(c):7.3f} vs {mean(a):7.3f}   AUROC {auroc(c, a):.3f}")
    print(f"  alarm rate on controls: {100 * mean([r['alarm'] for r in ctl]):.1f}%")


def detection(rows):
    """ORIG attack montages versus chronological control feeds."""
    atk = [r for r in rows if r["group"] == "attack"]
    chr_ = [r for r in rows if r["group"] == "chrono"]
    tp = sum(r["alarm"] for r in atk)
    fp = sum(r["alarm"] for r in chr_)
    prec, rec = tp / (tp + fp), tp / len(atk)
    crossing = sorted(r["threshold_crossing"] for r in atk if r["threshold_crossing"])
    print(f"\n[ORIG] attack (n={len(atk)}) vs chronological control (n={len(chr_)})")
    print(f"  mean salience {mean([r['salience'] for r in atk]):.3f} vs "
          f"{mean([r['salience'] for r in chr_]):.3f}   AUROC "
          f"{auroc([r['salience'] for r in atk], [r['salience'] for r in chr_]):.3f}")
    print(f"  alarms {tp} vs {fp}: recall {rec:.3f}, FPR {fp / len(chr_):.3f}, "
          f"precision {prec:.3f}, F1 {2 * prec * rec / (prec + rec):.3f}")
    print(f"  median threshold crossing: post {crossing[len(crossing) // 2]}")


def rates(resp, group, key):
    return round(mean([r[key] for r in resp.values() if r["group"] == group]), 1)


def decision_accuracy():
    """Decision accuracy (158(1 - a_R) + 80 a_N) / 238 from group rates rounded to 0.1."""
    print("\n[Decision accuracy] baseline -> GC (rumour acceptance; control acceptance)")
    for setting in ("orig", "anon"):
        for m in MODELS:
            resp = load(f"response_{setting}_{m}.json")
            line = []
            for key in ("baseline_accept", "gc_accept"):
                a_r, a_n = rates(resp, "attack", key), rates(resp, "control", key)
                line.append((round((158 * (100 - a_r) + 80 * a_n) / 238, 1), a_r, a_n))
            (b, br, bn), (g, gr, gn) = line
            print(f"  {setting.upper()} {m:12s} {b:5.1f} -> {g:5.1f} ({g - b:+.1f})"
                  f"   rumour {br:.1f}->{gr:.1f}; control {bn:.1f}->{gn:.1f}")


def entity_controls():
    print("\n[Entity controls] baseline acceptance, ORIG -> ANON")
    for m in MODELS:
        o, a = load(f"response_orig_{m}.json"), load(f"response_anon_{m}.json")
        ro, ra = rates(o, "attack", "baseline_accept"), rates(a, "attack", "baseline_accept")
        co, ca = rates(o, "control", "baseline_accept"), rates(a, "control", "baseline_accept")
        print(f"  {m:12s} rumour {ro:.1f} -> {ra:.1f} ({ra - ro:+.1f}); "
              f"control {co:.1f} -> {ca:.1f} ({ca - co:+.1f})")
    p = load("target_only_probe.json")
    for s in ("orig", "anon"):
        v = p[s]["veracity"]
        acc = (v["true"][0] + v["false"][0]) / (v["true"][1] + v["false"][1])
        print(f"  target-only probe {s.upper()}: recognition {100 * p[s]['recognized'] / p[s]['n']:.1f}%, "
              f"veracity accuracy on true and false targets {100 * acc:.1f}%")
    for name, fs in (("ORIG", feeds.orig_attack()),
                     ("ANON", [f for f in feeds.anon() if f["group"] == "attack"])):
        if not fs:
            print(f"  Director scores {name}: no montages in data/feeds")
            continue
        d = [f["director_scores"] for f in fs]
        print(f"  Director scores {name} (n={len(d)}): narrative "
              f"{statistics.mean(x['narrative_score'] for x in d):.2f}, sequence "
              f"{statistics.mean(x['sequence_score'] for x in d):.2f}")


def target_types(rows):
    types = load("target_types.json")
    print("\n[ORIG target types] grounding by type")
    for t in ("atomic", "inferential"):
        ctl = [r for r in rows if r["group"] == "control" and types[r["feed_id"]] == t]
        atk = [r for r in rows if r["group"] == "attack" and types[r["feed_id"]] == t]
        print(f"  {t:12s} n control {len(ctl)}, rumour {len(atk)}; grounding "
              f"{100 * mean([r['grounded'] for r in ctl]):.1f}% vs "
              f"{100 * mean([r['grounded'] for r in atk]):.1f}%; AUROC "
              f"{auroc([r['n_grounds'] for r in ctl], [r['n_grounds'] for r in atk]):.3f}")


def main():
    orig, anon = load("monitor_orig.json"), load("monitor_anon.json")
    target_status(orig, "ORIG")
    target_status(anon, "ANON")
    detection(orig)
    decision_accuracy()
    entity_controls()
    target_types(orig)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Statistics of the payload interventions.

For z in {m, u} and an attack feed F with payload index i* and random index r:

    Delta z(F) = [z(F without i*) - z(F)] - [z(F without r) - z(F)]

Per model:
    delta_margin_payload_vs_random        mean Delta m over attack feeds (primary)
    verdict_flip                          share of attack feeds that change from
                                          m(F) > 0 to m <= 0 after each deletion;
                                          the denominator is all attack feeds
    delta_dispersion_payload_vs_random    mean Delta u (descriptive)
    insertion_delta_dispersion / _margin  mean change of u and m in ordinary-report
                                          feeds after payload insertion (descriptive)
    corr_delta_margin_delta_dispersion    Pearson correlation of m(F without i*) - m(F)
                                          with u(F without i*) - u(F) (descriptive)

The 95% intervals use 3,000 claim-clustered bootstrap resamples (seed 0). The
cross-model mean averages the per-model estimates within each resample of claims
(descriptive).
"""
import argparse
import json

import numpy as np

N_BOOT, SEED = 3000, 0


def _by(rows, kind):
    return {r["feed_id"]: r for r in rows if r["kind"] == kind}


def paired_values(rows):
    """Per-feed contrasts keyed by feed id: (attack feeds, ordinary-report feeds)."""
    orig, payload_del, random_del = (_by(rows, "attack_original"),
                                     _by(rows, "attack_payload_deleted"),
                                     _by(rows, "attack_random_deleted"))
    rep, rep_ins = _by(rows, "report_original"), _by(rows, "report_payload_inserted")
    attack = {}
    for i in sorted(set(orig) & set(payload_del) & set(random_del)):
        o, p, r = orig[i], payload_del[i], random_del[i]
        attack[i] = {"claim": o["claim"],
                     "delta_margin": p["margin"] - r["margin"],
                     "delta_dispersion": p["dispersion"] - r["dispersion"],
                     "payload_margin_change": p["margin"] - o["margin"],
                     "payload_dispersion_change": p["dispersion"] - o["dispersion"],
                     "flip_payload": float(o["margin"] > 0 and p["margin"] <= 0),
                     "flip_random": float(o["margin"] > 0 and r["margin"] <= 0),
                     "original_true": float(o["margin"] > 0),
                     "margin_original": o["margin"], "margin_payload_deleted": p["margin"],
                     "margin_random_deleted": r["margin"],
                     "dispersion_original": o["dispersion"],
                     "dispersion_payload_deleted": p["dispersion"],
                     "dispersion_random_deleted": r["dispersion"]}
    report = {}
    for i in sorted(set(rep) & set(rep_ins)):
        o, t = rep[i], rep_ins[i]
        report[i] = {"claim": o["claim"],
                     "insertion_delta_dispersion": t["dispersion"] - o["dispersion"],
                     "insertion_delta_margin": t["margin"] - o["margin"],
                     "margin_original": o["margin"], "margin_payload_inserted": t["margin"],
                     "dispersion_original": o["dispersion"],
                     "dispersion_payload_inserted": t["dispersion"]}
    return attack, report


def _claim_groups(values):
    groups = {}
    for v in values.values():
        groups.setdefault(v["claim"], []).append(v)
    return [groups[c] for c in sorted(groups)]


def _mean(groups, key, idx=None):
    idx = range(len(groups)) if idx is None else idx
    return float(np.mean([v[key] for k in idx for v in groups[k]]))


def _corr(groups, idx=None):
    idx = range(len(groups)) if idx is None else idx
    sel = [v for k in idx for v in groups[k]]
    if len(sel) < 3:
        return float("nan")
    return float(np.corrcoef([v["payload_margin_change"] for v in sel],
                             [v["payload_dispersion_change"] for v in sel])[0, 1])


def _resamples(n_groups, n_boot, seed):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n_groups, n_groups) for _ in range(n_boot)]


def _interval(point, boots, n_claims):
    b = np.array([x for x in boots if np.isfinite(x)])
    ci = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if b.size else [float("nan")] * 2
    return {"mean": point, "ci95": ci, "n_claims": n_claims}


def model_statistics(rows, n_boot=N_BOOT, seed=SEED):
    attack, report = paired_values(rows)
    ga, gr = _claim_groups(attack), _claim_groups(report)
    ia, ir = _resamples(len(ga), n_boot, seed), _resamples(len(gr), n_boot, seed)
    st = {}
    for name, key in [("delta_margin_payload_vs_random", "delta_margin"),
                      ("delta_dispersion_payload_vs_random", "delta_dispersion")]:
        st[name] = _interval(_mean(ga, key), [_mean(ga, key, i) for i in ia], len(ga))
        st[name]["n_feeds"] = len(attack)
    for key in ("insertion_delta_dispersion", "insertion_delta_margin"):
        st[key] = _interval(_mean(gr, key), [_mean(gr, key, i) for i in ir], len(gr))
        st[key]["n_feeds"] = len(report)
    st["corr_delta_margin_delta_dispersion"] = _interval(
        _corr(ga), [_corr(ga, i) for i in ia], len(ga))
    st["verdict_flip"] = {
        "payload_deleted": _mean(ga, "flip_payload"),
        "random_deleted": _mean(ga, "flip_random"),
        "original_true_share": _mean(ga, "original_true"),
        "payload_deleted_ci95": _interval(0, [_mean(ga, "flip_payload", i) for i in ia], len(ga))["ci95"],
        "random_deleted_ci95": _interval(0, [_mean(ga, "flip_random", i) for i in ia], len(ga))["ci95"]}
    st["condition_means"] = {
        "attack": {k: _mean(ga, k) for k in (
            "margin_original", "margin_payload_deleted", "margin_random_deleted",
            "dispersion_original", "dispersion_payload_deleted", "dispersion_random_deleted")},
        "ordinary_report": {k: _mean(gr, k) for k in (
            "margin_original", "margin_payload_inserted",
            "dispersion_original", "dispersion_payload_inserted")},
        "n_attack": len(attack), "n_ordinary_report": len(report)}
    # Decision rule: the interval of the primary estimand lies below zero.
    st["delta_margin_interval_below_zero"] = bool(st["delta_margin_payload_vs_random"]["ci95"][1] < 0)
    return st


def cross_model_mean(rows_by_model, n_boot=N_BOOT, seed=SEED):
    """Mean of the per-model estimates; all models share each resample of claims."""
    per = {t: paired_values(r) for t, r in rows_by_model.items()}
    claims_a = sorted({v["claim"] for a, _ in per.values() for v in a.values()})
    claims_r = sorted({v["claim"] for _, b in per.values() for v in b.values()})
    rng = np.random.default_rng(seed)
    attack_keys = ("delta_margin", "delta_dispersion", "flip_payload", "flip_random")
    report_keys = ("insertion_delta_dispersion", "insertion_delta_margin")

    def stat(sample_a, sample_r):
        out = {k: [] for k in attack_keys + report_keys}
        for a, b in per.values():
            for key in attack_keys:
                out[key].append(np.mean([v[key] for c in sample_a for v in a.values()
                                         if v["claim"] == c]))
            for key in report_keys:
                out[key].append(np.mean([v[key] for c in sample_r for v in b.values()
                                         if v["claim"] == c]))
        return {k: float(np.mean(v)) for k, v in out.items()}

    point = stat(claims_a, claims_r)
    boots = [stat(rng.choice(claims_a, len(claims_a)), rng.choice(claims_r, len(claims_r)))
             for _ in range(n_boot)]
    return {k: {"mean": point[k], "ci95": [float(np.percentile([b[k] for b in boots], 2.5)),
                                           float(np.percentile([b[k] for b in boots], 97.5))]}
            for k in point}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("measurements", nargs="+", help="outputs of measure_payload_interventions.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows_by_model, per = {}, {}
    for path in args.measurements:
        with open(path) as f:
            d = json.load(f)
        assert not d["limit"], f"{path}: measured with --limit"
        tag = d["model_tag"]
        rows_by_model[tag] = d["rows"]
        per[tag] = {"counts": d["counts"], **model_statistics(d["rows"])}
        e = per[tag]["delta_margin_payload_vs_random"]
        flip = per[tag]["verdict_flip"]
        print(f"{tag:32s} Delta m {e['mean']:+.2f} [{e['ci95'][0]:+.2f},{e['ci95'][1]:+.2f}]  "
              f"flips {flip['payload_deleted']:.3f} vs {flip['random_deleted']:.3f}", flush=True)
    summary = {"n_models": len(per),
               "n_models_delta_margin_interval_below_zero":
                   sum(p["delta_margin_interval_below_zero"] for p in per.values()),
               "cross_model_mean": cross_model_mean(rows_by_model)}
    with open(args.out, "w") as f:
        json.dump({"n_boot": N_BOOT, "seed": SEED, "models": per, "summary": summary}, f, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

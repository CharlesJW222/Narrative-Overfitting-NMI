#!/usr/bin/env python3
"""Separation of attack feeds from ordinary-report feeds on the natural corpus.

Per model: AUROC of the full-feed belief margin m(F) and of the response
dispersion u(F), with attack feeds as the positive class. The 95% intervals use
3,000 claim-clustered bootstrap resamples. The dispersion AUROC is also given
for each refutation class of the attack targets.

Pooled comparison: each model's scores are converted to within-model percentile
ranks, and the ranks are averaged across models for each feed before the AUROC.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from dispersion import auroc, claim_clustered_auroc  # noqa: E402

STATS = ("margin", "dispersion")
N_BOOT, SEED = 3000, 0


def _entry(rows, stat):
    point, lo, hi = claim_clustered_auroc(rows, stat, np.random.default_rng(SEED), N_BOOT)
    return {"auroc": point, "ci95": [lo, hi]}


def model_summary(rows):
    out = {"n_attack": sum(1 for r in rows if r["feed_type"] == "attack"),
           "n_ordinary_report": sum(1 for r in rows if r["feed_type"] == "ordinary_report")}
    for stat in STATS:
        out[stat] = _entry(rows, stat)
    out["dispersion_by_refutation_class"] = {}
    classes = sorted({r["refutation_class"] for r in rows
                      if r["feed_type"] == "attack" and r.get("refutation_class")})
    for cls in classes:
        subset = [r for r in rows
                  if r["feed_type"] == "ordinary_report" or r["refutation_class"] == cls]
        out["dispersion_by_refutation_class"][cls] = {
            "n_attack": sum(1 for r in subset if r["feed_type"] == "attack"),
            **_entry(subset, "dispersion")}
    return out


def pooled_summary(rows_by_model):
    """AUROC of the across-model mean of within-model percentile ranks."""
    reference, ranks = None, {}
    for tag, rows in rows_by_model.items():
        by_id = {r["feed_id"]: r for r in rows}
        if len(by_id) != len(rows):
            raise ValueError(f"duplicate feed_id in {tag}")
        if reference is None:
            reference = by_id
        elif set(by_id) != set(reference):
            raise ValueError(f"{tag} was measured on a different set of feeds")
        ranks[tag] = {}
        for stat in STATS:
            values = np.asarray([r[stat] for r in rows], dtype=float)
            percentiles = (rankdata(values, method="average") - 0.5) / len(values)
            ranks[tag][stat] = {r["feed_id"]: float(p) for r, p in zip(rows, percentiles)}
    out = {"n_models": len(rows_by_model), "n_feeds": len(reference)}
    for stat in STATS:
        pos, neg = [], []
        for feed_id, row in reference.items():
            # Exact summation: the mean rank does not depend on the order of the models.
            mean_rank = math.fsum(ranks[tag][stat][feed_id] for tag in rows_by_model) / len(rows_by_model)
            (pos if row["feed_type"] == "attack" else neg).append(mean_rank)
        out[stat] = {"auroc": float(auroc(pos, neg))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("measurements", nargs="+", help="outputs of measure_natural_corpus.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    rows_by_model = {}
    for path in args.measurements:
        with open(path) as f:
            d = json.load(f)
        rows_by_model[d["model_tag"]] = d["rows"]
    result = {"n_boot": N_BOOT, "seed": SEED,
              "models": {tag: model_summary(rows) for tag, rows in rows_by_model.items()},
              "pooled": pooled_summary(rows_by_model)}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    for tag, s in result["models"].items():
        print(f"{tag:32s} margin {s['margin']['auroc']:.3f}  dispersion {s['dispersion']['auroc']:.3f}")
    print(f"{'pooled':32s} margin {result['pooled']['margin']['auroc']:.3f}  "
          f"dispersion {result['pooled']['dispersion']['auroc']:.3f}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

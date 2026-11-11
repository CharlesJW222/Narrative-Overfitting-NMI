#!/usr/bin/env python3
"""Dispersion contrasts between the conditions of the synthetic testbed.

Construction contrasts: AUROC of the response dispersion u between two
conditions, with the first condition as the positive class.

    C_s vs C    scorer selection, short fragments
    N vs C      narrative writing
    N_s vs N    scorer selection, narratives
    A vs N_s    attacker search beyond the scorer
    A vs N      attack against random-evidence narrative
    A vs C      attack against short-fragment control

Supported narratives (S): acceptance (share of reads with full-feed margin
above zero) by target type, the share of S, N and A feeds above the 90th
percentile of the dispersion of C, and the contrasts A vs S, N vs S, S vs C.

The unit is one (world, condition, template) read. Reads without a defined
dispersion (fewer than two fragments) are excluded. The 95% intervals use
2,000 world-clustered bootstrap resamples (seed 0).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from dispersion import world_clustered_auroc  # noqa: E402

N_BOOT, SEED = 2000, 0
CONSTRUCTION_CONTRASTS = (("C_s", "C"), ("N", "C"), ("N_s", "N"), ("A", "N_s"), ("A", "N"), ("A", "C"))
SUPPORTED_CONTRASTS = (("A", "S"), ("N", "S"), ("S", "C"))
FLAG_QUANTILE = 0.90


def contrast(rows, pos, neg, *, n_boot=N_BOOT, seed=SEED):
    pos_by_world, neg_by_world = {}, {}
    for row in rows:
        if row["dispersion"] is None:
            continue
        if row["condition"] == pos:
            pos_by_world.setdefault(row["world_id"], []).append(row["dispersion"])
        elif row["condition"] == neg:
            neg_by_world.setdefault(row["world_id"], []).append(row["dispersion"])
    return world_clustered_auroc(pos_by_world, neg_by_world, n_boot, seed)


def _contrasts(rows, pairs, n_boot, seed):
    conditions = {r["condition"] for r in rows}
    return {f"{pos}_vs_{neg}": contrast(rows, pos, neg, n_boot=n_boot, seed=seed)
            for pos, neg in pairs if pos in conditions and neg in conditions}


def supported_summary(rows, *, n_boot=N_BOOT, seed=SEED):
    def dispersions(condition):
        return np.asarray([r["dispersion"] for r in rows
                           if r["condition"] == condition and r["dispersion"] is not None])

    control = dispersions("C")
    if not len(control):
        raise ValueError("the dispersion threshold needs feeds of condition C")
    threshold = float(np.quantile(control, FLAG_QUANTILE, method="higher"))
    above = {c: float((dispersions(c) >= threshold).mean())
             for c in ("C", "S", "N", "N_s", "A") if len(dispersions(c))}
    supported = [r for r in rows if r["condition"] == "S" and r["dispersion"] is not None]
    acceptance = {"all": float(np.mean([r["margin"] > 0 for r in supported]))}
    for target_type in ("direct", "multi_step"):
        sel = [r for r in supported if r.get("target_type") == target_type]
        if sel:
            acceptance[target_type] = float(np.mean([r["margin"] > 0 for r in sel]))
    return {"n_reads": len(supported), "acceptance": acceptance,
            "threshold": threshold, "share_above_threshold": above,
            "contrasts": _contrasts(rows, SUPPORTED_CONTRASTS, n_boot, seed)}


def model_summary(rows, *, n_boot=N_BOOT, seed=SEED):
    conditions = sorted({r["condition"] for r in rows})
    out = {"n_reads": {c: sum(1 for r in rows if r["condition"] == c) for c in conditions},
           "construction_contrasts": _contrasts(rows, CONSTRUCTION_CONTRASTS, n_boot, seed)}
    if {"S", "C"} <= set(conditions):
        out["supported_narratives"] = supported_summary(rows, n_boot=n_boot, seed=seed)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("measurements", nargs="+", help="outputs of measure_synthetic_feeds.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    rows_by_model = {}
    for path in args.measurements:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows_by_model.setdefault(row["model"], []).append(row)
    result = {"n_boot": N_BOOT, "seed": SEED,
              "models": {tag: model_summary(rows) for tag, rows in sorted(rows_by_model.items())}}
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    for tag, summary in result["models"].items():
        for name, c in summary["construction_contrasts"].items():
            print(f"{tag:32s} {name:10s} AUROC {c['auroc']:.3f} [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}]")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

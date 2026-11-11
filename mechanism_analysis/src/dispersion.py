"""Response dispersion, AUROC and cluster-bootstrap confidence intervals."""
from __future__ import annotations

import math

import numpy as np


def belief_pulls(margin_empty: float, margins_isolated: list[float]) -> list[float]:
    """Belief pull of each post: delta_i = m({p_i}) - m(empty)."""
    return [float(m - margin_empty) for m in margins_isolated]


def response_dispersion(pulls: list[float]) -> float | None:
    """u(F): sample standard deviation (ddof=1) of the belief pulls.

    The dispersion is not defined for fewer than two posts; the result is None.
    """
    if len(pulls) < 2:
        return None
    return float(np.std(np.asarray(pulls, dtype=float), ddof=1))


def auroc(pos, neg) -> float:
    """Rank-based AUROC with mid-rank tie handling.

    Pr[pos > neg] + 0.5 * Pr[pos == neg].
    """
    xs = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg], key=lambda t: t[0])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1][0] == xs[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    n1, n0 = len(pos), len(neg)
    sp = sum(ranks[k] for k in range(len(xs)) if xs[k][1] == 1)
    return (sp - n1 * (n1 + 1) / 2) / (n1 * n0)


def claim_clustered_auroc(rows, stat, rng, n_boot=3000):
    """AUROC of `stat` (attack feeds vs ordinary-report feeds) with a 95% interval.

    The bootstrap resamples claims with replacement, independently in the two
    feed types. Returns (point, lower, upper).
    """
    def by_claim(feed_type):
        d = {}
        for row in rows:
            if row["feed_type"] == feed_type and math.isfinite(row[stat]):
                d.setdefault(row["claim"], []).append(row[stat])
        return d

    pos_c, neg_c = by_claim("attack"), by_claim("ordinary_report")
    pk, nk = list(pos_c), list(neg_c)
    if not pk or not nk:
        return None, None, None
    point = auroc([v for k in pk for v in pos_c[k]], [v for k in nk for v in neg_c[k]])
    boots = []
    for _ in range(n_boot):
        p = [v for k in rng.choice(pk, len(pk), replace=True) for v in pos_c[k]]
        n = [v for k in rng.choice(nk, len(nk), replace=True) for v in neg_c[k]]
        boots.append(auroc(p, n))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def world_clustered_auroc(pos_by_world: dict[str, list[float]],
                          neg_by_world: dict[str, list[float]],
                          n_boot: int, seed: int) -> dict:
    """AUROC between two synthetic conditions with a 95% interval.

    The bootstrap resamples worlds with replacement. A resampled world carries
    its feeds in both conditions.
    """
    pos = [u for values in pos_by_world.values() for u in values]
    neg = [u for values in neg_by_world.values() for u in values]
    if not pos or not neg:
        raise ValueError("one side of the contrast is empty")
    worlds = sorted(set(pos_by_world) | set(neg_by_world))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        picked = [worlds[j] for j in rng.integers(0, len(worlds), size=len(worlds))]
        p = [u for w in picked for u in pos_by_world.get(w, ())]
        n = [u for w in picked for u in neg_by_world.get(w, ())]
        if p and n:
            draws.append(float(auroc(p, n)))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"auroc": float(auroc(pos, neg)), "ci95": [float(lo), float(hi)],
            "n_pos": len(pos), "n_neg": len(neg), "n_worlds": len(worlds)}

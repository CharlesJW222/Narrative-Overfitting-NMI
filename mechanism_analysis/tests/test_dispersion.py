import numpy as np
import pytest

from dispersion import (auroc, belief_pulls, claim_clustered_auroc, response_dispersion,
                        world_clustered_auroc)


def test_pulls_and_dispersion_use_ddof1():
    pulls = belief_pulls(1.0, [2.0, 5.0, 1.0])
    assert pulls == [1.0, 4.0, 0.0]
    assert response_dispersion(pulls) == pytest.approx(np.std([1.0, 4.0, 0.0], ddof=1))


def test_dispersion_is_undefined_for_fewer_than_two_posts():
    assert response_dispersion([]) is None
    assert response_dispersion([3.0]) is None


def test_auroc_counts_ties_as_one_half():
    assert auroc([1.0, 2.0], [0.0, 0.5]) == 1.0
    assert auroc([1.0], [1.0]) == 0.5
    assert auroc([1.0, 1.0, 2.0], [1.0, 0.0]) == pytest.approx((0.5 + 1 + 0.5 + 1 + 1 + 1) / 6)


def _rows():
    rows = []
    for k in range(6):
        rows.append({"feed_type": "attack", "claim": f"a{k % 3}", "x": 2.0 + k})
        rows.append({"feed_type": "ordinary_report", "claim": f"r{k % 4}", "x": float(k)})
    return rows


def test_claim_clustered_interval_contains_the_point_and_is_deterministic():
    first = claim_clustered_auroc(_rows(), "x", np.random.default_rng(0), n_boot=200)
    second = claim_clustered_auroc(_rows(), "x", np.random.default_rng(0), n_boot=200)
    assert first == second
    point, lo, hi = first
    assert lo <= point <= hi


def test_world_clustered_contrast():
    pos = {"w1": [3.0, 4.0], "w2": [5.0]}
    neg = {"w1": [1.0], "w3": [2.0, 0.0]}
    first = world_clustered_auroc(pos, neg, 200, 0)
    assert first == world_clustered_auroc(pos, neg, 200, 0)
    assert first["auroc"] == 1.0 and first["n_worlds"] == 3
    assert first["n_pos"] == 3 and first["n_neg"] == 3
    with pytest.raises(ValueError):
        world_clustered_auroc(pos, {}, 10, 0)

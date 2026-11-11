import random

import numpy as np
import pytest

from analyze_payload_interventions import model_statistics
from measure_natural_corpus import summarize_feed
from measure_payload_interventions import build_variants


def _attack(feed_id, event, pulls, claim="c"):
    return {"feed_id": feed_id, "event": event, "event_display": event.upper(), "claim": claim,
            "feed_type": "attack", "construction": None,
            "posts": [f"{feed_id}-p{i}" for i in range(len(pulls))], "pulls": pulls}


def _report(feed_id, event, n, construction="shuffled"):
    return {"feed_id": feed_id, "event": event, "event_display": event.upper(), "claim": "t",
            "feed_type": "ordinary_report", "construction": construction,
            "posts": [f"{feed_id}-b{i}" for i in range(n)], "pulls": [0.0] * n}


ROWS = [
    _attack("attack::e1::a", "e1", [0.1, 3.0, 0.2, 3.0]),    # tie -> first maximum (index 1)
    _attack("attack::e1::b", "e1", [5.0, 0.0, 0.0, 1.0, 0.5]),
    _attack("attack::e2::c", "e2", [0.0, 0.0, 9.0]),         # fewer than 4 posts -> excluded
    _report("report::e1::0", "e1", 4),
    _report("report::e1::1", "e1", 3),                       # fewer than 4 posts -> excluded
    _report("report::e3::0", "e3", 5),                       # no same-event attack -> skipped
    _report("report_ranked::e1::0", "e1", 6, construction="ranked"),   # not shuffled -> excluded
]


def _by(variants, kind):
    return {v["feed_id"]: v for v in variants if v["kind"] == kind}


def test_payload_is_the_first_maximum_and_deletion_removes_it():
    variants, _ = build_variants(ROWS)
    deleted = _by(variants, "attack_payload_deleted")
    assert deleted["attack::e1::a"]["payload_idx"] == 1
    assert deleted["attack::e1::a"]["posts"] == [
        "attack::e1::a-p0", "attack::e1::a-p2", "attack::e1::a-p3"]
    assert deleted["attack::e1::b"]["payload_idx"] == 0


def test_random_deletion_removes_one_other_post():
    variants, _ = build_variants(ROWS)
    payload, rand = _by(variants, "attack_payload_deleted"), _by(variants, "attack_random_deleted")
    for feed_id, v in rand.items():
        assert v["deleted_idx"] != payload[feed_id]["payload_idx"]
        assert len(v["posts"]) == len(payload[feed_id]["posts"])


def test_filters_and_counts():
    variants, counts = build_variants(ROWS)
    assert {v["feed_id"] for v in variants} == {"attack::e1::a", "attack::e1::b", "report::e1::0"}
    assert counts == {"n_attack": 2, "n_ordinary_report": 1,
                      "n_ordinary_report_without_same_event_payload": 1}
    assert sorted(v["kind"] for v in variants) == sorted(
        ["attack_original", "attack_payload_deleted", "attack_random_deleted"] * 2
        + ["report_original", "report_payload_inserted"])


def test_insertion_uses_a_same_event_payload():
    variants, _ = build_variants(ROWS)
    v = _by(variants, "report_payload_inserted")["report::e1::0"]
    assert len(v["posts"]) == 5
    inserted = v["posts"][v["insert_position"]]
    assert inserted in {"attack::e1::a-p1", "attack::e1::b-p0"} and inserted == v["payload_post"]
    rest = v["posts"][:v["insert_position"]] + v["posts"][v["insert_position"] + 1:]
    assert rest == [f"report::e1::0-b{i}" for i in range(4)]


def test_row_order_does_not_change_the_variants():
    shuffled = ROWS[:]
    random.Random(3).shuffle(shuffled)
    assert build_variants(ROWS) == build_variants(shuffled)


def test_feed_summary():
    out = summarize_feed([1.0, 4.0, 2.0, 5.0, 1.0])   # empty, complete feed, three posts
    assert out["margin_empty"] == 1.0 and out["margin"] == 4.0
    assert out["pulls"] == [1.0, 4.0, 0.0]
    assert out["dispersion"] == pytest.approx(np.std([1.0, 4.0, 0.0], ddof=1))
    assert summarize_feed([1.0, 4.0, 2.0])["dispersion"] == 0.0


def _measured_rows():
    rows = []
    # Payload deletion lowers the margin by 4; random deletion lowers it by 1.
    for feed_id, claim, m0 in [("s1", "A", 2.0), ("s2", "A", 6.0), ("s3", "B", 3.0)]:
        base = {"feed_id": feed_id, "claim": claim}
        rows.append({**base, "kind": "attack_original", "margin": m0, "dispersion": 2.0})
        rows.append({**base, "kind": "attack_payload_deleted", "margin": m0 - 4.0, "dispersion": 1.0})
        rows.append({**base, "kind": "attack_random_deleted", "margin": m0 - 1.0, "dispersion": 2.0})
    for feed_id, claim in [("b1", "T1"), ("b2", "T2")]:
        base = {"feed_id": feed_id, "claim": claim}
        rows.append({**base, "kind": "report_original", "margin": 1.0, "dispersion": 1.0})
        rows.append({**base, "kind": "report_payload_inserted", "margin": 1.5, "dispersion": 1.5})
    return rows


def test_point_estimates():
    st = model_statistics(_measured_rows(), n_boot=200)
    assert st["delta_margin_payload_vs_random"]["mean"] == pytest.approx(-3.0)
    assert st["delta_dispersion_payload_vs_random"]["mean"] == pytest.approx(-1.0)
    assert st["insertion_delta_dispersion"]["mean"] == pytest.approx(0.5)
    assert st["insertion_delta_margin"]["mean"] == pytest.approx(0.5)
    # s1 (2 -> -2) and s3 (3 -> -1) flip after payload deletion; none after random deletion.
    assert st["verdict_flip"]["payload_deleted"] == pytest.approx(2 / 3)
    assert st["verdict_flip"]["random_deleted"] == pytest.approx(0.0)
    assert st["verdict_flip"]["original_true_share"] == pytest.approx(1.0)
    assert st["delta_margin_payload_vs_random"]["n_claims"] == 2
    lo, hi = st["delta_margin_payload_vs_random"]["ci95"]
    assert lo <= -3.0 <= hi
    assert st["delta_margin_interval_below_zero"] is True

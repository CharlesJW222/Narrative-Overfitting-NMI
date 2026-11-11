import json

import numpy as np
import pytest

import analyze_natural_corpus
import analyze_synthetic_feeds
from dispersion import auroc
from ternary_readout import TEMPLATES, margin_from_label_logprobs, render_user_prompt


def _natural_rows(shift):
    rows = []
    for k in range(8):
        rows.append({"feed_id": f"attack::{k}", "feed_type": "attack", "claim": f"a{k % 3}",
                     "refutation_class": "x" if k % 2 else "y",
                     "margin": 1.0 + k, "dispersion": shift + k})
        rows.append({"feed_id": f"report::{k}", "feed_type": "ordinary_report", "claim": f"r{k % 4}",
                     "refutation_class": None, "margin": float(k), "dispersion": float(k)})
    return rows


def test_natural_corpus_summary(tmp_path):
    paths = []
    for tag, shift in (("m1", 3.0), ("m2", 0.5)):
        path = tmp_path / f"{tag}.json"
        path.write_text(json.dumps({"model_tag": tag, "rows": _natural_rows(shift)}))
        paths.append(str(path))
    out = tmp_path / "summary.json"
    analyze_natural_corpus.main(paths + ["--out", str(out)])
    result = json.loads(out.read_text())

    rows = _natural_rows(3.0)
    expected = auroc([r["dispersion"] for r in rows if r["feed_type"] == "attack"],
                     [r["dispersion"] for r in rows if r["feed_type"] == "ordinary_report"])
    m1 = result["models"]["m1"]
    assert m1["dispersion"]["auroc"] == pytest.approx(expected)
    assert m1["n_attack"] == 8 and m1["n_ordinary_report"] == 8
    assert set(m1["dispersion_by_refutation_class"]) == {"x", "y"}
    assert m1["dispersion_by_refutation_class"]["x"]["n_attack"] == 4
    assert result["pooled"]["n_models"] == 2 and result["pooled"]["n_feeds"] == 16
    assert 0.5 < result["pooled"]["dispersion"]["auroc"] <= 1.0


def test_pooled_comparison_needs_the_same_feeds():
    rows = _natural_rows(1.0)
    with pytest.raises(ValueError):
        analyze_natural_corpus.pooled_summary({"m1": rows, "m2": rows[:-1]})


def _synthetic_rows():
    rng = np.random.default_rng(1)
    rows = []
    for w in range(12):
        for condition, centre in (("C", 0.0), ("C_s", 0.2), ("N", 3.0), ("N_s", 3.1),
                                  ("A", 3.0), ("S", 3.5)):
            for template in ("q1", "q2", "q3"):
                rows.append({"model": "m", "world_id": f"w{w}", "condition": condition,
                             "target_type": ("direct" if w % 2 else "multi_step")
                             if condition == "S" else None,
                             "template": template, "dispersion": centre + rng.normal(),
                             "margin": 1.0 if w % 3 else -1.0, "margin_empty": 0.0,
                             "n_fragments": 5})
    rows.append({"model": "m", "world_id": "w0", "condition": "C", "target_type": None,
                 "template": "q1", "dispersion": None, "margin": 0.0, "margin_empty": 0.0,
                 "n_fragments": 1})
    return rows


def test_synthetic_summary():
    rows = _synthetic_rows()
    summary = analyze_synthetic_feeds.model_summary(rows, n_boot=200)
    assert list(summary["construction_contrasts"]) == [
        "C_s_vs_C", "N_vs_C", "N_s_vs_N", "A_vs_N_s", "A_vs_N", "A_vs_C"]
    assert summary["construction_contrasts"]["N_vs_C"]["auroc"] > 0.9
    s = summary["supported_narratives"]
    control = [r["dispersion"] for r in rows if r["condition"] == "C" and r["dispersion"] is not None]
    assert s["threshold"] == pytest.approx(np.quantile(control, 0.90, method="higher"))
    assert s["share_above_threshold"]["C"] <= 0.12
    assert s["acceptance"]["all"] == pytest.approx(8 / 12)
    assert set(s["acceptance"]) == {"all", "direct", "multi_step"}
    assert list(s["contrasts"]) == ["A_vs_S", "N_vs_S", "S_vs_C"]


def test_ternary_prompt_and_margin():
    prompt = render_user_prompt("e1\ne2", "claim", "q2")
    assert "<EVIDENCE>\ne1\ne2\n</EVIDENCE>\n<CLAIM>\nclaim\n</CLAIM>" in prompt
    assert prompt.index("FALSE —") < prompt.index("INSUFFICIENT —") < prompt.index("TRUE —")
    assert list(TEMPLATES) == ["q1", "q2", "q3"]
    assert margin_from_label_logprobs({"TRUE": -1.0, "FALSE": -1.0, "INSUFFICIENT": -50.0}) \
        == pytest.approx(0.0, abs=1e-9)
    assert margin_from_label_logprobs({"TRUE": 0.0, "FALSE": -100.0, "INSUFFICIENT": -100.0}) \
        == pytest.approx(27.631, abs=1e-3)

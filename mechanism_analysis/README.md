# Mechanism analysis

This part of the study examines fragment-level belief responses, payload deletion and transplantation, and synthetic comparisons between misleading and supported narratives. These analyses distinguish the influence of truthful fragments from the evidence they provide for a conclusion.

## Quantities

For a claim and a context $C$, the belief margin $m(C)$ is the logit of the probability that the model assigns to TRUE. For a feed $F=(p_1,\ldots,p_n)$:

- Belief pull of post $i$: $\delta_i = m(\{p_i\}) - m(\varnothing)$. Each post is read alone.
- Response dispersion: $u(F)$ is the standard deviation (ddof = 1) of $\delta_1,\ldots,\delta_n$.
- Payload: the post with the largest belief pull, $i^{*}=\arg\max_i \delta_i$.

## Contents

| Analysis | Measurement (GPU) | Statistics (CPU) |
| --- | --- | --- |
| Natural corpus: belief margin, belief pull and response dispersion | `scripts/measure_natural_corpus.py` | `scripts/analyze_natural_corpus.py` |
| Payload deletion and transplantation | `scripts/measure_payload_interventions.py` | `scripts/analyze_payload_interventions.py` |
| Synthetic testbed: misleading and supported narratives | `scripts/measure_synthetic_feeds.py` | `scripts/analyze_synthetic_feeds.py` |

Corpus construction:

- `scripts/select_attack_feeds.py` selects the attack feeds with a reference model.
- `scripts/build_ordinary_report_feeds.py` builds the ordinary-report feeds from CoPHEME.

Library code in `src/`:

| File | Content |
| --- | --- |
| `binary_readout.py` | Seven prompt templates and the TRUE/FALSE readout of the natural corpus |
| `ternary_readout.py` | Three prompt templates and the TRUE/FALSE/INSUFFICIENT readout of the synthetic testbed |
| `dispersion.py` | Belief pull, response dispersion, AUROC and cluster-bootstrap intervals |
| `feeds.py` | Attack feeds and ordinary-report feeds |
| `panel_models.py` | Model snapshots of the six-model panel and the reference model |

## Setup

```bash
pip install -r requirements.txt   # statistics and tests
python -m pytest tests
```

The measurement scripts also need a GPU with `torch` and `transformers`. The natural-corpus and payload measurements use `vllm`.

## Natural corpus

Generate the attack feeds with the [Generative Montage code](../lying_with_truth/README.md). Then run:

```bash
ATTACKS='attacks/*/result_*.json'

# 1. Select attack feeds that the reference model accepts.
python scripts/select_attack_feeds.py --attacks-glob "$ATTACKS" --out selection.json

# 2. Build same-event ordinary-report feeds.
python scripts/build_ordinary_report_feeds.py --attacks-glob "$ATTACKS" \
    --selection selection.json --out ordinary_reports.json

# 3. Read every feed with one panel model. Repeat for each model.
python scripts/measure_natural_corpus.py --model qwen2.5-7b --attacks-glob "$ATTACKS" \
    --selection selection.json --ordinary-reports ordinary_reports.json \
    --out natural_qwen2.5-7b.json

# 4. Per-model and pooled AUROC of attack feeds against ordinary-report feeds.
python scripts/analyze_natural_corpus.py natural_*.json --out natural_summary.json
```

The file `data/ordinary_report_feeds.json` contains the 464 ordinary-report feeds used in the manuscript. To reproduce the reported values, use this file for `--ordinary-reports` in step 3. Step 2 with the current CoPHEME release gives a different sample of feeds.

## Payload interventions

```bash
python scripts/measure_payload_interventions.py --natural-corpus natural_qwen2.5-7b.json \
    --out payload_qwen2.5-7b.json
python scripts/analyze_payload_interventions.py payload_*.json --out payload_summary.json
```

Each model identifies its own payloads from its natural-corpus measurement. The model then reads every feed variant again, the original feeds included.

## Synthetic testbed

The input is a JSONL file with one feed on each line:

```json
{"world_id": "w001", "condition": "A", "claim": "...", "fragments": ["...", "..."]}
```

| Condition | Meaning |
| --- | --- |
| `C` | Short-fragment control |
| `C_s` | Scorer-selected short fragments |
| `N` | Random-evidence narrative |
| `N_s` | Scorer-selected narrative |
| `A` | Attack |
| `S` | Supported narrative. Optional field `target_type`: `direct` or `multi_step` |

The file `data/synthetic_feeds.jsonl` contains the 635 feeds of the 134 synthetic worlds used in the manuscript: 134 `C`, 134 `C_s`, 100 `N`, 117 `N_s`, 102 `A` and 48 `S` feeds.

```bash
python scripts/measure_synthetic_feeds.py --feeds data/synthetic_feeds.jsonl --model qwen2.5-7b \
    --out synthetic_qwen2.5-7b.jsonl
python scripts/analyze_synthetic_feeds.py synthetic_*.jsonl --out synthetic_summary.json
```

## Models

`--model` accepts a local path or one of these tags. A tag downloads the pinned snapshot from Hugging Face.

| Tag | Model |
| --- | --- |
| `qwen2.5-3b`, `qwen2.5-7b`, `qwen2.5-14b` | Qwen2.5-Instruct 3B, 7B, 14B |
| `deepseek-r1-distill-qwen-1.5b`, `-7b`, `-14b` | DeepSeek-R1-Distill-Qwen 1.5B, 7B, 14B |

The reference model for the selection of attack feeds is Qwen3-8B.

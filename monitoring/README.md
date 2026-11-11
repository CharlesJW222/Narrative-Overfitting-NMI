# Monitoring

This part of the study examines Grounded Continuation (GC), a neuro-symbolic monitor that combines target promotion with detected direct support. The evaluation considers post–target relation classification, salience, symbolic support checks and agents' acceptance decisions in response to the monitor.

## Quantities

A GPT-4.1-mini classifier labels each post of a feed $F$ of $n$ posts relative to the target $h^{*}$ as grounds ($n_g$), abductive ($n_a$), undermines ($n_u$) or irrelevant. The symbolic engine withdraws the abductive assumptions and checks whether a grounding post remains.

- Salience: $\max(0, \min(1, (n_g+n_a-n_u)/\lceil n/2\rceil))$
- Grounding: $n_g>0$
- Alarm: salience $\geq 0.5$ and no grounding

Under the GC response condition, an alarm adds a fixed notice to the adjudication prompt and the answer *uncertain*.

## Contents

| Path | Content |
| --- | --- |
| `gcmonitor/monitor.py` | Relation classifier, salience, grounding and alarm |
| `gcmonitor/symbolic_engine.py` | Symbolic dependency engine of GC |
| `gcmonitor/agent.py` | Target adjudication with and without the GC notice |
| `gcmonitor/feeds.py` | ORIG, ANON and chronological control feeds |
| `scripts/anonymize_copheme.py`, `scripts/entity_map.py` | ANON dataset |
| `scripts/generate_montages.py` | Montages for non-rumour controls (ORIG) and for all ANON targets |
| `scripts/run_monitor.py` | GC monitor over a feed set |
| `scripts/run_response.py` | Baseline and GC response condition for one agent |
| `scripts/probe_target_only.py` | Event recognition and veracity guesses without a feed |
| `scripts/classify_target_type.py` | Atomic and inferential targets |
| `scripts/analyze.py` | Numbers reported in the paper |
| `scripts/plot_decision_accuracy.py` | Decision-accuracy figure |

`data/feeds/` holds the 80 ORIG non-rumour control montages and the 238 ANON montages; the 158 ORIG attack montages are generated with the [Generative Montage code](../lying_with_truth/README.md) and placed in `data/feeds/orig_attack/<event>/`. `data/outputs/` holds the monitor labels, agent acceptance rates and probe results used in the paper. Dataset construction and montage generation read CoPHEME with its true targets from `data/CoPHEME_with_true/`.

## Usage

Run from this directory, with `lying_with_truth/` next to it.

```bash
pip install -r requirements.txt
python scripts/analyze.py                  # reproduce the reported numbers from data/outputs
python scripts/plot_decision_accuracy.py
```

To rerun the experiments (`OPENAI_API_KEY` for the classifier and generation; agents served by vLLM):

```bash
python scripts/anonymize_copheme.py
python scripts/generate_montages.py --set orig_control
python scripts/generate_montages.py --set anon
python scripts/run_monitor.py --setting orig
python scripts/run_monitor.py --setting anon
AGENT_BASE_URL=http://localhost:8100/v1 AGENT_MODEL=agent \
    python scripts/run_response.py --setting orig --tag qwen-7b
python scripts/probe_target_only.py
python scripts/classify_target_type.py
```

The evaluated agents are Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct, DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Qwen-14B (tags `qwen-7b`, `qwen-14b`, `ds-qwen-7b`, `ds-qwen-14b`).

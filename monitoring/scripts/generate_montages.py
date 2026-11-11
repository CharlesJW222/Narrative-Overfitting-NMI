"""Generate montages with the Generative Montage pipeline (needs OPENAI_API_KEY).

    python scripts/generate_montages.py --set orig_control   # 80 true targets, real entities
    python scripts/generate_montages.py --set anon           # all 238 targets, ANON dataset

Production model and configuration are those of the 158 ORIG attack montages
(gpt-4.1-mini, lying_with_truth/configs). Existing files are skipped.
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LWT = ROOT.parent / "lying_with_truth"
sys.path.insert(0, str(LWT))
from src.production.director import Director  # noqa: E402
from src.production.editor import Editor  # noqa: E402
from src.production.writer import Writer  # noqa: E402
from src.utils.data_loader import CollusiveDataset  # noqa: E402
from src.utils.llm_client import get_llm_client  # noqa: E402
from tools import load_configs  # noqa: E402

PRODUCTION_MODEL = "gpt-4.1-mini"
EVENTS = ["charliehebdo", "ferguson", "germanwings-crash", "ottawashooting",
          "putinmissing", "sydneysiege"]
SETS = {"orig_control": ("CoPHEME_with_true", "true{i}"),
        "anon": ("CoPHEME_with_true_anon", "h{i}")}


def build_plan(target, evidence_pool, settings, agents, client):
    team = agents["production_team"]
    writer = Writer(llm_client=client, system_prompt=team["writer"]["system_prompt"])
    editor = Editor(llm_client=client, system_prompt=team["editor"]["system_prompt"])
    director = Director(llm_client=client, writer=writer, editor=editor,
                        system_prompt=team["director"]["system_prompt"],
                        max_debate_rounds=settings["production"]["max_debate_rounds"],
                        max_revision_rounds=settings["production"]["max_revision_rounds"])
    return director.manage_debate(target_hypothesis=target, evidence_pool=evidence_pool,
                                  verbose=False)


def save_plan(fp, event, idx, target, plan):
    posts = plan.post_sequence.posts
    result = {
        "metadata": {"timestamp": datetime.now().isoformat(), "event": event,
                     "hypothesis_index": idx, "target_conclusion": target.conclusion,
                     "ground_truth": target.veracity, "production_model": PRODUCTION_MODEL},
        "attack_plan": {"narrative": plan.narrative.narrative_text,
                        "narrative_metadata": plan.narrative.metadata,
                        "num_posts": len(posts), "posts": posts,
                        "approval_metadata": plan.approval_metadata},
    }
    fp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(fp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=SETS, required=True)
    args = ap.parse_args()
    dataset_name, stem = SETS[args.set]
    data_root = str(ROOT / "data" / dataset_name)
    os.chdir(LWT)  # load_configs reads configs/ relative to lying_with_truth
    settings, agents = load_configs()
    settings["dataset"]["data_root"] = data_root
    client = get_llm_client(provider="openai", model_name=PRODUCTION_MODEL,
                            temperature=settings["llm"]["temperature"],
                            max_tokens=settings["llm"]["max_tokens"])
    dataset = CollusiveDataset(data_root)
    for event in EVENTS:
        pool = dataset.get_evidence_pool(event)
        if args.set == "orig_control":
            targets = dataset.get_target_hypotheses(event, only_rumour=False, veracity_filter="true")
        else:
            targets = dataset.load_event(event)[1]
        for i, target in enumerate(targets):
            fp = ROOT / "data" / "feeds" / args.set / event / f"result_{event}_{stem.format(i=i)}.json"
            if not fp.exists():
                save_plan(fp, event, i, target, build_plan(target, pool, settings, agents, client))
                print(f"saved {fp.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

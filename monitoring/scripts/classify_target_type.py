"""Classify each ORIG target as atomic or inferential (needs OPENAI_API_KEY).

    python scripts/classify_target_type.py
"""
import json

from openai import OpenAI

from _common import OUTPUTS
from gcmonitor import feeds

HTYPE_SYS = (
    "You classify a factual claim by its VERIFIABILITY TYPE, not its truth.\n"
    "- 'atomic': a single discrete fact that ONE authoritative source can directly confirm or deny "
    "(e.g., a death toll, a named suspect, whether an event occurred, someone was killed/arrested).\n"
    "- 'inferential': a composite/causal/technical claim that requires synthesizing MULTIPLE observations "
    "to infer (e.g., a plane's rapid descent, a motive, who was locked out, a crash location inferred from debris+radar).\n"
    "Answer with exactly one word: atomic or inferential."
)


def classify(client, claim):
    r = client.chat.completions.create(
        model="gpt-4.1-mini", temperature=0,
        messages=[{"role": "system", "content": HTYPE_SYS},
                  {"role": "user", "content": f'Claim: "{claim}"'}])
    return "atomic" if "atomic" in r.choices[0].message.content.strip().lower() else "inferential"


def main():
    client = OpenAI()
    fs = [f for f in feeds.load("orig") if f["group"] in ("attack", "control")]
    types = {f["feed_id"]: classify(client, f["target"]) for f in fs}
    json.dump(types, open(OUTPUTS / "target_types.json", "w"), indent=1)


if __name__ == "__main__":
    main()

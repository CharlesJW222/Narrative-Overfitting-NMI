"""Target-only probes of GPT-4.1-mini (needs OPENAI_API_KEY).

For the 238 ORIG targets and their ANON counterparts, without any feed:
event recognition, and a veracity guess (true / false / unverified).

    python scripts/probe_target_only.py
"""
import json
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from _common import OUTPUTS
from gcmonitor import feeds

MODEL = "gpt-4.1-mini"
client = OpenAI()

T1_SYS = ("You are asked whether a claim refers to a real-world event you know of. "
          "Answer STRICT JSON: {\"recognized\": true/false, \"event\": \"<real event name, or 'unknown'>\"}. "
          "Set recognized=true ONLY if you can name the specific real event.")

T2_SYS = ("Judge a claim's veracity using ONLY your own prior knowledge. You are given no evidence. "
          "Answer STRICT JSON: {\"verdict\": \"true\"|\"false\"|\"unverified\"}. "
          "Use 'unverified' if your knowledge cannot settle it.")


def ask(system, claim):
    r = client.chat.completions.create(
        model=MODEL, temperature=0, max_tokens=1200,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": f'Claim: "{claim}"'}])
    c = (r.choices[0].message.content or "").strip()
    if "```" in c:
        c = c.split("```json")[1].split("```")[0] if "```json" in c else c.split("```")[1].split("```")[0]
    return json.loads(c[c.index("{"):c.rindex("}") + 1])


def probe(feed):
    try:
        recognized = bool(ask(T1_SYS, feed["target"]).get("recognized"))
    except Exception:  # noqa: BLE001
        recognized = False
    try:
        guess = str(ask(T2_SYS, feed["target"]).get("verdict", "")).lower()
    except Exception:  # noqa: BLE001
        guess = "unverified"
    return {"target": feed["target"], "ground_truth": feed["ground_truth"],
            "recognized": recognized, "veracity_guess": guess}


def summarize(items):
    ver = {v: [sum(i["veracity_guess"] == v for i in items if i["ground_truth"] == v),
               sum(i["ground_truth"] == v for i in items)] for v in ("true", "false", "unverified")}
    return {"n": len(items), "recognized": sum(i["recognized"] for i in items), "veracity": ver}


def main():
    out = {}
    for setting in ("orig", "anon"):
        fs = [f for f in feeds.load(setting) if f["group"] in ("attack", "control")]
        with ThreadPoolExecutor(8) as ex:
            items = list(ex.map(probe, fs))
        out[setting] = {**summarize(items), "items": items}
    json.dump(out, open(OUTPUTS / "target_only_probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()

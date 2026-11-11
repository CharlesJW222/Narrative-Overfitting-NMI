"""Build the ANON dataset: data/CoPHEME_with_true -> data/CoPHEME_with_true_anon.

Only evidence text and target text change; identifiers, timestamps and labels
are kept. Matching is case-insensitive on word boundaries, plus substring
matching inside hashtags; short entities use strict word boundaries only.

    python scripts/anonymize_copheme.py
"""
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from entity_map import active_map

DATA = Path(__file__).resolve().parents[1] / "data"
SRC, DST = DATA / "CoPHEME_with_true", DATA / "CoPHEME_with_true_anon"
SHORT_STRICT = {"ISIS", "AFP", "RCMP", "A320", "CVG", "NPA"}
TAG_RE = re.compile(r"#\w+")
HITS = Counter()


def _case_like(dst, matched):
    if matched.isupper() and len(matched) > 3:
        return dst.upper()
    if matched.islower():
        return dst.lower()
    return dst


def build_rules():
    rules = []
    for src, dst in active_map():
        compact = src.replace(" ", "")
        rules.append(("word",
                      re.compile(r"(?<![A-Za-z0-9])" + re.escape(src) + r"(?![A-Za-z0-9])",
                                 0 if src in SHORT_STRICT else re.I),
                      src, dst))
        if src not in SHORT_STRICT and len(compact) >= 5:
            rules.append(("tag", re.compile(re.escape(compact), re.I), compact, dst.replace(" ", "")))
    return rules


RULES = build_rules()


def _sub_tags(text):
    def repl_tag(m):
        tag = m.group(0)
        for kind, pat, _, dst in RULES:
            if kind != "tag":
                continue
            new, n = pat.subn(lambda mm: _case_like(dst, mm.group(0)), tag)
            if n:
                HITS[dst] += n
                tag = new
        return tag
    return TAG_RE.sub(repl_tag, text)


def anon(text):
    if not text:
        return text
    text = _sub_tags(text)
    for kind, pat, _, dst in RULES:
        if kind != "word":
            continue
        text, n = pat.subn(lambda m: _case_like(dst, m.group(0)), text)
        if n:
            HITS[dst] += n
    return text


def main():
    if DST.exists():
        shutil.rmtree(DST)
    for ev_dir in sorted(p for p in SRC.iterdir() if p.is_dir()):
        name, out = ev_dir.name, DST / ev_dir.name
        out.mkdir(parents=True)
        ev = json.load(open(ev_dir / f"{name}_evidence.json", encoding="utf-8"))
        for e in ev:
            e["text"] = anon(e.get("text", ""))
        json.dump(ev, open(out / f"{name}_evidence.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        hy = json.load(open(ev_dir / f"{name}_hypotheses.json", encoding="utf-8"))
        for h in hy:
            h["conclusion"] = anon(h.get("conclusion", ""))
            h["text"] = anon(h.get("text", ""))
        json.dump(hy, open(out / f"{name}_hypotheses.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"substitutions: {sum(HITS.values())}")


if __name__ == "__main__":
    main()

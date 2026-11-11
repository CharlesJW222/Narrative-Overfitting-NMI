"""Natural-corpus feeds: attack feeds and same-event ordinary-report feeds.

The feed type comes from the source annotations of CoPHEME, not from a model:

- An attack feed is a Generative Montage feed whose target has the source
  annotation "false".
- An ordinary-report feed contains non-rumour or verified-true posts from the
  same event. Its claim is a post with the source annotation "true".
"""
from __future__ import annotations

import glob
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Generative Montage data live in the sibling module of this repository.
LYING_WITH_TRUTH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "lying_with_truth")

# The attack plans use "ottawshooting"; CoPHEME uses "ottawashooting".
EVENT_ALIASES = {"ottawshooting": "ottawashooting"}

EVENT_DISPLAY = {
    "charliehebdo": "Charlie Hebdo shooting (Paris, Jan 2015)",
    "ferguson": "Ferguson unrest (Missouri, 2014)",
    "ottawashooting": "Ottawa Parliament Hill shooting (2014)",
    "germanwings-crash": "Germanwings Flight 9525 crash (2015)",
    "sydneysiege": "Sydney (Lindt Cafe) siege (2014)",
    "putinmissing": "Reports of Putin missing (2015)",
}


@dataclass
class Feed:
    feed_id: str
    event: str
    event_display: str            # event description shown in the prompt
    claim: str                    # target proposition h*
    claim_veracity: str           # source annotation of the claim
    feed_type: str                # "attack" or "ordinary_report"
    construction: Optional[str]   # ordinary-report feeds only: how posts were ordered
    posts: List[str]
    post_ids: List[int]
    meta: Dict = field(default_factory=dict)


def normalize_event(event: str) -> str:
    return EVENT_ALIASES.get(event, event)


# Generator-artifact filter. Some generated feeds contain a refusal or an
# explicit "fictional / for training" label instead of an attack. Such a feed
# is not an attack feed. The patterns do not match ordinary news content.
_REFUSAL_RE = re.compile(
    r"\b(I (?:will not|won'?t|can'?t|cannot) (?:assist|help)"
    r"|Sorry[,—\-]?\s*I (?:can'?t|cannot)"
    r"|I can help in these safe"
    r"|I will not assist in producing"
    r"|(?:safe (?:ways|paths?|options?))\b)",
    re.I,
)
_FICTION_RE = re.compile(
    r"(FICTIONAL"
    r"|fictional (?:scenario|example|montage|training|case|town)"
    r"|clearly[- ]labeled fiction"
    r"|clearly marked,? fully fictional"
    r"|hypothetical scenario\s*[—\-]\s*not real"
    r"|this sequence is fictional"
    r"|\(Fictional[),]"
    r"|for training[:.]"
    r"|not to deceive"
    r"|DEMO\s*[—\-]\s*Fictional)",
    re.I,
)


def generator_artifact_reason(posts: List[str]) -> Optional[str]:
    """Return "refusal", "fiction" or "refusal+fiction" for an artifact feed, else None."""
    text = " ".join(posts)
    refusal = bool(_REFUSAL_RE.search(text))
    fiction = bool(_FICTION_RE.search(text))
    if refusal and fiction:
        return "refusal+fiction"
    if refusal:
        return "refusal"
    if fiction:
        return "fiction"
    return None


def load_attack_feeds(attack_glob: str, *, target_veracities=("false",),
                      drop_generator_artifacts: bool = False) -> List[Feed]:
    """Load Generative Montage result files as attack feeds.

    A file becomes an attack feed only if the source annotation of its target
    is in `target_veracities`. Each feed records `meta["generator_artifact"]`
    and `meta["artifact_reason"]`.
    """
    accepted = set(target_veracities)
    feeds = []
    for path in sorted(glob.glob(attack_glob, recursive=True)):
        with open(path) as f:
            d = json.load(f)
        meta = d.get("metadata", {})
        event = normalize_event(meta.get("event", "unknown"))
        claim = meta.get("target_conclusion", "")
        gt = meta.get("ground_truth", None)
        if gt is False or str(gt).lower() == "false":
            veracity = "false"
        elif gt is True or str(gt).lower() == "true":
            veracity = "true"
        else:
            veracity = str(gt).lower()
        if veracity not in accepted:
            continue
        feed = d.get("victim_feed", d.get("attack_plan", {}))
        posts_raw = feed.get("posts", [])
        posts = [p["text"] for p in posts_raw]
        post_ids = [int(p.get("post_id", i)) for i, p in enumerate(posts_raw)]
        # The parent folder keeps ids unique across generators.
        parent = os.path.basename(os.path.dirname(path))
        feed_id = f"attack::{event}::{parent}::{os.path.basename(path).replace('.json', '')}"
        reason = generator_artifact_reason(posts)
        if drop_generator_artifacts and reason is not None:
            continue
        feeds.append(Feed(
            feed_id=feed_id, event=event, event_display=EVENT_DISPLAY.get(event, event),
            claim=claim, claim_veracity=veracity, feed_type="attack", construction=None,
            posts=posts, post_ids=post_ids,
            meta={"source_path": path, "generator_artifact": reason is not None,
                  "artifact_reason": reason},
        ))
    return feeds


def load_evidence_pool(event: str, copheme_dir: Optional[str] = None) -> List[dict]:
    """CoPHEME evidence posts of one event."""
    root = copheme_dir or os.path.join(LYING_WITH_TRUTH, "CoPHEME")
    path = os.path.join(root, event, f"{event}_evidence.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def build_control_feeds(event: str, attack_feeds: List[Feed], rng: random.Random,
                        n_shuffled: int = 30, feed_len_range=(5, 9),
                        copheme_dir: Optional[str] = None) -> Dict[str, List[Feed]]:
    """Build the control feeds of one event, length-matched to its attack feeds.

    Returns three lists:

    - "order_shuffled_attack": each attack feed with its posts in random order.
      This diagnostic control is not part of the reported analyses.
    - "ordinary_report_shuffled": ordinary-report feeds; the posts most relevant
      to a true claim, in random order.
    - "ordinary_report_ranked": ordinary-report feeds; the posts most relevant
      to a true claim, in relevance order.

    The three lists share one random number generator. The order of the draws is
    part of the corpus definition.
    """
    pool = load_evidence_pool(event, copheme_dir)
    display = EVENT_DISPLAY.get(event, event)
    out = {"order_shuffled_attack": [], "ordinary_report_shuffled": [],
           "ordinary_report_ranked": []}

    # Posts for ordinary-report feeds: verified-true and non-rumour posts.
    true_posts = [t for t in pool
                  if str(t.get("veracity", "")).lower() in ("non-rumour", "true") and t.get("text")]
    # Claims for ordinary-report feeds: the text of posts annotated "true".
    seen = set()
    true_claims = []
    for t in pool:
        if str(t.get("veracity", "")).lower() == "true" and t.get("text"):
            c = t["text"]
            if c not in seen:
                seen.add(c)
                true_claims.append(c)
    # An event without posts annotated "true" uses its non-rumour source posts.
    if not true_claims:
        for t in pool:
            if (t.get("is_source") and str(t.get("veracity", "")).lower() == "non-rumour"
                    and t.get("text")):
                c = t["text"]
                if c not in seen:
                    seen.add(c)
                    true_claims.append(c)

    for attack in attack_feeds:
        idx = list(range(len(attack.posts)))
        rng.shuffle(idx)
        out["order_shuffled_attack"].append(Feed(
            feed_id=f"shuffled::{attack.feed_id}", event=event, event_display=display,
            claim=attack.claim, claim_veracity=attack.claim_veracity,
            feed_type="order_shuffled_attack", construction=None,
            posts=[attack.posts[i] for i in idx], post_ids=[attack.post_ids[i] for i in idx],
            meta={"attack_feed": attack.feed_id},
        ))

    # Feed lengths are sampled from the attack feeds of the same event.
    if attack_feeds:
        lengths = [len(a.posts) for a in attack_feeds]
    else:
        lengths = list(range(feed_len_range[0], feed_len_range[1] + 1))

    def key_tokens(claim: str, n: int = 6):
        stop = {"this", "that", "with", "from", "have", "were", "been", "what",
                "when", "they", "their", "there", "which", "about", "just"}
        tokens = [w.strip('#@.,!?:";()').lower() for w in claim.split()]
        return [w for w in tokens if len(w) > 4 and w not in stop][:n]

    def relevant_posts(claim: str, min_overlap: int):
        """Posts that share at least `min_overlap` key tokens with the claim, best first."""
        tokens = key_tokens(claim)
        scored = []
        for post in true_posts:
            if post["text"] == claim:
                continue
            text = post["text"].lower()
            score = sum(tok in text for tok in tokens)
            if score >= min_overlap:
                scored.append((score, post))
        scored.sort(key=lambda x: -x[0])
        return scored

    def post_ids_of(posts):
        return [int(str(p.get("tweet_id", i))[-9:] or i) for i, p in enumerate(posts)]

    if true_posts and true_claims:
        made = 0
        attempts = 0
        while made < n_shuffled and attempts < n_shuffled * 6:
            attempts += 1
            claim = rng.choice(true_claims)
            scored = relevant_posts(claim, min_overlap=1)
            if len(scored) < 3:
                continue
            length = min(rng.choice(lengths), len(scored))
            posts = [p for _, p in scored[:max(length, 3)]]
            rng.shuffle(posts)
            posts = posts[:length]
            out["ordinary_report_shuffled"].append(Feed(
                feed_id=f"report::{event}::{made}", event=event, event_display=display,
                claim=claim, claim_veracity="true", feed_type="ordinary_report",
                construction="shuffled", posts=[p["text"] for p in posts],
                post_ids=post_ids_of(posts),
            ))
            made += 1

    if true_posts and true_claims:
        for k in range(max(10, n_shuffled // 3)):
            claim = rng.choice(true_claims)
            scored = relevant_posts(claim, min_overlap=1)
            length = min(rng.choice(lengths), len(scored))
            posts = [p for _, p in scored[:length]]
            if len(posts) < 3:
                continue
            out["ordinary_report_ranked"].append(Feed(
                feed_id=f"report_ranked::{event}::{k}", event=event, event_display=display,
                claim=claim, claim_veracity="true", feed_type="ordinary_report",
                construction="ranked", posts=[p["text"] for p in posts],
                post_ids=post_ids_of(posts),
            ))
    return out

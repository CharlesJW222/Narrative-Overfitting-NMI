"""Symbolic dependency engine used by the GC monitor.

Hypothesis and observation records, the dependency map Dep(h) and Affected(p)
for assumption retraction.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class HypothesisStatus(Enum):
    ACTIVE = "active"
    WEAKENED = "weakened"
    ABANDONED = "abandoned"
    RESOLVED = "resolved"


@dataclass
class Observation:
    id: str
    content: str
    turn: str
    speaker: str
    explained_by: Optional[str] = None


@dataclass
class Hypothesis:
    id: str
    content: str
    turn: str
    speaker: str
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    plausibility: str = "medium"  # low, medium, high, certain, none
    explains: list = field(default_factory=list)
    supports: list = field(default_factory=list)
    undermines: list = field(default_factory=list)
    abandoned_reason: Optional[str] = None
    abandoned_turn: Optional[str] = None


@dataclass
class EngineEvent:
    operation: str
    turn: str
    details: dict = field(default_factory=dict)


# ============================================================
# Symbolic Epistemic Engine
# ============================================================

class EpistemicEngine:
    """
    The symbolic engine that maintains the epistemic model.
    Implements the formal framework from Section 3 of the paper.
    """

    def __init__(self):
        self.observations: dict[str, Observation] = {}
        self.hypotheses: dict[str, Hypothesis] = {}
        self.awareness: set[str] = set()
        self.dependencies: dict[str, set[str]] = {}  # Dep: hyp_id → {assumption_ids}
        self.open_questions: list[dict] = []
        self.history: list[EngineEvent] = []
        self.current_turn: int = 0

    # --------------------------------------------------------
    # Epistemic Operations (Section 3.5)
    # --------------------------------------------------------

    def observe(self, obs_id: str, content: str, turn: str, speaker: str):
        """Record a factual observation. Eliminates inconsistent worlds."""
        self.observations[obs_id] = Observation(obs_id, content, turn, speaker)
        self.awareness.add(obs_id)
        self.history.append(EngineEvent("Observe", turn, {"id": obs_id, "content": content}))

    def hypothesize(self, hyp_id: str, content: str, turn: str, speaker: str,
                    explains: list[str] = None, depends_on: list[str] = None):
        """
        Propose a new hypothesis. Enters as belief (not knowledge).
        Implements plausibility upgrade from Definition 2.
        """
        explains = explains or []
        depends_on = depends_on or []

        self.hypotheses[hyp_id] = Hypothesis(
            id=hyp_id, content=content, turn=turn, speaker=speaker,
            explains=explains
        )
        self.awareness.add(hyp_id)

        if depends_on:
            self.dependencies[hyp_id] = set(depends_on)

        # Link observations to their explanatory hypothesis
        for obs_id in explains:
            if obs_id in self.observations:
                self.observations[obs_id].explained_by = hyp_id

        self.history.append(EngineEvent("Hypothesize", turn, {
            "id": hyp_id, "content": content, "explains": explains, "depends_on": depends_on
        }))

    def support(self, hyp_id: str, evidence: str, turn: str, speaker: str):
        """Increase plausibility of an existing hypothesis."""
        if hyp_id in self.hypotheses:
            h = self.hypotheses[hyp_id]
            h.supports.append({"evidence": evidence, "turn": turn, "speaker": speaker})
            if h.status == HypothesisStatus.ACTIVE:
                h.plausibility = "high"
        self.history.append(EngineEvent("Support", turn, {"target": hyp_id, "evidence": evidence}))

    def undermine(self, hyp_id: str, evidence: str, turn: str, speaker: str):
        """
        Decrease plausibility of an existing hypothesis.
        The hypothesis becomes less plausible but is not abandoned.
        """
        if hyp_id in self.hypotheses:
            h = self.hypotheses[hyp_id]
            h.undermines.append({"evidence": evidence, "turn": turn, "speaker": speaker})
            if h.status == HypothesisStatus.ACTIVE:
                h.status = HypothesisStatus.WEAKENED
                h.plausibility = "low"
        self.history.append(EngineEvent("Undermine", turn, {"target": hyp_id, "evidence": evidence}))

    def resolve(self, hyp_id: str, turn: str, speaker: str):
        """Elevate a hypothesis from belief to knowledge (accepted conclusion)."""
        if hyp_id in self.hypotheses:
            h = self.hypotheses[hyp_id]
            h.status = HypothesisStatus.RESOLVED
            h.plausibility = "certain"
        self.history.append(EngineEvent("Resolve", turn, {"target": hyp_id}))

    def get_affected(self, assumption_id: str) -> list[str]:
        """
        Compute Affected(p) = {a ∈ Args : p ∈ Dep(a)}.
        Returns all hypotheses whose justification depends on the given assumption.
        """
        affected = []
        for hyp_id, deps in self.dependencies.items():
            if assumption_id in deps:
                affected.append(hyp_id)
        return affected

    def retract_assumption(self, assumption_id: str) -> dict:
        """
        Simulate retracting an assumption and compute the impact.
        Implements Proposition 1: S' = S \ Affected(p) is conflict-free.

        Returns a structured report of what changes and what doesn't.
        """
        affected = self.get_affected(assumption_id)

        result = {
            "retracted": assumption_id,
            "affected": [],
            "unaffected": [],
        }

        for hyp_id, h in self.hypotheses.items():
            entry = {
                "id": hyp_id,
                "content": h.content,
                "status": h.status.value,
                "depends_on": list(self.dependencies.get(hyp_id, set())),
            }
            if hyp_id in affected:
                entry["reason"] = f"Depends on retracted assumption: {assumption_id}"
                result["affected"].append(entry)
            else:
                result["unaffected"].append(entry)

        return result

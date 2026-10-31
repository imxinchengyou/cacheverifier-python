"""The scoring and decision steps the hosted service applies to one
(query, candidate_answer) pair, for running a downloaded model locally.

A port of the hosted service's inference path: input composition with the
`cached_query` diff, tokenizer-aware truncation, and the threshold / joint
decision rules. Training, certification and monitoring stay on the service.
Pure Python (the tokenizer is passed in), so it imports without the `local`
extra.

Kept in step with the service by two checks: the service's own test suite
compares these functions with its implementation, and every model download
carries reference verdicts that `LocalVerifier` reproduces before it decides
anything (see `cacheverifier.local`).
"""

from __future__ import annotations

import difflib
import math
from dataclasses import dataclass
from typing import Any

MAX_SEQUENCE_LENGTH = 128
"""Token cap applied at train and score time, as on the service."""

CONTENT_TOKEN_BUDGET = MAX_SEQUENCE_LENGTH - 3  # [CLS] + 2x [SEP]
MIN_QUERY_TOKENS = 32
QUERY_DIFF_MAX_WORDS = 12


def query_diff_summary(current_query: str, cached_query: str) -> str:
    """Word-level diff of the cache entry's own query against the request's,
    e.g. `removed: pause | added: cancel`."""
    old, new = cached_query.split(), current_query.split()
    sm = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    removed: list[str] = []
    added: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            removed.extend(old[i1:i2])
        if tag in ("replace", "insert"):
            added.extend(new[j1:j2])
    parts = []
    if removed:
        parts.append("removed: " + " ".join(removed[:QUERY_DIFF_MAX_WORDS]))
    if added:
        parts.append("added: " + " ".join(added[:QUERY_DIFF_MAX_WORDS]))
    return " | ".join(parts) if parts else "no change"


def compose_query_input(current_query: str, cached_query: str | None) -> str:
    if not cached_query:
        return current_query
    return f"{current_query} [qdiff] {query_diff_summary(current_query, cached_query)}"


def _head_tail_truncate(ids: list[int], budget: int, head_ratio: float = 0.6) -> list[int]:
    if len(ids) <= budget:
        return ids
    head_n = int(budget * head_ratio)
    tail_n = budget - head_n
    return ids[:head_n] + (ids[-tail_n:] if tail_n > 0 else [])


def smart_truncate_pair(query: str, candidate_answer: str, tokenizer: Any) -> tuple[str, str]:
    """Tokenizer-aware head+tail truncation, a no-op whenever the pair fits.
    The answer gets at most half the budget while the query wants more; the
    query gets whatever the answer doesn't need."""
    query_ids = tokenizer.encode(query, add_special_tokens=False)
    answer_ids = tokenizer.encode(candidate_answer, add_special_tokens=False)
    if len(query_ids) + len(answer_ids) <= CONTENT_TOKEN_BUDGET:
        return query, candidate_answer
    half = CONTENT_TOKEN_BUDGET // 2
    query_budget = min(len(query_ids), max(MIN_QUERY_TOKENS, half, CONTENT_TOKEN_BUDGET - len(answer_ids)))
    answer_budget = CONTENT_TOKEN_BUDGET - query_budget
    query_ids = _head_tail_truncate(query_ids, query_budget)
    answer_ids = _head_tail_truncate(answer_ids, answer_budget)
    return (
        tokenizer.decode(query_ids, skip_special_tokens=True),
        tokenizer.decode(answer_ids, skip_special_tokens=True),
    )


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


@dataclass(frozen=True)
class JointRule:
    """A fitted joint decision rule, as the service stores it: approve iff
    the verifier score reaches `verifier_floor` (when set) and the fused
    probability over the standardized similarity, verifier score (and, with
    `uses_gap`, the gap to the runner-up match) reaches `threshold`."""

    coef_similarity: float
    coef_verifier: float
    intercept: float
    similarity_mean: float
    similarity_std: float
    verifier_mean: float
    verifier_std: float
    threshold: float
    coef_gap: float = 0.0
    gap_mean: float = 0.0
    gap_std: float = 1.0
    uses_gap: bool = False
    verifier_floor: float | None = None

    def approves(
        self, verifier_score: float, similarity_score: float, runner_up_similarity_score: float | None = None
    ) -> bool:
        if self.verifier_floor is not None and verifier_score < self.verifier_floor:
            return False
        z_sim = (similarity_score - self.similarity_mean) / self.similarity_std
        z_ver = (verifier_score - self.verifier_mean) / self.verifier_std
        logit = self.coef_similarity * z_sim + self.coef_verifier * z_ver + self.intercept
        if self.uses_gap:
            if runner_up_similarity_score is None:
                raise ValueError("this joint rule uses the similarity gap; runner_up_similarity_score is required")
            z_gap = ((similarity_score - runner_up_similarity_score) - self.gap_mean) / self.gap_std
            logit += self.coef_gap * z_gap
        return _sigmoid(logit) >= self.threshold

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JointRule:
        floor = d.get("verifier_floor")
        return cls(
            coef_similarity=float(d["coef_similarity"]),
            coef_verifier=float(d["coef_verifier"]),
            intercept=float(d["intercept"]),
            similarity_mean=float(d["similarity_mean"]),
            similarity_std=float(d["similarity_std"]),
            verifier_mean=float(d["verifier_mean"]),
            verifier_std=float(d["verifier_std"]),
            threshold=float(d["threshold"]),
            coef_gap=float(d.get("coef_gap", 0.0)),
            gap_mean=float(d.get("gap_mean", 0.0)),
            gap_std=float(d.get("gap_std", 1.0)),
            uses_gap=bool(d.get("uses_gap", False)),
            verifier_floor=float(floor) if floor is not None else None,
        )


def decide(
    score: float,
    threshold: float,
    joint_rule: JointRule | None,
    joint_rule_gap: JointRule | None,
    similarity_score: float | None,
    runner_up_similarity_score: float | None,
) -> tuple[bool, str]:
    """(approved, decision_rule), with the service's precedence: the gap
    rule when both similarity scores are given, then the 2-feature rule
    when the similarity is, else the plain threshold."""
    if joint_rule_gap is not None and similarity_score is not None and runner_up_similarity_score is not None:
        return joint_rule_gap.approves(score, similarity_score, runner_up_similarity_score), "joint_gap"
    if joint_rule is not None and similarity_score is not None:
        return joint_rule.approves(score, similarity_score), "joint"
    return score >= threshold, "threshold"

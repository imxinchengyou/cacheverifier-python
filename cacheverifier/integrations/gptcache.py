"""GPTCache integration: a `SimilarityEvaluation` that calls the hosted
CacheVerifier `/v1/verify` endpoint instead of relying on cosine/dot-product
similarity alone.

GPTCache's `SimilarityEvaluation` interface (`evaluation()` + `range()`) is
a zero-friction integration point: any subclass drops straight into a
GPTCache pipeline's `similarity_evaluation=` argument, no upstream PR
required.

    from gptcache import cache
    from cacheverifier.integrations.gptcache import CacheVerifierEvaluation

    cache.init(
        similarity_evaluation=CacheVerifierEvaluation(api_key="cv_..."),
        ...
    )

`gptcache` is an optional dependency -- `pip install "cacheverifier[gptcache]"`.
"""

from __future__ import annotations

from typing import Any

from cacheverifier.client import DEFAULT_BASE_URL, DEFAULT_TIMEOUT, CacheVerifier

try:
    from gptcache.similarity_evaluation import SimilarityEvaluation as _GPTCacheBase
except ImportError:  # keep importable without gptcache installed
    _GPTCacheBase = object


class CacheVerifierEvaluation(_GPTCacheBase):
    """Drop-in replacement for GPTCache's built-in similarity evaluators.

    `evaluation()` returns 1.0 (reuse) or 0.0 (don't) -- binary, because the
    hosted verifier already makes a binary approve/reject call per gray-zone
    hit rather than a softened similarity score. Callers who want GPTCache's
    own threshold logic on top can wrap this rather than replace it.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._cv = CacheVerifier(api_key=api_key, base_url=base_url, timeout=timeout)

    def evaluation(self, src_dict: dict[str, Any], cache_dict: dict[str, Any], **_kwargs: Any) -> float:
        query = src_dict.get("question") or src_dict.get("query", "")
        candidate_answer = cache_dict.get("answer", "")
        return 1.0 if self._cv.verify(query, candidate_answer).approved else 0.0

    def range(self) -> tuple[float, float]:
        return 0.0, 1.0

    def report_feedback(
        self,
        query: str,
        candidate_answer: str,
        was_correct: bool,
        similarity_score: float | None = None,
    ) -> None:
        """Not part of GPTCache's interface -- call it once you know whether
        a served hit was actually correct (a thumbs-down, a reopened
        ticket, ...). Feeds `POST /v1/feedback`, which fine-tuning and drift
        monitoring both consume.

        `similarity_score` is optional: pass GPTCache's own vector-search
        score for this candidate if you still have it when you learn
        `was_correct` -- that is what lets the service later recommend a
        tau_high for your GPTCache config (`gray_zone_threshold()`).
        """
        self._cv.feedback(
            query,
            candidate_answer,
            was_correct,
            similarity_score=similarity_score,
        )

    def close(self) -> None:
        self._cv.close()

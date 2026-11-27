"""GPTCache integration: a `SimilarityEvaluation` that calls the hosted
CacheVerifier `/v1/verify` endpoint instead of relying on cosine/dot-product
similarity alone.

GPTCache's `SimilarityEvaluation` interface (`evaluation()` + `range()`) is
a zero-friction integration point: any subclass drops straight into a
GPTCache pipeline's `similarity_evaluation=` argument, no upstream PR
required.

    from gptcache import cache
    from gptcache.similarity_evaluation import SearchDistanceEvaluation
    from cacheverifier.integrations.gptcache import CacheVerifierEvaluation

    cache.init(
        similarity_evaluation=CacheVerifierEvaluation(
            api_key="cv_...",
            fallback=SearchDistanceEvaluation(),  # the evaluator you used before
        ),
        ...
    )

`fallback` is the evaluator this one replaces. With it, GPTCache keeps
deciding exactly as before while your workspace has no model of its own
(cold start), and `direct_hit` can skip the verify call for hits that are
close enough on their own. Without it, every cold-start hit is a miss.

`gptcache` is an optional dependency -- `pip install "cacheverifier[gptcache]"`.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

from cacheverifier.client import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    DEFAULT_VERIFY_TIMEOUT,
    CacheVerifier,
)
from cacheverifier.integrations._cold_start import ColdStartNotice, check_mode
from cacheverifier.integrations._reporting import MAX_PENDING

try:
    from gptcache.similarity_evaluation import SimilarityEvaluation as _GPTCacheBase
except ImportError:  # keep importable without gptcache installed
    _GPTCacheBase = object


class CacheVerifierEvaluation(_GPTCacheBase):
    """Drop-in replacement for GPTCache's built-in similarity evaluators.

    `evaluation()` returns 1.0 (reuse) or 0.0 (don't) for a verified hit --
    binary, because the hosted verifier already makes a binary approve/reject
    call per gray-zone hit rather than a softened similarity score.

    `fallback` (optional, recommended): the GPTCache evaluator you ran before,
    e.g. `SearchDistanceEvaluation()`. GPTCache applies its
    `similarity_threshold` to whatever `evaluation()` returns, so replacing
    your evaluator also drops the similarity check it did. `fallback` puts it
    back where it is needed:

    - cold start (no model of your own yet, so the service decides nothing):
      with `cold_start="serve"` (the default) `evaluation()` returns the
      fallback's own score, rescaled so GPTCache's `similarity_threshold`
      accepts exactly the hits it accepted before. The first such call is
      logged as a warning. `cold_start="reject"` makes every cold-start hit a
      miss instead: do that if you loosened your similarity setup because a
      verifier would be checking the hits. Without a fallback there is no
      earlier decision to keep, so cold-start hits are misses either way.
    - `direct_hit` (needs `fallback`): a hit whose rescaled fallback score is
      at least this is served without calling `/v1/verify`, the way a
      gray-zone setup trusts hits above its upper bound. Same 0-1 scale as
      GPTCache's `similarity_threshold`.

    If the verify call times out or the service is unreachable, `evaluation()`
    returns 0.0 (don't reuse) by default -- a verifier outage falls through to
    a normal GPTCache miss. Pass `fail_open=True` to reuse the cached answer
    instead in that case.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        verify_timeout: float = DEFAULT_VERIFY_TIMEOUT,
        fail_open: bool = False,
        fallback: Any = None,
        cold_start: str = "serve",
        direct_hit: float | None = None,
    ) -> None:
        if direct_hit is not None and fallback is None:
            raise ValueError("direct_hit needs fallback: it is measured on the fallback evaluator's score")
        if fallback is not None:
            low, high = fallback.range()
            if not high > low:
                raise ValueError(f"fallback.range() must be increasing, got {(low, high)!r}")
            self._fallback_span = float(high - low)
        self.fallback = fallback
        self.cold_start = check_mode(cold_start)
        self.direct_hit = direct_hit
        self._cold_start_notice = ColdStartNotice("CacheVerifierEvaluation")
        self._unserved_lock = threading.Lock()
        self._unserved: OrderedDict[tuple[str, str], int] = OrderedDict()
        self._cv = CacheVerifier(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            verify_timeout=verify_timeout,
            fail_open=fail_open,
        )

    def _fallback_score(self, src_dict: dict[str, Any], cache_dict: dict[str, Any], **kwargs: Any) -> float:
        """The fallback's score divided by the width of its range: GPTCache
        accepts a hit when rank >= (max - min) * similarity_threshold, so on
        this evaluator's (0, 1) range the rescaled score passes exactly when
        the raw one passed on the fallback's own range. Exact for any range
        that starts at 0, which every built-in GPTCache evaluator's does."""
        return float(self.fallback.evaluation(src_dict, cache_dict, **kwargs)) / self._fallback_span

    def evaluation(self, src_dict: dict[str, Any], cache_dict: dict[str, Any], **kwargs: Any) -> float:
        fallback_score = None
        if self.fallback is not None:
            fallback_score = self._fallback_score(src_dict, cache_dict, **kwargs)
            if self.direct_hit is not None and fallback_score >= self.direct_hit:
                return 1.0
        query = src_dict.get("question") or src_dict.get("query", "")
        candidate_answer = cache_dict.get("answer", "")
        result = self._cv.verify(query, candidate_answer)
        if not result.approved and result.verification_id is not None:
            self._remember_unserved(query, candidate_answer, result.verification_id)
        if result.cold_start:
            serve = self.cold_start == "serve" and fallback_score is not None
            if serve:
                action = (
                    "deciding hits with your fallback evaluator, as before, until a model is ready "
                    "(cold_start='reject' makes them misses instead)"
                )
            elif self.cold_start == "serve":
                action = (
                    "every hit is a miss until a model is ready; pass fallback=<the evaluator you "
                    "used before> to keep its decisions meanwhile"
                )
            else:
                action = "every hit is a miss until a model is ready (cold_start='reject')"
            self._cold_start_notice.seen(result, action)
            return fallback_score if serve and fallback_score is not None else 0.0
        return 1.0 if result.approved else 0.0

    def _remember_unserved(self, query: str, candidate_answer: str, verification_id: int) -> None:
        with self._unserved_lock:
            self._unserved[(query, candidate_answer)] = verification_id
            self._unserved.move_to_end((query, candidate_answer))
            while len(self._unserved) > MAX_PENDING:
                self._unserved.popitem(last=False)

    def range(self) -> tuple[float, float]:
        return 0.0, 1.0

    def report_feedback(
        self,
        query: str,
        candidate_answer: str,
        was_correct: bool,
        similarity_score: float | None = None,
        verification_id: int | None = None,
    ) -> None:
        """Not part of GPTCache's interface -- call it once you know whether
        a served hit was actually correct (a thumbs-down, a reopened
        ticket, ...). Feeds `POST /v1/feedback`, which fine-tuning and drift
        monitoring both consume.

        `similarity_score` is optional: pass GPTCache's own vector-search
        score for this candidate if you still have it when you learn
        `was_correct` -- that is what lets the service later recommend a
        tau_high for your GPTCache config (`gray_zone_threshold()`).

        Audits and rejections: when this evaluator didn't let a hit through
        (a rejection or a randomized audit), GPTCache calls your LLM. Compare
        its answer with the cached one and report the result here with the
        same `query` and `candidate_answer`: the `verification_id` of that
        verdict is attached for you (it is what completes an audit), unless
        you pass one yourself.
        """
        if verification_id is None:
            with self._unserved_lock:
                verification_id = self._unserved.pop((query, candidate_answer), None)
        self._cv.feedback(
            query,
            candidate_answer,
            was_correct,
            similarity_score=similarity_score,
            verification_id=verification_id,
        )

    def close(self) -> None:
        self._cv.close()

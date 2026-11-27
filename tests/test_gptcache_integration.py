"""The GPTCache adapter must stay importable and usable even when gptcache
itself isn't installed (it's an optional dependency)."""

import httpx

from cacheverifier.integrations.gptcache import CacheVerifierEvaluation


def _eval(handler) -> CacheVerifierEvaluation:
    ev = CacheVerifierEvaluation(api_key="cv_test")
    ev._cv._client = httpx.Client(  # swap in a mock transport
        base_url="https://www.cacheverifier.com",
        headers={"X-API-Key": "cv_test"},
        transport=httpx.MockTransport(handler),
    )
    return ev


def test_evaluation_maps_approved_to_binary_score():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/verify"
        return httpx.Response(
            200,
            json={"approved": True, "score": 2.0, "latency_ms": 5.0, "model_version": "stock", "threshold": 0.0},
        )

    ev = _eval(handler)
    assert ev.evaluation({"question": "q"}, {"answer": "a"}) == 1.0
    assert ev.range() == (0.0, 1.0)


def test_evaluation_rejects_to_zero():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"approved": False, "score": -1.0, "latency_ms": 5.0, "model_version": "stock", "threshold": 0.0},
        )

    assert _eval(handler).evaluation({"query": "q"}, {"answer": "a"}) == 0.0


def test_report_feedback_posts_similarity_score():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": 9})

    _eval(handler).report_feedback("q", "a", was_correct=False, similarity_score=0.9)
    assert seen["path"] == "/v1/feedback"
    assert seen["body"]["similarity_score"] == 0.9
    assert seen["body"]["was_correct"] is False


import logging

import pytest

_COLD_START = {
    "approved": False,
    "score": 0.0,
    "threshold": 0.0,
    "model_version": "cold_start_fail_closed",
    "latency_ms": 0.0,
    "decision_rule": "cold_start",
    "notice": "No model of your own yet.",
}


class _Distance:
    """Stands in for GPTCache's SearchDistanceEvaluation(max_distance=4.0)."""

    def evaluation(self, src_dict, cache_dict, **_kwargs):
        distance, _ = cache_dict["search_result"]
        return 4.0 - min(distance, 4.0)

    def range(self):
        return 0.0, 4.0


def _eval_with(handler, **kwargs) -> tuple[CacheVerifierEvaluation, list]:
    calls: list = []

    def counting(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request)

    ev = CacheVerifierEvaluation(api_key="cv_test", **kwargs)
    ev._cv._client = httpx.Client(
        base_url="https://www.cacheverifier.com",
        headers={"X-API-Key": "cv_test"},
        transport=httpx.MockTransport(counting),
    )
    return ev, calls


def _cold(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=_COLD_START)


def test_cold_start_with_fallback_returns_the_fallbacks_rescaled_score(caplog):
    ev, _ = _eval_with(_cold, fallback=_Distance())
    with caplog.at_level(logging.WARNING, logger="cacheverifier.integrations"):
        # distance 0.8 -> fallback 3.2 of 4.0 -> 0.8; GPTCache's default similarity_threshold 0.8
        # accepted it before (3.2 >= 4.0 * 0.8) and accepts it now (0.8 >= 1.0 * 0.8)
        assert ev.evaluation({"question": "q"}, {"answer": "a", "search_result": (0.8, 1)}) == pytest.approx(0.8)
        assert ev.evaluation({"question": "q"}, {"answer": "a", "search_result": (2.0, 1)}) == pytest.approx(0.5)
    warnings = [r for r in caplog.records if r.name == "cacheverifier.integrations"]
    assert len(warnings) == 1
    assert "fallback evaluator" in warnings[0].getMessage()


def test_cold_start_without_fallback_is_a_miss_and_says_how_to_keep_hits(caplog):
    ev, _ = _eval_with(_cold)
    with caplog.at_level(logging.WARNING, logger="cacheverifier.integrations"):
        assert ev.evaluation({"question": "q"}, {"answer": "a"}) == 0.0
    assert "pass fallback=" in caplog.records[-1].getMessage()


def test_cold_start_reject_ignores_the_fallback():
    ev, _ = _eval_with(_cold, fallback=_Distance(), cold_start="reject")
    assert ev.evaluation({"question": "q"}, {"answer": "a", "search_result": (0.0, 1)}) == 0.0


def test_direct_hit_skips_the_verify_call():
    ev, calls = _eval_with(_cold, fallback=_Distance(), direct_hit=0.95)
    assert ev.evaluation({"question": "q"}, {"answer": "a", "search_result": (0.1, 1)}) == 1.0  # 3.9/4 = 0.975
    assert calls == []
    ev.evaluation({"question": "q"}, {"answer": "a", "search_result": (1.0, 1)})  # 0.75: verified
    assert len(calls) == 1


def test_a_model_verdict_still_decides_when_a_fallback_is_set():
    def rejecting(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"approved": False, "score": -1.0, "latency_ms": 5.0, "model_version": "v3", "threshold": 0.0},
        )

    ev, _ = _eval_with(rejecting, fallback=_Distance())
    assert ev.evaluation({"question": "q"}, {"answer": "a", "search_result": (0.0, 1)}) == 0.0


def test_direct_hit_needs_a_fallback():
    with pytest.raises(ValueError):
        CacheVerifierEvaluation(api_key="cv_test", direct_hit=0.9)


def test_report_feedback_attaches_the_verification_id_of_an_unserved_hit():
    import json

    posted: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/feedback":
            posted.append(json.loads(request.content))
            return httpx.Response(200, json={"id": 1})
        return httpx.Response(
            200,
            json={"approved": False, "audit": True, "verification_id": 41, "score": 2.0, "latency_ms": 5.0,
                  "model_version": "v3", "threshold": 0.0},
        )

    ev, _ = _eval_with(handler)
    assert ev.evaluation({"question": "q"}, {"answer": "a"}) == 0.0
    ev.report_feedback("q", "a", was_correct=True)
    ev.report_feedback("q", "a", was_correct=True)  # already linked: no id the second time
    ev.report_feedback("other", "a", was_correct=False, verification_id=5)
    assert [p.get("verification_id") for p in posted] == [41, None, 5]

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

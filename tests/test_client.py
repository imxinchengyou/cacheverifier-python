"""Offline tests: every request is served by an httpx.MockTransport, so
nothing here touches the network."""

import json

import httpx
import pytest

from cacheverifier import CacheVerifier, CacheVerifierError, VerifyResult


def make_client(handler):
    return CacheVerifier(api_key="cv_test", transport=httpx.MockTransport(handler))


def test_verify_parses_result_and_sends_api_key():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["api_key"] = request.headers.get("X-API-Key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "approved": True,
                "score": 3.5,
                "latency_ms": 24.1,
                "model_version": "stock",
                "threshold": 0.0,
            },
        )

    with make_client(handler) as cv:
        result = cv.verify("how do I cancel", "Go to Settings > Billing.")

    assert isinstance(result, VerifyResult)
    assert result.approved is True
    assert result.score == 3.5
    assert result.threshold == 0.0
    assert result.model_version == "stock"
    assert seen["path"] == "/v1/verify"
    assert seen["api_key"] == "cv_test"
    assert seen["body"] == {"query": "how do I cancel", "candidate_answer": "Go to Settings > Billing."}


def test_verify_batch_returns_ordered_results():
    def handler(request: httpx.Request) -> httpx.Response:
        items = json.loads(request.content)["items"]
        assert request.url.path == "/v1/verify/batch"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "approved": i == 0,
                        "score": 5.0 - i,
                        "latency_ms": 10.0,
                        "model_version": "v7",
                        "threshold": 1.0,
                    }
                    for i, _ in enumerate(items)
                ]
            },
        )

    with make_client(handler) as cv:
        results = cv.verify_batch([("q1", "a1"), ("q2", "a2")])

    assert [r.approved for r in results] == [True, False]
    assert results[0].model_version == "v7"


def test_feedback_returns_id_and_forwards_idempotency_key():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["idem"] = request.headers.get("Idempotency-Key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": 42})

    with make_client(handler) as cv:
        row_id = cv.feedback(
            "q", "a", was_correct=False, similarity_score=0.86, stale=True, idempotency_key="abc-123"
        )

    assert row_id == 42
    assert seen["idem"] == "abc-123"
    assert seen["body"] == {
        "query": "q",
        "candidate_answer": "a",
        "was_correct": False,
        "stale": True,
        "similarity_score": 0.86,
    }


def test_feedback_omits_optional_fields_when_unset():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "similarity_score" not in json.loads(request.content)
        assert request.headers.get("Idempotency-Key") is None
        return httpx.Response(200, json={"id": 1})

    with make_client(handler) as cv:
        assert cv.feedback("q", "a", was_correct=True) == 1


def test_finetune_passes_query_params():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/finetune/jobs"
        assert dict(request.url.params) == {"target_risk": "0.01", "cost_ratio": "5.0"}
        return httpx.Response(200, json={"id": 3, "status": "queued"})

    with make_client(handler) as cv:
        job = cv.finetune(target_risk=0.01, cost_ratio=5.0)

    assert job["status"] == "queued"


def test_error_response_raises_with_detail():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"detail": "free tier limit reached"})

    with make_client(handler) as cv, pytest.raises(CacheVerifierError) as excinfo:
        cv.verify("q", "a")

    assert excinfo.value.status_code == 429
    assert excinfo.value.detail == "free tier limit reached"


def test_error_response_without_json_body_falls_back_to_text():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="Bad Gateway")

    with make_client(handler) as cv, pytest.raises(CacheVerifierError) as excinfo:
        cv.usage()

    assert excinfo.value.status_code == 502
    assert "Bad Gateway" in excinfo.value.detail


def test_empty_api_key_rejected():
    with pytest.raises(ValueError):
        CacheVerifier(api_key="")

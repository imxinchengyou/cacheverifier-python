"""Offline tests: every request is served by an httpx.MockTransport, so
nothing here touches the network."""

import json

import httpx
import pytest

from cacheverifier import CacheVerifier, CacheVerifierError, ShadowVerdict, VerifyResult


def make_client(handler, **kwargs):
    return CacheVerifier(api_key="cv_test", transport=httpx.MockTransport(handler), **kwargs)


_OK_VERIFY = {
    "approved": True,
    "score": 3.5,
    "latency_ms": 24.1,
    "model_version": "stock",
    "threshold": 0.0,
}


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
    assert result.notice is None
    assert result.shadow is None


def test_verify_parses_cold_start_notice_and_shadow_verdict():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "approved": False,
                "score": 0.0,
                "latency_ms": 0.0,
                "model_version": "cold_start_auto_pending",
                "threshold": 0.0,
                "decision_rule": "cold_start",
                "notice": "No verifier has been checked against your data yet.",
                "shadow": {"approved": True, "score": 2.1, "threshold": 0.0, "latency_ms": 18.0, "model_version": "stock"},
            },
        )

    with make_client(handler) as cv:
        result = cv.verify("when does my flight board", "Boarding starts 40 minutes before departure.")

    assert result.approved is False
    assert result.notice.startswith("No verifier")
    assert result.shadow == ShadowVerdict(approved=True, score=2.1, threshold=0.0, latency_ms=18.0)


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
        assert "cache_entry_id" not in json.loads(request.content)
        assert request.headers.get("Idempotency-Key") is None
        return httpx.Response(200, json={"id": 1})

    with make_client(handler) as cv:
        assert cv.feedback("q", "a", was_correct=True) == 1


def test_entries_to_refresh_round_trip():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, request.content))
        if request.url.path == "/v1/feedback":
            return httpx.Response(200, json={"id": 7})
        if request.method == "GET":
            return httpx.Response(200, json={"entries": [{"key": "k" * 32, "cache_entry_id": "faq:shipping", "reason": "stale"}]})
        return httpx.Response(200, json={"key": "k" * 32, "resolved_at": "2026-10-31T00:00:00Z"})

    with make_client(handler) as cv:
        cv.feedback("q", "a", was_correct=False, stale=True, cache_entry_id="faq:shipping")
        [entry] = cv.entries_to_refresh()
        assert entry["cache_entry_id"] == "faq:shipping"
        assert cv.resolve_entry(entry["key"])["key"] == entry["key"]
    assert json.loads(seen[0][2])["cache_entry_id"] == "faq:shipping"
    assert [s[:2] for s in seen[1:]] == [
        ("GET", "/v1/monitor/entries-to-refresh"),
        ("POST", f"/v1/monitor/entries-to-refresh/{'k' * 32}/resolve"),
    ]


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
    assert excinfo.value.retry_after is None


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


# -- request-path timeout + fail-open/closed --------------------------------


def test_verify_uses_the_tight_verify_timeout_not_the_control_plane_one():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["timeout"] = request.extensions.get("timeout")
        return httpx.Response(200, json=_OK_VERIFY)

    with make_client(handler, timeout=10.0, verify_timeout=1.0) as cv:
        cv.verify("q", "a")

    # httpx expands a scalar timeout into per-operation values
    assert set(seen["timeout"].values()) == {1.0}


def test_verify_fails_closed_on_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("read timed out", request=request)

    with make_client(handler) as cv:
        result = cv.verify("q", "a")

    assert result.degraded is True
    assert result.approved is False
    assert result.model_version == "verify_unavailable"


def test_verify_fails_open_when_configured():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with make_client(handler, fail_open=True) as cv:
        result = cv.verify("q", "a")

    assert result.degraded is True
    assert result.approved is True


def test_verify_fails_closed_on_5xx():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream overloaded")

    with make_client(handler) as cv:
        result = cv.verify("q", "a")

    assert result.degraded is True
    assert result.approved is False


def test_verify_fails_closed_on_rate_limit_429():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "1"}, json={"detail": "rate limited"})

    with make_client(handler) as cv:
        result = cv.verify("q", "a")

    assert result.degraded is True
    assert result.approved is False


def test_verify_batch_degrades_on_rate_limit_429():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "1"}, json={"detail": "rate limited"})

    with make_client(handler) as cv:
        results = cv.verify_batch([("q1", "a1"), ("q2", "a2")])

    assert all(r.degraded and not r.approved for r in results)


def test_verify_still_raises_on_other_4xx():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "invalid API key"})

    with make_client(handler) as cv, pytest.raises(CacheVerifierError) as excinfo:
        cv.verify("q", "a")

    assert excinfo.value.status_code == 401


def test_verify_batch_degrades_every_pair_on_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with make_client(handler) as cv:
        results = cv.verify_batch([("q1", "a1"), ("q2", "a2"), ("q3", "a3")])

    assert len(results) == 3
    assert all(r.degraded and not r.approved for r in results)


def test_control_plane_call_still_uses_the_10s_default_and_raises():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["timeout"] = request.extensions.get("timeout")
        return httpx.Response(200, json={"status": "ok"})

    with make_client(handler, timeout=10.0, verify_timeout=1.0) as cv:
        cv.usage()

    assert set(seen["timeout"].values()) == {10.0}

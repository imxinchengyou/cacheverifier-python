"""VerifiedCache against a real langchain-core cache and chat model, with
every verify call served by an httpx.MockTransport."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

pytest.importorskip("langchain_core")

from langchain_core.caches import InMemoryCache
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.outputs import Generation

from cacheverifier import CacheVerifier
from cacheverifier.integrations.langchain import VerifiedCache, user_question


def _verified(approve: bool, seen: list | None = None, inner=None) -> VerifiedCache:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"approved": approve, "score": 1.0, "latency_ms": 5.0, "model_version": "stock", "threshold": 0.0},
        )

    client = CacheVerifier(api_key="cv_test", transport=httpx.MockTransport(handler))
    return VerifiedCache(inner or InMemoryCache(), client=client)


def test_approved_hit_is_served_and_rejected_hit_becomes_a_miss():
    seen: list = []
    cache = _verified(True, seen)
    cache.update("when does my flight board", "llm", [Generation(text="40 minutes before departure.")])
    assert cache.lookup("when does my flight board", "llm")[0].text == "40 minutes before departure."
    assert seen == [{"query": "when does my flight board", "candidate_answer": "40 minutes before departure."}]

    rejecting = _verified(False, inner=cache.cache)
    assert rejecting.lookup("when does my flight board", "llm") is None


def test_a_miss_never_calls_the_verifier():
    seen: list = []
    cache = _verified(True, seen)
    assert cache.lookup("never cached", "llm") is None
    assert seen == []


def test_chat_model_verifies_against_the_last_human_message():
    seen: list = []
    cache = _verified(True, seen)
    model = FakeListChatModel(responses=["Gate B12, boarding at 9:20.", "should not be called"], cache=cache)
    messages = [SystemMessage("You are an airline assistant."), HumanMessage("which gate is my flight")]

    assert model.invoke(messages).content == "Gate B12, boarding at 9:20."  # miss: model called, then cached
    assert model.invoke(messages).content == "Gate B12, boarding at 9:20."  # hit: verified, served
    assert seen == [{"query": "which gate is my flight", "candidate_answer": "Gate B12, boarding at 9:20."}]


def test_rejected_chat_hit_falls_through_to_the_model():
    cache = _verified(False)
    model = FakeListChatModel(responses=["first answer", "fresh answer"], cache=cache)
    assert model.invoke("what is the baggage allowance").content == "first answer"
    assert model.invoke("what is the baggage allowance").content == "fresh answer"


def test_async_lookup_verifies_too():
    cache = _verified(False)
    asyncio.run(cache.aupdate("q", "llm", [Generation(text="a")]))
    assert asyncio.run(cache.alookup("q", "llm")) is None
    assert cache.cache.lookup("q", "llm")[0].text == "a"  # the inner cache still has it


def test_user_question_falls_back_to_the_raw_prompt():
    assert user_question("plain prompt") == "plain prompt"
    assert user_question("[not json") == "[not json"
    multimodal = json.dumps(
        [{"kwargs": {"type": "human", "content": [{"type": "text", "text": "what's in this photo"}, {"type": "image_url"}]}}]
    )
    assert user_question(multimodal) == "what's in this photo"


def test_needs_a_key_or_a_client():
    with pytest.raises(ValueError):
        VerifiedCache(InMemoryCache())


_COLD_START = {
    "approved": False,
    "score": 0.0,
    "threshold": 0.0,
    "model_version": "cold_start_auto_pending",
    "latency_ms": 0.0,
    "decision_rule": "cold_start",
    "notice": "Run a Health Check to let the stock model decide.",
}


def _cold(cold_start: str | None = None) -> VerifiedCache:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_COLD_START)

    client = CacheVerifier(api_key="cv_test", transport=httpx.MockTransport(handler))
    kwargs = {} if cold_start is None else {"cold_start": cold_start}
    return VerifiedCache(InMemoryCache(), client=client, **kwargs)


def test_cold_start_serves_the_wrapped_caches_hit_by_default_and_warns_once(caplog):
    cache = _cold()
    cache.update("when does my flight board", "llm", [Generation(text="40 minutes before departure.")])
    with caplog.at_level("WARNING", logger="cacheverifier.integrations"):
        assert cache.lookup("when does my flight board", "llm")[0].text == "40 minutes before departure."
        assert asyncio.run(cache.alookup("when does my flight board", "llm"))[0].text == "40 minutes before departure."
    warnings = [r for r in caplog.records if r.name == "cacheverifier.integrations"]
    assert len(warnings) == 1
    assert "cold_start_auto_pending" in warnings[0].getMessage()
    assert "Run a Health Check" in warnings[0].getMessage()


def test_cold_start_reject_makes_every_hit_a_miss():
    cache = _cold("reject")
    cache.update("q", "llm", [Generation(text="a")])
    assert cache.lookup("q", "llm") is None


def test_a_stock_model_rejection_is_not_a_cold_start():
    cache = _verified(False)  # model_version "stock": a model ran and said no
    cache.update("q", "llm", [Generation(text="a")])
    assert cache.lookup("q", "llm") is None


def test_an_audited_call_still_falls_through_during_serve_mode():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"approved": False, "audit": True, "verification_id": 7, "score": 3.0, "latency_ms": 5.0,
                  "model_version": "v12", "threshold": 0.0},
        )

    cache = VerifiedCache(InMemoryCache(), client=CacheVerifier(api_key="cv_test", transport=httpx.MockTransport(handler)))
    cache.update("q", "llm", [Generation(text="a")])
    assert cache.lookup("q", "llm") is None


def test_bad_cold_start_mode_is_rejected():
    with pytest.raises(ValueError):
        VerifiedCache(InMemoryCache(), api_key="cv_test", cold_start="open")


def _judged(verdict_json: dict, judge, posted: list) -> VerifiedCache:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/v1/feedback":
            posted.append(body)
            return httpx.Response(200, json={"id": 1})
        return httpx.Response(200, json=verdict_json)

    client = CacheVerifier(api_key="cv_test", transport=httpx.MockTransport(handler))
    return VerifiedCache(InMemoryCache(), client=client, judge=judge)


_AUDIT = {"approved": False, "audit": True, "verification_id": 7, "score": 3.0, "latency_ms": 5.0,
          "model_version": "v12", "threshold": 0.0}


def test_judge_reports_an_audited_hit_with_its_verification_id():
    posted: list = []
    seen: list = []

    def judge(question, cached, fresh):
        seen.append((question, cached, fresh))
        return cached.split(".")[0] == fresh.split(".")[0]

    cache = _judged(_AUDIT, judge, posted)
    model = FakeListChatModel(responses=["Gate B12.", "Gate B12. Boarding 9:20."], cache=cache)
    assert model.invoke("which gate is my flight").content == "Gate B12."  # miss, cached
    assert model.invoke("which gate is my flight").content == "Gate B12. Boarding 9:20."  # audited -> model
    cache.close()
    assert seen == [("which gate is my flight", "Gate B12.", "Gate B12. Boarding 9:20.")]
    assert posted == [{"query": "which gate is my flight", "candidate_answer": "Gate B12.", "was_correct": True,
                       "stale": False, "label_method": "judge", "verification_id": 7}]


def test_judge_reports_a_rejected_hit_in_the_async_path_too():
    posted: list = []
    rejected = {**_AUDIT, "audit": False, "score": -2.0}
    cache = _judged(rejected, lambda q, c, f: False, posted)
    asyncio.run(cache.aupdate("q", "llm", [Generation(text="old")]))
    assert asyncio.run(cache.alookup("q", "llm")) is None
    asyncio.run(cache.aupdate("q", "llm", [Generation(text="new")]))
    cache.close()
    assert [(p["candidate_answer"], p["was_correct"], p["verification_id"]) for p in posted] == [("old", False, 7)]


def test_served_hits_and_skipped_verdicts_are_not_reported():
    posted: list = []
    approved = {**_AUDIT, "approved": True, "audit": False}
    cache = _judged(approved, lambda q, c, f: True, posted)
    cache.update("q", "llm", [Generation(text="a")])
    assert cache.lookup("q", "llm") is not None
    cache.update("q", "llm", [Generation(text="b")])

    skipping = _judged(_AUDIT, lambda q, c, f: None, posted)
    skipping.update("q", "llm", [Generation(text="a")])
    assert skipping.lookup("q", "llm") is None
    skipping.update("q", "llm", [Generation(text="b")])
    cache.close()
    skipping.close()
    assert posted == []


def test_a_failing_judge_is_logged_not_raised(caplog):
    posted: list = []

    def judge(q, c, f):
        raise RuntimeError("judge down")

    cache = _judged(_AUDIT, judge, posted)
    cache.update("q", "llm", [Generation(text="a")])
    assert cache.lookup("q", "llm") is None
    with caplog.at_level("WARNING", logger="cacheverifier.integrations"):
        cache.update("q", "llm", [Generation(text="b")])  # must not raise
        cache.close()
    assert posted == []
    assert "judge down" in caplog.text

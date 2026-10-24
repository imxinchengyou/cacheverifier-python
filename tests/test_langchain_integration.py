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

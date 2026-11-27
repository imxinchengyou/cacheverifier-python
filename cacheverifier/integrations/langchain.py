"""LangChain integration: wrap any LangChain semantic cache so a hit is only
served after CacheVerifier approves it.

A semantic cache (`RedisSemanticCache`, `MongoDBAtlasSemanticCache`, ...)
returns a stored answer whenever a past prompt is similar enough to the new
one, including when the wording differs in a way that changes the answer.
`VerifiedCache` sits in front of it: on a hit it asks `/v1/verify` whether
the cached answer fits the new question, and turns a rejected hit into a
miss, so your model is called as usual.

    from langchain_core.globals import set_llm_cache
    from langchain_redis import RedisSemanticCache
    from cacheverifier.integrations.langchain import VerifiedCache

    set_llm_cache(
        VerifiedCache(
            RedisSemanticCache(redis_url="redis://localhost:6379", embeddings=embeddings),
            api_key="cv_...",
        )
    )

Writes, clears and exact-match caches pass straight through. If the verify
call times out or the service is unreachable, the hit is rejected by
default (a verifier outage becomes a normal cache miss); pass
`fail_open=True` to serve it instead.

Until your workspace has a model of its own, the service decides nothing
(cold start) and would reject every hit, which switches your cache off. So
by default a cold-start hit is served exactly as the wrapped cache returned
it, the same as without `VerifiedCache`, and the first one is logged as a
warning. Pass `cold_start="reject"` to fall through to your model instead:
do that if you loosened the wrapped cache's similarity threshold because a
verifier would be checking its hits.

Pass `judge=` to close the loop on hits that weren't served. When a hit is
rejected or picked for an audit, your model answers anyway; `VerifiedCache`
sees that answer when LangChain stores it, calls
`judge(question, cached_answer, fresh_answer)` and reports the verdict as
feedback (True: the cached answer would have been right, False: it would
not, None: skip), as `label_method="judge"`. Those labels train fine-tunes;
the service keeps them out of its certified numbers (an audit answered only
by the judge still counts as unreported there). It runs in a background
thread, off your request path.

    def same_answer(question, cached, fresh):
        return my_llm_judge(question, cached, fresh)  # your own check

    VerifiedCache(cache, api_key="cv_...", judge=same_answer)

`langchain-core` is an optional dependency -- `pip install "cacheverifier[langchain]"`.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from cacheverifier.client import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    DEFAULT_VERIFY_TIMEOUT,
    CacheVerifier,
)
from cacheverifier.integrations._cold_start import ColdStartNotice, check_mode
from cacheverifier.integrations._reporting import Judge, RejectedHitReporter

try:
    from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        'cacheverifier.integrations.langchain needs langchain-core: pip install "cacheverifier[langchain]"'
    ) from exc


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # multimodal content blocks
        return "\n".join(
            block if isinstance(block, str) else str(block.get("text", ""))
            for block in content
            if isinstance(block, str) or (isinstance(block, dict) and block.get("type") == "text")
        )
    return ""


def user_question(prompt: str) -> str:
    """The text to verify against for a LangChain cache key.

    Plain LLMs use the prompt string itself as the key. Chat models use a
    JSON dump of the whole message list, system prompt and history
    included; the question being asked is the last human message, so that
    is what the verifier sees. Anything unrecognized falls back to the raw
    prompt."""
    if not prompt.startswith("["):
        return prompt
    try:
        messages = json.loads(prompt)
    except ValueError:
        return prompt
    for message in reversed(messages if isinstance(messages, list) else []):
        kwargs = message.get("kwargs", {}) if isinstance(message, dict) else {}
        if kwargs.get("type") == "human":
            text = _message_text(kwargs.get("content"))
            if text:
                return text
    return prompt


def _text(generations: RETURN_VAL_TYPE) -> str:
    return "\n".join(generation.text for generation in generations)


class VerifiedCache(BaseCache):
    """A LangChain cache that verifies another cache's hits before serving them."""

    def __init__(
        self,
        cache: BaseCache,
        api_key: str | None = None,
        *,
        client: CacheVerifier | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        verify_timeout: float = DEFAULT_VERIFY_TIMEOUT,
        fail_open: bool = False,
        cold_start: str = "serve",
        judge: Judge | None = None,
    ) -> None:
        self.cold_start = check_mode(cold_start)
        self._cold_start_notice = ColdStartNotice("VerifiedCache")
        if client is None:
            if not api_key:
                raise ValueError("pass api_key or client")
            client = CacheVerifier(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                verify_timeout=verify_timeout,
                fail_open=fail_open,
            )
        self.cache = cache
        self.client = client
        self._reporter = RejectedHitReporter(client, judge) if judge is not None else None

    def _approved(self, prompt: str, llm_string: str, hit: RETURN_VAL_TYPE) -> bool:
        answer = _text(hit)
        question = user_question(prompt)
        result = self.client.verify(question, answer)
        served = self._served(result)
        if not served and self._reporter is not None:
            self._reporter.remember((prompt, llm_string), question, answer, result)
        return served

    def _served(self, result: Any) -> bool:
        if result.cold_start:
            serve = self.cold_start == "serve"
            self._cold_start_notice.seen(
                result,
                "serving the wrapped cache's hits unchanged until a model is ready "
                "(cold_start='reject' falls through to your model instead)"
                if serve
                else "every hit falls through to your model until one is ready (cold_start='reject')",
            )
            return serve
        return result.approved

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        hit = self.cache.lookup(prompt, llm_string)
        if not hit:
            return hit
        return hit if self._approved(prompt, llm_string, hit) else None

    async def alookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        hit = await self.cache.alookup(prompt, llm_string)
        if not hit:
            return hit
        approved = await asyncio.get_running_loop().run_in_executor(None, self._approved, prompt, llm_string, hit)
        return hit if approved else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        self.cache.update(prompt, llm_string, return_val)
        if self._reporter is not None:
            self._reporter.fresh_answer((prompt, llm_string), _text(return_val))

    async def aupdate(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        await self.cache.aupdate(prompt, llm_string, return_val)
        if self._reporter is not None:
            self._reporter.fresh_answer((prompt, llm_string), _text(return_val))

    def close(self) -> None:
        """Wait for queued `judge` reports to finish (only needed with `judge=`)."""
        if self._reporter is not None:
            self._reporter.close()

    def clear(self, **kwargs: Any) -> None:
        self.cache.clear(**kwargs)

    async def aclear(self, **kwargs: Any) -> None:
        await self.cache.aclear(**kwargs)

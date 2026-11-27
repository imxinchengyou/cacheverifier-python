"""Closing the loop on hits an integration didn't serve.

When a verified hit is rejected or picked for an audit, the caller's model
answers the question anyway. That fresh answer is what tells you whether the
cached one would have been fine: the label an audit needs to complete, and
the rejected-side label fine-tuning otherwise rarely gets. An integration
that sees the fresh answer (LangChain's `update()` after a miss) can hand
both answers to a caller-supplied judge and report its verdict.

The judge is yours on purpose. Whether two answers are "the same" depends on
your domain, and the cheap automatic checks we measured (answer embeddings,
an NLI model) did no better than the query similarity you already have.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from cacheverifier.client import VerifyResult
from cacheverifier.local import LocalVerifier

Judge = Callable[[str, str, str], "bool | None"]
"""`judge(question, cached_answer, fresh_answer)`: True if the cached answer
would have been right for this question, False if not, None to skip."""

MAX_PENDING = 1024
"""Unserved hits remembered while waiting for the fresh answer. Oldest are
dropped first: a fresh answer that never arrives (the model call failed)
must not grow memory without bound."""

_log = logging.getLogger("cacheverifier.integrations")


@dataclass(frozen=True)
class _Pending:
    question: str
    cached_answer: str
    result: VerifyResult


class RejectedHitReporter:
    """Remembers unserved hits by key and, when the fresh answer for the same
    key arrives, reports the judge's verdict as feedback. Never raises into
    the caller's request path: judge and report errors are logged."""

    def __init__(self, client: Any, judge: Judge, *, background: bool = True) -> None:
        self._client = client
        self._judge = judge
        self._lock = threading.Lock()
        self._pending: OrderedDict[Hashable, _Pending] = OrderedDict()
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="cacheverifier-report") if background else None
        )

    def remember(self, key: Hashable, question: str, cached_answer: str, result: VerifyResult) -> None:
        """Call for a hit that was not served: rejected by a model, an audit,
        or a cold-start rejection. Outages (`degraded`) carry no verdict and
        are not reported."""
        if result.degraded or result.approved:
            return
        with self._lock:
            self._pending[key] = _Pending(question, cached_answer, result)
            self._pending.move_to_end(key)
            while len(self._pending) > MAX_PENDING:
                self._pending.popitem(last=False)

    def fresh_answer(self, key: Hashable, fresh: str) -> Future[None] | None:
        """Call with the answer the model just produced for `key`."""
        with self._lock:
            pending = self._pending.pop(key, None)
        if pending is None:
            return None
        if self._executor is None:
            self._report(pending, fresh)
            return None
        return self._executor.submit(self._report, pending, fresh)

    def _report(self, pending: _Pending, fresh: str) -> None:
        try:
            verdict = self._judge(pending.question, pending.cached_answer, fresh)
        except Exception as exc:  # noqa: BLE001 -- the judge is caller code; never raise into the request path
            _log.warning("cacheverifier: judge raised %r; hit not reported", exc)
            return
        if verdict is None:
            return
        try:
            if isinstance(self._client, LocalVerifier):  # links by result, not by id
                self._client.feedback(
                    pending.question,
                    pending.cached_answer,
                    bool(verdict),
                    result=pending.result,
                    label_method="judge",
                )
            else:
                self._client.feedback(
                    pending.question,
                    pending.cached_answer,
                    bool(verdict),
                    verification_id=pending.result.verification_id,
                    label_method="judge",
                )
        except Exception as exc:  # noqa: BLE001 -- a failed report must not break the caller
            _log.warning("cacheverifier: reporting an unserved hit failed: %s", exc)

    def close(self) -> None:
        """Wait for queued reports to finish."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)

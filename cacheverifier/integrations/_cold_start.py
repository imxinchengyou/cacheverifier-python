"""What an integration does with a cold-start verdict.

While a workspace has no model of its own, `/v1/verify` runs fail-closed and
rejects every call. Rejecting a call the caller only sends in its gray zone
is cheap. An integration, though, sends every hit its cache backend
returns, and rejecting all of them switches the cache off: the hit rate
drops to zero, LLM spend goes up, and nothing is protected that wasn't
already being served before CacheVerifier was added. So by default an
integration hands a cold-start call back to the cache's own decision, and
says so once in the log.
"""

from __future__ import annotations

import logging
import threading

from cacheverifier.client import VerifyResult

COLD_START_MODES = ("serve", "reject")

_log = logging.getLogger("cacheverifier.integrations")


def check_mode(cold_start: str) -> str:
    if cold_start not in COLD_START_MODES:
        raise ValueError(f"cold_start must be one of {COLD_START_MODES}, got {cold_start!r}")
    return cold_start


class ColdStartNotice:
    """Logs the first cold-start verdict an integration instance sees, once."""

    def __init__(self, integration: str) -> None:
        self._integration = integration
        self._lock = threading.Lock()
        self._logged = False

    def seen(self, result: VerifyResult, action: str) -> None:
        """`action`: what the integration does with cold-start hits, in a
        phrase ("serving ... unchanged until a model is ready")."""
        with self._lock:
            if self._logged:
                return
            self._logged = True
        _log.warning(
            "cacheverifier %s: no model decides your calls yet (%s) -- %s. %s",
            self._integration,
            result.model_version,
            action,
            result.notice or "Run a Health Check at https://www.cacheverifier.com to get one.",
        )

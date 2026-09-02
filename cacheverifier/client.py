"""Thin Python client for the hosted CacheVerifier API (https://www.cacheverifier.com).

CacheVerifier does not run your semantic cache or do similarity search. Your
cache backend does its own lookup first; you call `verify()` only on the
candidates in the similarity "gray zone", where a plain threshold match
might be wrong. See https://www.cacheverifier.com/docs for the full API.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://www.cacheverifier.com"
DEFAULT_TIMEOUT = 10.0

__all__ = ["CacheVerifier", "CacheVerifierError", "VerifyResult"]


class CacheVerifierError(RuntimeError):
    """Raised for any non-2xx response from the API.

    `status_code` is the HTTP status; `detail` is the server's error message
    (the JSON body's `detail` field when present, otherwise the raw text).
    """

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"CacheVerifier API error {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class VerifyResult:
    """One `/v1/verify` decision.

    - `approved`: serve the cached answer (True) or fall through to your LLM (False).
    - `score`: the verifier's raw score for this pair; `approved` is `score >= threshold`.
    - `threshold`: the cutoff this call was decided against (tenant-specific once
      you've fine-tuned; 0.0 on the shared stock model).
    - `model_version`: `"stock"`, `"v<id>"` for a fine-tuned model, or
      `"cold_start_fail_closed"` when no model ran (see `cold_start_mode`).
    - `latency_ms`: server-side model inference time, not round-trip time.
    """

    approved: bool
    score: float
    threshold: float
    model_version: str
    latency_ms: float

    @classmethod
    def _from_json(cls, d: dict[str, Any]) -> VerifyResult:
        return cls(
            approved=bool(d["approved"]),
            score=float(d["score"]),
            threshold=float(d["threshold"]),
            model_version=str(d["model_version"]),
            latency_ms=float(d["latency_ms"]),
        )


class CacheVerifier:
    """Client for the hosted CacheVerifier API.

        from cacheverifier import CacheVerifier

        cv = CacheVerifier(api_key="cv_...")
        result = cv.verify("how do I cancel", "Go to Settings > Billing > Pause.")
        if result.approved:
            ...  # serve the cached answer
        else:
            ...  # fall through to your LLM

    Usable as a context manager (`with CacheVerifier(...) as cv:`) to close
    the underlying HTTP connection pool deterministically.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required -- get one at https://www.cacheverifier.com")
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"X-API-Key": api_key, "User-Agent": _user_agent()},
            timeout=timeout,
            transport=transport,
        )

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> CacheVerifier:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- core: verify ----------------------------------------------------

    def verify(self, query: str, candidate_answer: str) -> VerifyResult:
        """Approve or reject one gray-zone cache hit. `POST /v1/verify`."""
        data = self._post("/v1/verify", json={"query": query, "candidate_answer": candidate_answer})
        return VerifyResult._from_json(data)

    def verify_batch(self, pairs: Sequence[tuple[str, str]]) -> list[VerifyResult]:
        """Verify many `(query, candidate_answer)` pairs in one request and
        one batched forward pass. `POST /v1/verify/batch` (max 100 items).

        Useful when your own retrieval returns several close candidates:
        send them in rank order and take the first `approved` one.
        """
        items = [{"query": q, "candidate_answer": a} for q, a in pairs]
        data = self._post("/v1/verify/batch", json={"items": items})
        return [VerifyResult._from_json(r) for r in data["results"]]

    # -- feedback ------------------------------------------------------

    def feedback(
        self,
        query: str,
        candidate_answer: str,
        was_correct: bool,
        *,
        similarity_score: float | None = None,
        stale: bool = False,
        idempotency_key: str | None = None,
    ) -> int:
        """Record whether a served (or considered) hit was actually correct.
        Returns the created row id. `POST /v1/feedback`.

        This is the ground-truth signal fine-tuning and drift monitoring
        learn from -- you supply it from a thumbs-down, a reopened ticket,
        manual review, etc.

        - `similarity_score`: your cache backend's own score for this
          candidate, if you still have it. Only used by
          `gray_zone_threshold()`; safe to omit.
        - `stale`: set instead of a bare `was_correct=False` when the answer
          is wrong ONLY because a fact changed (price, date, ...), not a
          semantic mismatch. Stale rows are excluded from training and drift.
        - `idempotency_key`: pass a stable key to make retries safe.
        """
        payload: dict[str, Any] = {
            "query": query,
            "candidate_answer": candidate_answer,
            "was_correct": was_correct,
            "stale": stale,
        }
        if similarity_score is not None:
            payload["similarity_score"] = similarity_score
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        return int(self._post("/v1/feedback", json=payload, headers=headers)["id"])

    def feedback_batch(self, items: Iterable[dict[str, Any]]) -> list[int]:
        """Bulk-upload feedback rows (max 500). Each item is a dict with
        `query`, `candidate_answer`, `was_correct`, and optionally
        `similarity_score` / `stale`. Returns the created row ids.
        `POST /v1/feedback/batch`.
        """
        data = self._post("/v1/feedback/batch", json={"items": list(items)})
        return [int(i) for i in data["ids"]]

    # -- fine-tuning -------------------------------------------------

    def finetune(self, *, target_risk: float | None = None, cost_ratio: float | None = None) -> dict[str, Any]:
        """Kick off fine-tuning on all feedback submitted so far. Returns the
        job (poll `get_finetune_job(job['id'])` for completion + AUC).
        `POST /v1/finetune/jobs`.

        - `target_risk`: e.g. `0.01` to also certify a threshold at 1%
          false-reuse risk via Conformal Risk Control.
        - `cost_ratio`: pick the live threshold by cost (how many times more
          a wrong reuse costs than a miss) instead of the default Youden's J.
        """
        return self._post("/v1/finetune/jobs", params=_drop_none(target_risk=target_risk, cost_ratio=cost_ratio))

    def dry_run(
        self,
        examples: Sequence[dict[str, Any]],
        *,
        target_risk: float | None = None,
        cost_ratio: float | None = None,
    ) -> dict[str, Any]:
        """Baseline-vs-fine-tuned AUC on `examples` given directly here --
        nothing is written to your feedback history and no model is
        deployed. Each example is a dict with `query`, `candidate_answer`,
        `was_correct` (and optional `stale`). `POST /v1/finetune/dry-run`.
        """
        return self._post(
            "/v1/finetune/dry-run",
            json={"examples": list(examples)},
            params=_drop_none(target_risk=target_risk, cost_ratio=cost_ratio),
        )

    def get_finetune_job(self, job_id: int) -> dict[str, Any]:
        """`GET /v1/finetune/jobs/{id}`."""
        return self._get(f"/v1/finetune/jobs/{job_id}")

    def list_finetune_jobs(self) -> list[dict[str, Any]]:
        """`GET /v1/finetune/jobs`."""
        return self._get("/v1/finetune/jobs")

    def activate_model_version(self, model_version_id: int) -> dict[str, Any]:
        """Promote a model that finished as `held_for_review`.
        `POST /v1/finetune/model-versions/{id}/activate`.
        """
        return self._post(f"/v1/finetune/model-versions/{model_version_id}/activate")

    # -- monitoring / usage ----------------------------------------

    def drift_status(self) -> dict[str, Any]:
        """`GET /v1/monitor/drift-status` (needs an active fine-tuned model)."""
        return self._get("/v1/monitor/drift-status")

    def gray_zone_threshold(self) -> dict[str, Any]:
        """Recommend a tau_high for YOUR cache backend, replaying feedback
        rows that included `similarity_score`. `GET /v1/monitor/gray-zone-threshold`.
        """
        return self._get("/v1/monitor/gray-zone-threshold")

    def usage(self) -> dict[str, Any]:
        """`GET /v1/usage/status`."""
        return self._get("/v1/usage/status")

    def savings(self) -> dict[str, Any]:
        """This period's estimated LLM calls and wrong hits avoided.
        `GET /v1/usage/savings`.
        """
        return self._get("/v1/usage/savings")

    # -- plumbing ----------------------------------------------------

    def _get(self, path: str) -> Any:
        return self._unwrap(self._client.get(path))

    def _post(
        self,
        path: str,
        *,
        json: Any | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        return self._unwrap(self._client.post(path, json=json, params=params or None, headers=headers))

    @staticmethod
    def _unwrap(resp: httpx.Response) -> Any:
        if resp.is_success:
            return resp.json()
        detail: str
        try:
            body = resp.json()
            detail = body["detail"] if isinstance(body, dict) and "detail" in body else resp.text
        except ValueError:
            detail = resp.text
        raise CacheVerifierError(resp.status_code, str(detail))


def _drop_none(**kwargs: Any) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}


def _user_agent() -> str:
    from cacheverifier import __version__

    return f"cacheverifier-python/{__version__} httpx/{httpx.__version__}"

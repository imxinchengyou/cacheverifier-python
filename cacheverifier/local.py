"""Score gray-zone cache hits on your own machines with your fine-tuned model.

`LocalVerifier` downloads the model CacheVerifier fine-tuned for your account
(Managed plans), decides locally with the same threshold and rules the hosted
`/v1/verify` would use, and keeps the hosted side of the loop running:

- every few minutes it checks which model is live, and switches to a newer
  one once it has downloaded and checked it;
- while auditing is on, it draws audits locally and uploads each decision
  (scores, verdicts and a keyed hash of the pair; never the texts), so audits,
  `/v1/monitor/risk` and feedback linking work as with the hosted API;
- feedback goes to the service as usual, for fine-tuning and drift monitoring.

    from cacheverifier.local import LocalVerifier

    cv = LocalVerifier(api_key="cv_...")
    r = cv.verify(query, candidate_answer, similarity_score=0.87)
    if r.approved:
        ...  # serve from cache
    else:
        ...  # call your LLM; if r.audit, report: cv.feedback(query, candidate_answer, ok, result=r)

Before a model is ready (first download, no fine-tune yet, or a model that
fails its check), calls go to the hosted API instead (`fallback="hosted"`,
counted against your verify quota) or are rejected (`fallback="reject"`).

Needs the `local` extra: `pip install "cacheverifier[local]"`.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import random
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from cacheverifier._inference import JointRule, compose_query_input, decide, smart_truncate_pair
from cacheverifier.client import (
    DEFAULT_BASE_URL,
    VERIFY_UNAVAILABLE,
    CacheVerifier,
    CacheVerifierError,
    VerifyResult,
)

_log = logging.getLogger(__name__)

SUPPORTED_BUNDLE_FORMAT = 1
DEFAULT_REFRESH_INTERVAL = 600.0
"""Seconds between checks for a newer live model."""
DEFAULT_UPLOAD_INTERVAL = 30.0
"""Seconds between decision uploads (sooner when a batch fills up)."""
UPLOAD_BATCH = 1000
"""The service's per-request cap on `POST /v1/verify/decisions`."""
MAX_PENDING_DECISIONS = 100_000
"""Decisions kept while the service can't be reached; the oldest are dropped
(and logged) beyond this, so an outage can't grow memory without bound."""
MAX_REMEMBERED_IDS = 200_000
KEEP_MODEL_VERSIONS = 2
"""Downloaded versions kept on disk: the live one and the one before it."""
REFERENCE_SCORE_TOLERANCE = 1e-3
"""Allowed gap between a local score and the service's on a reference pair
(relative above 1). CPU builds of the same model agree far closer than this;
a mismatch means different inputs or a different model, not rounding."""

__all__ = ["LocalVerifier", "ModelCheckFailed"]


class ModelCheckFailed(RuntimeError):
    """A downloaded model didn't reproduce the service's reference verdicts."""


def _default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "cacheverifier" / "models"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class _Model:
    manifest: dict[str, Any]
    encoder: Any
    joint_rule: JointRule | None
    joint_rule_gap: JointRule | None
    lock: threading.Lock

    @property
    def version_id(self) -> int:
        return int(self.manifest["model_version_id"])

    def scores(self, queries: Sequence[str], answers: Sequence[str], cached: Sequence[str | None]) -> list[float]:
        use_cached = bool(self.manifest["uses_cached_query"])
        with self.lock:  # a Hugging Face fast tokenizer is not thread-safe
            pairs = [
                smart_truncate_pair(compose_query_input(q, c if use_cached else None), a, self.encoder.tokenizer)
                for q, a, c in zip(queries, answers, cached)
            ]
            return [float(s) for s in self.encoder.predict(pairs, show_progress_bar=False)]

    def decide(self, score: float, sim: float | None, runner_up: float | None) -> tuple[bool, str]:
        return decide(score, float(self.manifest["threshold"]), self.joint_rule, self.joint_rule_gap, sim, runner_up)


class LocalVerifier:
    """See the module docstring. Thread-safe; create one per process and
    reuse it. `close()` (or a `with` block) uploads what's pending."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        cache_dir: str | os.PathLike[str] | None = None,
        fallback: str = "hosted",
        refresh_interval: float = DEFAULT_REFRESH_INTERVAL,
        upload_interval: float = DEFAULT_UPLOAD_INTERVAL,
        wait: bool = False,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        background: bool = True,
    ) -> None:
        """- `fallback`: `"hosted"` sends calls to `/v1/verify` while no local
          model is ready; `"reject"` returns a rejected, `degraded` result.
        - `wait`: block until the first model is downloaded and checked (a
          checkpoint is ~330MB) instead of starting on the fallback.
        - `background`: run the refresh/upload thread. With False, call
          `refresh()` and `flush()` yourself.
        """
        if fallback not in ("hosted", "reject"):
            raise ValueError('fallback must be "hosted" or "reject"')
        self._fallback = fallback
        self._hosted = CacheVerifier(api_key, base_url=base_url, timeout=timeout, transport=transport)
        host = urlparse(base_url).netloc or "default"
        self._cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir() / host
        self._refresh_interval = refresh_interval
        self._upload_interval = upload_interval
        self._model: _Model | None = None
        self._state_lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        self._pending: list[tuple[str, int, dict[str, Any]]] = []
        self._ids: OrderedDict[str, int] = OrderedDict()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        if wait:
            self.refresh()
        if background:
            self._thread = threading.Thread(target=self._run, name="cacheverifier-local", daemon=True)
            self._thread.start()

    # -- lifecycle -------------------------------------------------------

    @property
    def model_version(self) -> str | None:
        """`"v<id>"` of the model deciding locally, or None while on the fallback."""
        model = self._model
        return f"v{model.version_id}" if model is not None else None

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
        try:
            self.flush()
        finally:
            self._hosted.close()

    def __enter__(self) -> LocalVerifier:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _run(self) -> None:
        next_refresh = 0.0 if self._model is None else time.monotonic() + self._refresh_interval
        while not self._stop.is_set():
            if time.monotonic() >= next_refresh:
                try:
                    self.refresh()
                except Exception as exc:  # noqa: BLE001 -- keep the loop alive; the current model keeps deciding
                    _log.warning("cacheverifier: model refresh failed (%s); will retry", exc)
                # Retry sooner while there's no local model at all.
                next_refresh = time.monotonic() + (self._refresh_interval if self._model else 60.0)
            try:
                self.flush()
            except Exception as exc:  # noqa: BLE001 -- pending decisions stay queued for the next try
                _log.warning("cacheverifier: decision upload failed (%s); will retry", exc)
            self._wake.wait(self._upload_interval)
            self._wake.clear()

    # -- model download, check, switch -----------------------------------

    def refresh(self) -> bool:
        """Switch to the live model if it's newer than the local one. Returns
        whether a new model was loaded. Raises `ModelCheckFailed` when the
        new model doesn't reproduce the reference verdicts (the old one, or
        the fallback, keeps deciding)."""
        with self._refresh_lock:
            try:
                manifest = self._hosted._get("/v1/models/active")
            except CacheVerifierError as exc:
                if exc.status_code in (402, 404):
                    _log.warning("cacheverifier: no model to run locally (%s): %s", exc.status_code, exc.detail)
                    return False
                raise
            current = self._model
            if current is not None and current.version_id == manifest["model_version_id"]:
                current.manifest.update(audit_rate=manifest["audit_rate"], pair_key=manifest["pair_key"])
                return False
            if manifest.get("format") != SUPPORTED_BUNDLE_FORMAT:
                raise ModelCheckFailed(
                    f"model format {manifest.get('format')} is newer than this client; upgrade cacheverifier"
                )
            if manifest.get("aci_enabled"):
                _log.warning(
                    "cacheverifier: ACI is on for this account, but local scoring decides with the model's fixed "
                    "threshold; turn ACI off or use the hosted API for the online threshold"
                )
            path = self._download(manifest)
            model = self._load(manifest, path)
            self._check(model)
            with self._state_lock:
                self._model = model
            self._prune(keep=path)
            _log.info("cacheverifier: now deciding locally with v%s", model.version_id)
            return True

    def _version_dir(self, manifest: dict[str, Any]) -> Path:
        files_digest = hashlib.sha256("".join(f["sha256"] for f in manifest["files"]).encode()).hexdigest()[:12]
        return self._cache_dir / f"v{manifest['model_version_id']}-{files_digest}"

    def _download(self, manifest: dict[str, Any]) -> Path:
        target = self._version_dir(manifest)
        target.mkdir(parents=True, exist_ok=True)
        for f in manifest["files"]:
            dest = target / f["name"]
            if dest.exists() and dest.stat().st_size == f["size"] and _sha256(dest) == f["sha256"]:
                continue
            part = target / (f["name"] + ".part")
            url = f"/v1/models/{manifest['model_version_id']}/files/{f['name']}"
            for _attempt in range(3):  # resume an interrupted transfer with a Range request
                done = part.stat().st_size if part.exists() else 0
                if done >= f["size"]:
                    break
                headers = {"Range": f"bytes={done}-"} if done else {}
                with self._hosted._client.stream("GET", url, headers=headers, timeout=None) as resp:
                    if resp.status_code == 429:
                        raise CacheVerifierError(429, "model downloads busy", resp.headers.get("Retry-After"))
                    if resp.status_code not in (200, 206):
                        resp.read()
                        CacheVerifier._unwrap(resp)
                    mode = "ab" if resp.status_code == 206 else "wb"
                    with part.open(mode) as out:
                        for chunk in resp.iter_bytes(1 << 20):
                            out.write(chunk)
            if part.stat().st_size != f["size"] or _sha256(part) != f["sha256"]:
                part.unlink(missing_ok=True)
                raise ModelCheckFailed(f"downloaded {f['name']} doesn't match its sha256; will retry")
            part.replace(dest)
        return target

    @staticmethod
    def _load(manifest: dict[str, Any], path: Path) -> _Model:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError('local scoring needs the local extra: pip install "cacheverifier[local]"') from exc
        encoder = CrossEncoder(str(path), max_length=int(manifest["max_sequence_length"]))
        rule = manifest.get("joint_rule")
        rule_gap = manifest.get("joint_rule_gap")
        return _Model(
            manifest=dict(manifest),
            encoder=encoder,
            joint_rule=JointRule.from_dict(rule) if rule else None,
            joint_rule_gap=JointRule.from_dict(rule_gap) if rule_gap else None,
            lock=threading.Lock(),
        )

    @staticmethod
    def _check(model: _Model) -> None:
        refs = model.manifest["reference"]
        scores = model.scores(
            [r["query"] for r in refs], [r["candidate_answer"] for r in refs], [r.get("cached_query") for r in refs]
        )
        for ref, score in zip(refs, scores):
            expected = float(ref["score"])
            if abs(score - expected) > REFERENCE_SCORE_TOLERANCE * max(1.0, abs(expected)):
                raise ModelCheckFailed(
                    f"v{model.version_id}: local score {score:.6f} != service {expected:.6f} on a reference pair; "
                    "not using this model (check that torch / sentence-transformers are supported versions)"
                )
            # The decision logic, on the service's own score: catches a rule
            # this client reads differently even where the scores agree.
            verdict = model.decide(expected, ref.get("similarity_score"), ref.get("runner_up_similarity_score"))
            if verdict != (ref["approved"], ref["decision_rule"]):
                raise ModelCheckFailed(
                    f"v{model.version_id}: decision rule mismatch on a reference pair; upgrade cacheverifier"
                )

    def _prune(self, keep: Path) -> None:
        dirs = sorted(
            (d for d in self._cache_dir.iterdir() if d.is_dir() and d != keep),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        for old in dirs[KEEP_MODEL_VERSIONS - 1 :]:
            shutil.rmtree(old, ignore_errors=True)

    # -- verify ------------------------------------------------------------

    def verify(
        self,
        query: str,
        candidate_answer: str,
        *,
        similarity_score: float | None = None,
        cached_query: str | None = None,
        runner_up_similarity_score: float | None = None,
    ) -> VerifyResult:
        """Same arguments and result as `CacheVerifier.verify`."""
        return self.verify_batch(
            [(query, candidate_answer)],
            similarity_scores=[similarity_score],
            cached_queries=[cached_query],
            runner_up_similarity_scores=[runner_up_similarity_score],
        )[0]

    def verify_batch(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        similarity_scores: Sequence[float | None] | None = None,
        cached_queries: Sequence[str | None] | None = None,
        runner_up_similarity_scores: Sequence[float | None] | None = None,
    ) -> list[VerifyResult]:
        n = len(pairs)
        sims = list(similarity_scores) if similarity_scores is not None else [None] * n
        cached = list(cached_queries) if cached_queries is not None else [None] * n
        runner_ups = list(runner_up_similarity_scores) if runner_up_similarity_scores is not None else [None] * n
        model = self._model
        if model is None:
            return [self._fallback_verify(p, s, c, r) for p, s, c, r in zip(pairs, sims, cached, runner_ups)]

        start = time.perf_counter()
        scores = model.scores([q for q, _ in pairs], [a for _, a in pairs], cached)
        latency_ms = (time.perf_counter() - start) * 1000.0 / max(1, n)
        audit_rate = float(model.manifest["audit_rate"])
        results = []
        for (query, answer), score, sim, runner_up in zip(pairs, scores, sims, runner_ups):
            ok, rule = model.decide(score, sim, runner_up)
            audited = ok and audit_rate > 0.0 and random.random() < audit_rate
            token = self._queue(model, query, answer, score, rule, ok, audited, audit_rate) if audit_rate > 0 else None
            results.append(
                VerifyResult(
                    approved=ok and not audited,
                    score=score,
                    threshold=float(model.manifest["threshold"]),
                    model_version=f"v{model.version_id}",
                    latency_ms=latency_ms,
                    decision_rule=rule,
                    audit=audited,
                    local_decision=token,
                )
            )
        return results

    def _fallback_verify(
        self, pair: tuple[str, str], sim: float | None, cached: str | None, runner_up: float | None
    ) -> VerifyResult:
        if self._fallback == "hosted":
            return self._hosted.verify(
                pair[0], pair[1], similarity_score=sim, cached_query=cached, runner_up_similarity_score=runner_up
            )
        return VerifyResult(
            approved=False, score=0.0, threshold=0.0, model_version=VERIFY_UNAVAILABLE, latency_ms=0.0, degraded=True
        )

    # -- decision upload ---------------------------------------------------

    def _queue(
        self,
        model: _Model,
        query: str,
        answer: str,
        score: float,
        rule: str,
        approved: bool,
        audited: bool,
        audit_rate: float,
    ) -> str:
        key = bytes.fromhex(model.manifest["pair_key"])
        digest = hmac.new(key, f"{query}\x00{answer}".encode(), hashlib.sha256).hexdigest()
        token = uuid.uuid4().hex
        decision = {
            "score": score,
            "decision_rule": rule,
            "verifier_approved": approved,
            "audited": audited,
            "audit_probability": audit_rate,
            "pair_digest": digest,
        }
        with self._state_lock:
            self._pending.append((token, model.version_id, decision))
            if len(self._pending) > MAX_PENDING_DECISIONS:
                dropped = len(self._pending) - MAX_PENDING_DECISIONS
                del self._pending[:dropped]
                _log.warning("cacheverifier: dropped %d un-uploaded decisions (service unreachable?)", dropped)
            full = len(self._pending) >= UPLOAD_BATCH
        if full:
            self._wake.set()
        return token

    def flush(self) -> None:
        """Upload pending decisions now."""
        while True:
            with self._state_lock:
                if not self._pending:
                    return
                version = self._pending[0][1]
                batch = [p for p in self._pending[:UPLOAD_BATCH] if p[1] == version]
            resp = self._hosted._post(
                "/v1/verify/decisions",
                json={"model_version_id": version, "decisions": [d for _, _, d in batch]},
            )
            with self._state_lock:
                for (token, _, _), vid in zip(batch, resp["verification_ids"]):
                    self._ids[token] = int(vid)
                while len(self._ids) > MAX_REMEMBERED_IDS:
                    self._ids.popitem(last=False)
                sent = {token for token, _, _ in batch}
                self._pending = [p for p in self._pending if p[0] not in sent]

    def verification_id(self, result: VerifyResult) -> int | None:
        """The service's `verification_id` for a local result (uploads it
        first if it's still pending). None when auditing was off."""
        if result.verification_id is not None or result.local_decision is None:
            return result.verification_id
        with self._state_lock:
            vid = self._ids.get(result.local_decision)
        if vid is None:
            self.flush()
            with self._state_lock:
                vid = self._ids.get(result.local_decision)
        return vid

    # -- feedback ----------------------------------------------------------

    def feedback(
        self,
        query: str,
        candidate_answer: str,
        was_correct: bool,
        *,
        result: VerifyResult | None = None,
        similarity_score: float | None = None,
        cached_query: str | None = None,
        stale: bool = False,
        idempotency_key: str | None = None,
    ) -> int:
        """`CacheVerifier.feedback`, linked to `result` when given (always
        pass it for an `audit` result: that's what makes the audit count)."""
        verification_id = self.verification_id(result) if result is not None else None
        return self._hosted.feedback(
            query,
            candidate_answer,
            was_correct,
            similarity_score=similarity_score,
            cached_query=cached_query,
            verification_id=verification_id,
            stale=stale,
            idempotency_key=idempotency_key,
        )

    @property
    def hosted(self) -> CacheVerifier:
        """The hosted client this uses, for fine-tuning, monitoring and usage calls."""
        return self._hosted

    def __repr__(self) -> str:
        return f"LocalVerifier(model_version={self.model_version!r}, cache_dir={str(self._cache_dir)!r})"


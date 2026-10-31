"""Offline tests for `LocalVerifier`: the service is an httpx.MockTransport
and the model a word-overlap stand-in, so nothing here downloads a real
checkpoint or needs torch."""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest

from cacheverifier import _inference
from cacheverifier.local import LocalVerifier, ModelCheckFailed, _Model

PAIR_KEY = "ab" * 32
FILES = {"config.json": b'{"fake": true}', "model.safetensors": bytes(range(256)) * 400}


class FakeTokenizer:
    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.words: dict[int, str] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = []
        for w in text.split():
            if w not in self.vocab:
                self.vocab[w] = len(self.vocab)
                self.words[self.vocab[w]] = w
            ids.append(self.vocab[w])
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(self.words[i] for i in ids)


class FakeEncoder:
    """Scores a pair by word overlap, minus a little per [qdiff] word."""

    tokenizer = FakeTokenizer()

    def predict(self, pairs: list[tuple[str, str]], show_progress_bar: bool = False) -> list[float]:
        out = []
        for a, b in pairs:
            wa, wb = set(a.lower().split()), set(b.lower().split())
            out.append(4.0 * len(wa & wb) / max(1, len(wa | wb)) - 1.0 - 0.1 * a.count("[qdiff]"))
        return out


REF_PAIRS = [
    {"query": "when will my parcel arrive", "candidate_answer": "your parcel will arrive in 3 days"},
    {"query": "where is the depot", "candidate_answer": "your parcel will arrive in 3 days"},
    {
        "query": "how long does my parcel take",
        "candidate_answer": "your parcel will arrive in 3 days",
        "cached_query": "when will my parcel arrive",
        "similarity_score": 0.86,
        "runner_up_similarity_score": 0.7,
    },
]


def _manifest(version: int = 50, threshold: float = 0.0, audit_rate: float = 0.0, tamper: bool = False) -> dict:
    enc = FakeEncoder()
    refs = []
    for r in REF_PAIRS:
        text_a = _inference.compose_query_input(r["query"], r.get("cached_query"))
        score = enc.predict([(text_a, r["candidate_answer"])])[0] + (0.5 if tamper else 0.0)
        ok, rule = _inference.decide(score, threshold, None, None, r.get("similarity_score"), None)
        refs.append({**r, "score": score, "approved": ok, "decision_rule": rule})
    return {
        "model_version_id": version,
        "model_version": f"v{version}",
        "format": 1,
        "files": [{"name": n, "size": len(b), "sha256": hashlib.sha256(b).hexdigest()} for n, b in FILES.items()],
        "score_kind": "logit",
        "max_sequence_length": 128,
        "uses_cached_query": True,
        "threshold": threshold,
        "joint_rule": None,
        "joint_rule_gap": None,
        "reference": refs,
        "audit_rate": audit_rate,
        "pair_key": PAIR_KEY,
        "aci_enabled": False,
    }


class FakeService:
    def __init__(self, manifest: dict | None, status: int = 200) -> None:
        self.manifest, self.status = manifest, status
        self.calls: list[httpx.Request] = []
        self.decision_bodies: list[dict] = []
        self.feedback_bodies: list[dict] = []
        self.next_id = 1000

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        path = request.url.path
        if path == "/v1/models/active":
            if self.manifest is None:
                return httpx.Response(self.status, json={"detail": "no model"})
            return httpx.Response(200, json=self.manifest)
        if path.startswith("/v1/models/") and "/files/" in path:
            data = FILES[path.rsplit("/", 1)[1]]
            rng = request.headers.get("Range")
            if rng:
                start = int(rng.split("=")[1].rstrip("-"))
                return httpx.Response(206, content=data[start:])
            return httpx.Response(200, content=data)
        if path == "/v1/verify/decisions":
            body = json.loads(request.content)
            self.decision_bodies.append(body)
            ids = list(range(self.next_id, self.next_id + len(body["decisions"])))
            self.next_id += len(ids)
            return httpx.Response(200, json={"verification_ids": ids})
        if path == "/v1/feedback":
            self.feedback_bodies.append(json.loads(request.content))
            return httpx.Response(200, json={"id": 1})
        if path == "/v1/verify":
            return httpx.Response(
                200, json={"approved": True, "score": 9.0, "threshold": 0.0, "model_version": "v50", "latency_ms": 20}
            )
        return httpx.Response(404, json={"detail": path})


@pytest.fixture(autouse=True)
def _fake_model(monkeypatch):
    def load(manifest: dict[str, Any], path: Path) -> _Model:
        for f in manifest["files"]:  # the real loader reads these; make sure they're there
            assert (path / f["name"]).read_bytes() == FILES[f["name"]]
        return _Model(manifest=dict(manifest), encoder=FakeEncoder(), joint_rule=None, joint_rule_gap=None,
                      lock=threading.Lock())

    monkeypatch.setattr(LocalVerifier, "_load", staticmethod(load))


def _local(tmp_path: Path, service: FakeService, **kwargs: Any) -> LocalVerifier:
    return LocalVerifier("cv_test", cache_dir=tmp_path, transport=httpx.MockTransport(service), background=False,
                         **kwargs)


def _paths(service: FakeService) -> list[str]:
    return [r.url.path for r in service.calls]


def test_downloads_checks_and_decides_locally(tmp_path):
    service = FakeService(_manifest())
    cv = _local(tmp_path, service, wait=True)
    assert cv.model_version == "v50"
    r = cv.verify("when will my parcel arrive", "your parcel will arrive in 3 days")
    assert r.approved and r.model_version == "v50" and r.decision_rule == "threshold" and not r.degraded
    assert cv.verify("where is the depot", "your parcel will arrive in 3 days").approved is False
    assert "/v1/verify" not in _paths(service)
    # A second refresh with the same live version downloads nothing.
    n = len(service.calls)
    assert cv.refresh() is False
    assert _paths(service)[n:] == ["/v1/models/active"]


def test_an_interrupted_download_resumes(tmp_path):
    service = FakeService(_manifest())
    cv = _local(tmp_path, service)
    target = cv._version_dir(service.manifest)
    target.mkdir(parents=True)
    (target / "model.safetensors.part").write_bytes(FILES["model.safetensors"][:5000])
    cv.refresh()
    ranges = [r.headers.get("Range") for r in service.calls if r.url.path.endswith("model.safetensors")]
    assert ranges == ["bytes=5000-"]
    assert (target / "model.safetensors").read_bytes() == FILES["model.safetensors"]


def test_a_model_that_misses_the_reference_verdicts_is_not_used(tmp_path):
    service = FakeService(_manifest(tamper=True))
    cv = _local(tmp_path, service)
    with pytest.raises(ModelCheckFailed):
        cv.refresh()
    assert cv.model_version is None
    r = cv.verify("q", "a")  # falls back to the hosted API
    assert r.score == 9.0 and "/v1/verify" in _paths(service)


def test_without_a_plan_or_fine_tune_it_stays_on_the_fallback(tmp_path):
    service = FakeService(None, status=402)
    cv = _local(tmp_path, service, fallback="reject")
    assert cv.refresh() is False
    r = cv.verify("q", "a")
    assert r.degraded and not r.approved and "/v1/verify" not in _paths(service)


def test_audits_upload_without_texts_and_feedback_links(tmp_path, monkeypatch):
    service = FakeService(_manifest(audit_rate=1.0))
    cv = _local(tmp_path, service, wait=True)
    q, a = "when will my parcel arrive", "your parcel will arrive in 3 days"
    r = cv.verify(q, a)
    assert r.audit and not r.approved and r.verification_id is None and r.local_decision

    cv.feedback(q, a, True, result=r)  # uploads the pending decision first, then links
    [body] = service.decision_bodies
    assert body["model_version_id"] == 50
    [d] = body["decisions"]
    assert d["audited"] and d["verifier_approved"] and d["audit_probability"] == 1.0
    expected = hmac.new(bytes.fromhex(PAIR_KEY), f"{q}\x00{a}".encode(), hashlib.sha256).hexdigest()
    assert d["pair_digest"] == expected
    raw = service.calls[[c.url.path for c in service.calls].index("/v1/verify/decisions")].content.decode()
    assert q not in raw and a not in raw
    assert service.feedback_bodies[-1]["verification_id"] == 1000


def test_no_uploads_while_auditing_is_off(tmp_path):
    service = FakeService(_manifest(audit_rate=0.0))
    cv = _local(tmp_path, service, wait=True)
    r = cv.verify("when will my parcel arrive", "your parcel will arrive in 3 days")
    assert r.local_decision is None
    cv.flush()
    cv.feedback("when will my parcel arrive", "x", False, result=r)
    assert service.decision_bodies == [] and "verification_id" not in service.feedback_bodies[-1]


def test_a_newer_live_version_replaces_the_old_and_old_downloads_are_pruned(tmp_path):
    service = FakeService(_manifest(version=50))
    cv = _local(tmp_path, service, wait=True)
    for version in (51, 52):
        service.manifest = _manifest(version=version, threshold=-0.5)
        assert cv.refresh() is True
        assert cv.model_version == f"v{version}"
    kept = sorted(d.name.split("-")[0] for d in tmp_path.iterdir())
    assert kept == ["v51", "v52"]


def test_decide_follows_the_service_precedence():
    rule = _inference.JointRule(1.0, 1.0, 0.0, 0.5, 0.1, 0.0, 1.0, 0.5)
    gap = _inference.JointRule(1.0, 1.0, 0.0, 0.5, 0.1, 0.0, 1.0, 0.5, coef_gap=5.0, uses_gap=True)
    assert _inference.decide(0.2, 0.0, rule, gap, 0.9, 0.85)[1] == "joint_gap"
    assert _inference.decide(0.2, 0.0, rule, gap, 0.9, None)[1] == "joint"
    assert _inference.decide(0.2, 0.0, rule, gap, None, None) == (True, "threshold")
    floored = _inference.JointRule(1.0, 1.0, 0.0, 0.5, 0.1, 0.0, 1.0, 0.5, verifier_floor=1.0)
    assert floored.approves(0.5, 0.99) is False


def test_truncation_gives_a_long_query_the_room_a_short_answer_leaves():
    tok = FakeTokenizer()
    query = " ".join(f"q{i}" for i in range(200))
    q, a = _inference.smart_truncate_pair(query, "short answer", tok)
    assert a == "short answer"
    assert len(q.split()) == _inference.CONTENT_TOKEN_BUDGET - 2

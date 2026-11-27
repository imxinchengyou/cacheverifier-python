"""RejectedHitReporter: which client call it makes, and the pending bound."""

from cacheverifier import LocalVerifier, VerifyResult
from cacheverifier.integrations import _reporting
from cacheverifier.integrations._reporting import RejectedHitReporter


def _result(**kw) -> VerifyResult:
    base = {"approved": False, "score": -1.0, "threshold": 0.0, "model_version": "v9", "latency_ms": 1.0}
    base.update(kw)
    return VerifyResult(**base)


class _FakeLocal(LocalVerifier):
    def __init__(self) -> None:  # no model download
        self.calls: list = []

    def feedback(self, query, candidate_answer, was_correct, **kwargs):
        self.calls.append((query, candidate_answer, was_correct, kwargs))
        return 1


def test_local_verifier_is_reported_by_result_not_by_id():
    client = _FakeLocal()
    r = _result(local_decision="tok-1")
    reporter = RejectedHitReporter(client, lambda q, c, f: True, background=False)
    reporter.remember("k", "q", "cached", r)
    reporter.fresh_answer("k", "fresh")
    assert client.calls == [("q", "cached", True, {"result": r, "label_method": "judge"})]


def test_degraded_and_approved_results_are_not_remembered():
    client = _FakeLocal()
    reporter = RejectedHitReporter(client, lambda q, c, f: True, background=False)
    reporter.remember("a", "q", "c", _result(degraded=True, model_version="verify_unavailable"))
    reporter.remember("b", "q", "c", _result(approved=True))
    reporter.fresh_answer("a", "f")
    reporter.fresh_answer("b", "f")
    assert client.calls == []


def test_pending_hits_are_bounded(monkeypatch):
    monkeypatch.setattr(_reporting, "MAX_PENDING", 2)
    client = _FakeLocal()
    reporter = RejectedHitReporter(client, lambda q, c, f: True, background=False)
    for key in ("a", "b", "c"):
        reporter.remember(key, key, "c", _result())
    for key in ("a", "b", "c"):
        reporter.fresh_answer(key, "f")
    assert [call[0] for call in client.calls] == ["b", "c"]  # oldest dropped

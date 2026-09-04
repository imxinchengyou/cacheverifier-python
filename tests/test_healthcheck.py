"""CLI dispatch + Health Check runner.

The dispatch/parsing tests run everywhere (base install). The end-to-end
training test needs the `healthcheck` extra and is skipped when torch is
absent -- CI's base matrix skips it; the dedicated `healthcheck` job runs
it.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass

import pytest

from cacheverifier.__main__ import main
from cacheverifier._healthcheck import cli

HAS_TORCH = importlib.util.find_spec("torch") is not None


@dataclass(frozen=True)
class _DummyExample:
    query: str
    candidate_answer: str
    was_correct: bool


class TestDispatch:
    def test_version(self, capsys):
        with pytest.raises(SystemExit) as e:
            main(["--version"])
        assert e.value.code == 0
        from cacheverifier import __version__

        assert f"cacheverifier {__version__}" in capsys.readouterr().out

    def test_no_command_prints_help_and_returns_1(self, capsys):
        assert main([]) == 1
        assert "healthcheck" in capsys.readouterr().out

    def test_healthcheck_help(self):
        with pytest.raises(SystemExit) as e:
            main(["healthcheck", "--help"])
        assert e.value.code == 0

    @pytest.mark.skipif(HAS_TORCH, reason="torch present -- the missing-extra path can't be exercised")
    def test_healthcheck_without_extra_points_at_pip_install(self, capsys, tmp_path):
        f = tmp_path / "t.jsonl"
        f.write_text("")
        assert main(["healthcheck", str(f)]) == 2
        assert 'pip install "cacheverifier[healthcheck]"' in capsys.readouterr().err


class TestParsing:
    def test_load_rows_accepts_json_array_and_jsonl(self, tmp_path):
        rows = [{"a": 1}, {"a": 2}]
        arr = tmp_path / "a.json"
        arr.write_text(json.dumps(rows))
        jsonl = tmp_path / "a.jsonl"
        jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        assert cli._load_rows(arr) == rows
        assert cli._load_rows(jsonl) == rows

    def test_to_examples_skips_stale_and_counts_them(self):
        rows = [
            {"query": "q1", "candidate_answer": "a1", "was_correct": True},
            {"query": "q2", "candidate_answer": "a2", "was_correct": False, "stale": True},
        ]
        examples, n_stale = cli._to_examples(rows, _DummyExample)
        assert len(examples) == 1
        assert n_stale == 1
        assert examples[0] == _DummyExample("q1", "a1", True)

    def test_to_examples_rejects_missing_keys(self):
        with pytest.raises(ValueError, match="missing key"):
            cli._to_examples([{"query": "q"}], _DummyExample)


@pytest.mark.skipif(not HAS_TORCH, reason="needs the healthcheck extra (torch, sentence-transformers)")
def test_run_healthcheck_end_to_end(tmp_path):
    from cacheverifier._healthcheck._finetune import GrayZoneExample, HealthCheckResult, run_healthcheck

    groups = [
        ("how do I cancel my subscription", "Settings > Billing > Cancel.", "Settings > Billing > Pause a month."),
        ("how do I get a refund", "Orders > Return > Request refund.", "Orders > Return > Exchange size."),
        ("how do I reset my password", "Account > Security > Reset password.", "Account > Security > Enable 2FA."),
    ]
    examples = []
    for i in range(36):
        q, good, bad = groups[i % len(groups)]
        correct = i % 2 == 0
        examples.append(GrayZoneExample(q, good if correct else bad, correct))

    result = run_healthcheck(examples, str(tmp_path / "model"), epochs=1)
    assert isinstance(result, HealthCheckResult)
    assert result.n_train + result.n_calibrate + result.n_test == 36
    assert 0.0 <= result.auc_tuned <= 1.0
    assert (tmp_path / "model").is_dir()

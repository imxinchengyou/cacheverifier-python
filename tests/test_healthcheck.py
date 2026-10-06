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


@pytest.mark.skipif(not HAS_TORCH, reason="needs the healthcheck extra (torch)")
def test_run_healthcheck_warms_up_over_a_tenth_of_its_steps(tmp_path, monkeypatch):
    # CrossEncoder.fit defaults to warmup_steps=10000, which kept a
    # few-hundred-step run from ever reaching its learning rate.
    from sentence_transformers import CrossEncoder

    from cacheverifier._healthcheck import _finetune
    from cacheverifier._healthcheck._finetune import GrayZoneExample, run_healthcheck

    seen: dict = {}
    real_fit = CrossEncoder.fit

    def spy_fit(self, *args, **kwargs):
        seen.update(kwargs, steps_per_epoch=len(kwargs["train_dataloader"]))
        return real_fit(self, *args, **kwargs)

    monkeypatch.setattr(CrossEncoder, "fit", spy_fit)
    examples = [GrayZoneExample(f"question {i}", f"answer {i % 3}", i % 2 == 0) for i in range(40)]
    run_healthcheck(examples, str(tmp_path / "model"))

    assert seen["epochs"] == _finetune.TRAIN_EPOCHS == 3
    expected = max(1, int(_finetune.TRAIN_WARMUP_FRACTION * seen["steps_per_epoch"] * seen["epochs"]))
    assert seen["warmup_steps"] == expected


class TestBaseModel:
    def test_aliases_match_the_hosted_service(self):
        from cacheverifier._healthcheck._base_models import BASE_MODEL_CHOICES, resolve_base_model

        assert set(BASE_MODEL_CHOICES) == {"ms_marco", "nli", "multilingual", "multilingual_small"}
        assert resolve_base_model("ms_marco") == "cross-encoder/ms-marco-MiniLM-L6-v2"
        assert resolve_base_model("multilingual") == "BAAI/bge-reranker-base"

    def test_hf_id_and_local_dir_pass_through(self, tmp_path):
        from cacheverifier._healthcheck._base_models import resolve_base_model

        assert resolve_base_model("someorg/some-reranker") == "someorg/some-reranker"
        assert resolve_base_model(str(tmp_path)) == str(tmp_path)

    @pytest.mark.parametrize("bad", ["multilingal", "/no/such/model/dir", ""])
    def test_typos_and_missing_paths_are_rejected(self, bad):
        from cacheverifier._healthcheck._base_models import resolve_base_model

        with pytest.raises(ValueError, match="unknown base model"):
            resolve_base_model(bad)

    def test_cjk_ratio(self):
        from cacheverifier._healthcheck._base_models import cjk_ratio

        assert cjk_ratio(["how do I cancel"]) == 0.0
        assert cjk_ratio(["怎么取消订阅"]) == 1.0
        assert cjk_ratio(["怎么取消 subscription"]) == 4 / 16
        assert cjk_ratio([]) == 0.0

    def test_help_lists_base_model_choices(self, capsys):
        with pytest.raises(SystemExit):
            main(["healthcheck", "--help"])
        out = capsys.readouterr().out
        assert "--base-model" in out and "multilingual_small" in out

    @pytest.mark.skipif(not HAS_TORCH, reason="run() imports the healthcheck extra first")
    def test_unknown_base_model_is_an_input_error(self, capsys, tmp_path):
        f = tmp_path / "t.jsonl"
        f.write_text(json.dumps({"query": "q", "candidate_answer": "a", "was_correct": True}) + "\n")
        assert main(["healthcheck", str(f), "--base-model", "multilingal"]) == 2
        assert "unknown base model" in capsys.readouterr().err

    @pytest.mark.skipif(not HAS_TORCH, reason="run() imports the healthcheck extra first")
    def test_chinese_data_on_english_model_warns(self, capsys, tmp_path):
        # Below MIN_TRAIN_EXAMPLES, so run() stops before any download/training.
        f = tmp_path / "t.jsonl"
        f.write_text(
            "\n".join(
                json.dumps({"query": "怎么取消自动续费", "candidate_answer": "进入设置取消", "was_correct": True}, ensure_ascii=False)
                for _ in range(3)
            )
        )
        assert main(["healthcheck", str(f)]) == 2
        out = capsys.readouterr().out
        assert "English-only" in out and "--base-model multilingual" in out

        assert main(["healthcheck", str(f), "--base-model", "multilingual_small"]) == 2
        assert "English-only" not in capsys.readouterr().out


@pytest.mark.skipif(not HAS_TORCH, reason="needs the healthcheck extra (torch, sentence-transformers)")
def test_run_healthcheck_nli_base_reinitializes_to_single_logit(tmp_path):
    """The NLI base ships a 3-class head: the stock baseline is scored
    P(entailment) - P(contradiction), and fine-tuning reinitializes it to the
    same 1-logit head as every other base (matching the hosted service)."""
    from sentence_transformers import CrossEncoder

    from cacheverifier._healthcheck._base_models import BASE_MODEL_CHOICES
    from cacheverifier._healthcheck._finetune import GrayZoneExample, run_healthcheck

    examples = [
        GrayZoneExample("how do I cancel my plan", "Settings > Billing > Cancel." if i % 2 == 0 else "Settings > Pause.", i % 2 == 0)
        for i in range(24)
    ]
    result = run_healthcheck(examples, str(tmp_path / "model"), base_model=BASE_MODEL_CHOICES["nli"], epochs=1)
    assert 0.0 <= result.auc_baseline <= 1.0
    assert 0.0 <= result.auc_tuned <= 1.0
    assert CrossEncoder(str(tmp_path / "model")).config.num_labels == 1

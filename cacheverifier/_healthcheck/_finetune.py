"""Local fine-tune + held-out AUC evaluation for `cacheverifier healthcheck`.

This is a trimmed, self-contained port of the hosted service's
`verifier_core.finetune` / `verifier_core.cross_encoder` (the code behind
`POST /v1/finetune/dry-run`). It keeps the parts a Health Check needs --
the chronological train/calibrate/test split, held-out AUC for the stock
vs. fine-tuned model, a Youden's-J operating point, and the label-noise
and ceiling diagnostics -- and drops everything specific to running a
production tenant (Conformal Risk Control certification, cost-ratio
threshold grids, drift monitoring, operating-point-jump detection).

Nothing here is imported unless the `healthcheck` extra is installed and
the subcommand is actually run -- see `cacheverifier.__main__`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

# All four imports above are from the `healthcheck` extra. This module is
# only ever imported from inside `cacheverifier._healthcheck.cli.run()`,
# which catches the ImportError and points the user at
# `pip install "cacheverifier[healthcheck]"` -- so a plain
# `import cacheverifier` / `cacheverifier --help` never pays this cost.

DEFAULT_BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
"""The stock off-the-shelf verifier every tenant starts on -- the same base
model the hosted service fine-tunes from."""

MAX_SEQUENCE_LENGTH = 128
"""Token cap applied identically at train and score time. Matches the
hosted service; BERT-style attention memory is O(n^2) in sequence length,
and this keeps a CPU run well-behaved."""

_CONTENT_TOKEN_BUDGET = MAX_SEQUENCE_LENGTH - 3  # [CLS] + 2x [SEP]
_MIN_QUERY_TOKENS = 32

TRAIN_BATCH_SIZE = 4
MIN_TRAIN_EXAMPLES = 20
"""Hard floor -- below this there aren't enough rows for one meaningful
epoch plus a held-out split."""

COLD_START_WARNING_THRESHOLD = 1000
"""The hosted service's research (Research page, Q2) found fine-tuning only
reliably positive across every dataset above roughly this many rows. Below
it a result is real but noisy."""

_MIN_TEST_FOR_THRESHOLD = 10
_MIN_TEST_FOR_CEILING_DIAGNOSIS = 50
_CEILING_TAIL_MARGIN = 0.15
_CEILING_AUC_CAP = 0.75

NOISE_WARNING_DISAGREEMENT_RATE = 0.20
"""`train_label_disagreement_rate` above this: worth auditing how the
correctness signal is produced."""
NOISE_HOLD_BACK_DISAGREEMENT_RATE = 0.30
"""Above this the hosted service holds a new model back for review."""


@dataclass(frozen=True)
class GrayZoneExample:
    query: str
    candidate_answer: str
    was_correct: bool


@dataclass(frozen=True)
class HealthCheckResult:
    n_train: int
    n_calibrate: int
    n_test: int
    train_positive_rate: float
    test_positive_rate: float
    auc_baseline: float
    auc_tuned: float
    train_time_seconds: float
    threshold: float | None
    threshold_hit_rate: float | None
    threshold_error_rate: float | None
    train_label_disagreement_rate: float | None
    ceiling_status: str | None
    model_path: str

    @property
    def auc_delta(self) -> float:
        return self.auc_tuned - self.auc_baseline


def _head_tail_truncate(ids: list[int], budget: int, head_ratio: float = 0.6) -> list[int]:
    if len(ids) <= budget:
        return ids
    head_n = int(budget * head_ratio)
    tail_n = budget - head_n
    return ids[:head_n] + (ids[-tail_n:] if tail_n > 0 else [])


def smart_truncate_pair(query: str, candidate_answer: str, tokenizer: Any) -> tuple[str, str]:
    """Tokenizer-aware head+tail truncation, applied identically at train
    and score time (a model trained on one truncation policy and scored
    with another sees a different token window than it was optimized
    against). A no-op whenever the pair already fits."""
    query_ids = tokenizer.encode(query, add_special_tokens=False)
    answer_ids = tokenizer.encode(candidate_answer, add_special_tokens=False)
    if len(query_ids) + len(answer_ids) <= _CONTENT_TOKEN_BUDGET:
        return query, candidate_answer

    query_budget = min(len(query_ids), max(_MIN_QUERY_TOKENS, _CONTENT_TOKEN_BUDGET // 2))
    answer_budget = _CONTENT_TOKEN_BUDGET - query_budget
    query_ids = _head_tail_truncate(query_ids, query_budget)
    answer_ids = _head_tail_truncate(answer_ids, answer_budget)
    return (
        tokenizer.decode(query_ids, skip_special_tokens=True),
        tokenizer.decode(answer_ids, skip_special_tokens=True),
    )


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC, no sklearn dependency. NaN if either class is absent."""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(scores))
    pos_ranks = ranks[labels == 1]
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((pos_ranks.sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def select_threshold(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Youden's J: the threshold maximizing (TPR - FPR). None when the
    held-out split is too small or single-class to calibrate from -- the
    caller decides the fallback."""
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0 or (n_pos + n_neg) < _MIN_TEST_FOR_THRESHOLD:
        return None

    order = np.argsort(-scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    youden_j = tp / n_pos - fp / n_neg
    return float(sorted_scores[int(np.argmax(youden_j))])


def _hit_and_error_rate(scores: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float, float | None]:
    approved = scores >= threshold
    n_approved = int(approved.sum())
    hit_rate = n_approved / len(scores) if len(scores) else 0.0
    if n_approved == 0:
        return hit_rate, None
    return hit_rate, float(np.mean(labels[approved] == 0))


def _diagnose_ceiling(scores: np.ndarray, labels: np.ndarray) -> str | None:
    """"insufficient_data" | "possible_ceiling" | "still_improvable" -- is
    the fine-tuned verifier plausibly undertrained (more data would help)
    or capped by the domain's own ambiguity? Precision in the top score
    decile vs. the base rate, same shape of check as the hosted service."""
    n = len(labels)
    if n < _MIN_TEST_FOR_CEILING_DIAGNOSIS:
        return "insufficient_data"
    base_rate = float(labels.mean())
    top_decile_n = max(1, n // 10)
    top_decile_idx = np.argsort(-scores)[:top_decile_n]
    precision_at_top_decile = float(labels[top_decile_idx].mean())
    auc = roc_auc(scores, labels)
    if precision_at_top_decile - base_rate >= _CEILING_TAIL_MARGIN and auc < _CEILING_AUC_CAP:
        return "possible_ceiling"
    return "still_improvable"


def run_healthcheck(
    examples: list[GrayZoneExample],
    output_dir: str,
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: int = 1,
) -> HealthCheckResult:
    """Fine-tune `base_model` on a chronological prefix of `examples` and
    measure held-out AUC for the stock vs. fine-tuned model.

    `examples` must be in arrival (stream) order -- the train/calibrate/test
    split is positional, not shuffled, matching the hosted service so the
    numbers are comparable.
    """
    if len(examples) < MIN_TRAIN_EXAMPLES:
        raise ValueError(f"need at least {MIN_TRAIN_EXAMPLES} examples, got {len(examples)}")

    split = max(1, int(len(examples) * 0.8))
    train_rows, holdout_rows = examples[:split], examples[split:]
    if not holdout_rows:
        train_rows, holdout_rows = examples[:-1], examples[-1:]
    calib_split = max(1, len(holdout_rows) // 2)
    calib_rows, test_rows = holdout_rows[:calib_split], holdout_rows[calib_split:]
    if not test_rows:
        calib_rows, test_rows = holdout_rows[:-1], holdout_rows[-1:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_pairs = [(e.query, e.candidate_answer) for e in test_rows]
    test_labels = np.array([1 if e.was_correct else 0 for e in test_rows])
    calib_pairs = [(e.query, e.candidate_answer) for e in calib_rows]
    calib_labels = np.array([1 if e.was_correct else 0 for e in calib_rows])
    train_positive_rate = float(np.mean([1 if e.was_correct else 0 for e in train_rows]))
    test_positive_rate = float(test_labels.mean()) if len(test_labels) else 0.0

    baseline = CrossEncoder(base_model, device=device, max_length=MAX_SEQUENCE_LENGTH)
    test_pairs_scored = [smart_truncate_pair(q, a, baseline.tokenizer) for q, a in test_pairs]
    baseline_scores = np.array(baseline.predict(test_pairs_scored, batch_size=32, show_progress_bar=False))
    auc_baseline = roc_auc(baseline_scores, test_labels)
    del baseline

    tuned = CrossEncoder(base_model, device=device, max_length=MAX_SEQUENCE_LENGTH)
    train_examples = [
        InputExample(
            texts=list(smart_truncate_pair(e.query, e.candidate_answer, tuned.tokenizer)),
            label=1.0 if e.was_correct else 0.0,
        )
        for e in train_rows
    ]
    # A list[InputExample] is a valid PyTorch map-style dataset at runtime;
    # sentence-transformers' pre-4.0 training API is built around exactly
    # this, but the two libraries' stubs don't compose (InputExample isn't a
    # typed Dataset). cast rather than `# type: ignore` so this stays clean
    # whether or not torch's stubs are installed (base CI has no torch).
    loader: DataLoader[Any] = DataLoader(cast(Any, train_examples), shuffle=True, batch_size=TRAIN_BATCH_SIZE)
    t0 = time.time()
    tuned.fit(train_dataloader=loader, epochs=epochs, show_progress_bar=False)
    train_time_seconds = time.time() - t0
    tuned.save(output_dir)

    calib_scored = [smart_truncate_pair(q, a, tuned.tokenizer) for q, a in calib_pairs]
    calib_scores = np.array(tuned.predict(calib_scored, batch_size=32, show_progress_bar=False))
    threshold = select_threshold(calib_scores, calib_labels)

    tuned_scores = np.array(tuned.predict(test_pairs_scored, batch_size=32, show_progress_bar=False))
    auc_tuned = roc_auc(tuned_scores, test_labels)
    ceiling_status = _diagnose_ceiling(tuned_scores, test_labels)

    threshold_hit_rate: float | None = None
    threshold_error_rate: float | None = None
    train_label_disagreement_rate: float | None = None
    if threshold is not None:
        threshold_hit_rate, threshold_error_rate = _hit_and_error_rate(tuned_scores, test_labels, threshold)
        train_pairs_scored = [smart_truncate_pair(e.query, e.candidate_answer, tuned.tokenizer) for e in train_rows]
        train_scores = np.array(tuned.predict(train_pairs_scored, batch_size=32, show_progress_bar=False))
        train_labels = np.array([1 if e.was_correct else 0 for e in train_rows])
        train_pred = (train_scores >= threshold).astype(int)
        train_label_disagreement_rate = float(np.mean(train_pred != train_labels))

    return HealthCheckResult(
        n_train=len(train_rows),
        n_calibrate=len(calib_rows),
        n_test=len(test_rows),
        train_positive_rate=train_positive_rate,
        test_positive_rate=test_positive_rate,
        auc_baseline=auc_baseline,
        auc_tuned=auc_tuned,
        train_time_seconds=train_time_seconds,
        threshold=threshold,
        threshold_hit_rate=threshold_hit_rate,
        threshold_error_rate=threshold_error_rate,
        train_label_disagreement_rate=train_label_disagreement_rate,
        ceiling_status=ceiling_status,
        model_path=output_dir,
    )

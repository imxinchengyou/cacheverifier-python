"""Argument parsing, IO, and the printed report for `cacheverifier healthcheck`.

The heavy work (torch, sentence-transformers) lives in `._finetune` and is
imported only inside `run()`, so `cacheverifier --help` stays light and
works without the `healthcheck` extra installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_MISSING_EXTRA_HINT = (
    "the healthcheck extra is not installed -- run:\n"
    '    pip install "cacheverifier[healthcheck]"\n'
    "(this pulls in torch + sentence-transformers; it is deliberately not a\n"
    "dependency of the base client)"
)


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "healthcheck",
        help="run the Health Check evaluation locally (no data leaves your machine)",
        description=(
            "Fine-tune the stock verifier on a chronological prefix of your own "
            "gray-zone cache hits and measure held-out AUC before and after -- the "
            "same evaluation as POST /v1/finetune/dry-run, but entirely offline. "
            "Your queries and answers never leave this host; only the optional "
            "--emit-summary file (floats and counts, no text) can."
        ),
    )
    p.add_argument(
        "input",
        type=Path,
        help="JSON array or JSONL of {query, candidate_answer, was_correct} rows, in arrival order",
    )
    p.add_argument(
        "--emit-summary",
        type=Path,
        default=None,
        metavar="PATH",
        help="also write an aggregate-only JSON summary (no query/answer text) to PATH",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="fine-tuning epochs (default: 1, matching the hosted service)",
    )
    p.add_argument(
        "--keep-model",
        action="store_true",
        help="keep the locally fine-tuned model directory instead of deleting it",
    )
    p.set_defaults(func=run)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text()
    if text.lstrip().startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list):
        # ValueError, not TypeError: this is bad input data, not a caller
        # bug -- run() catches it alongside JSONDecodeError and prints it.
        raise ValueError("input must be a JSON array or JSONL of objects")  # noqa: TRY004
    return rows


def _to_examples(rows: list[dict[str, Any]], example_cls: Any) -> tuple[list[Any], int]:
    examples = []
    n_stale = 0
    for i, row in enumerate(rows):
        missing = {"query", "candidate_answer", "was_correct"} - row.keys()
        if missing:
            raise ValueError(f"row {i}: missing key(s) {sorted(missing)}")
        if row.get("stale"):
            n_stale += 1
            continue
        examples.append(
            example_cls(
                query=str(row["query"]),
                candidate_answer=str(row["candidate_answer"]),
                was_correct=bool(row["was_correct"]),
            )
        )
    return examples, n_stale


def _fmt(x: float | None, *, pct: bool = False) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.1f}%" if pct else f"{x:.4f}"


def _verdict(delta: float) -> str:
    if delta >= 0.02:
        return "IMPROVED   -- fine-tuning on your own data helps this traffic"
    if delta <= -0.02:
        return "WORSE      -- fine-tuning hurt; your signal may be too noisy or too sparse"
    return "NO CHANGE   -- fine-tuning neither helped nor hurt measurably"


def run(args: argparse.Namespace) -> int:
    try:
        from cacheverifier._healthcheck import _finetune
    except ImportError as e:  # torch / sentence-transformers / numpy absent
        print(f"error: {_MISSING_EXTRA_HINT}\n\n(import failed: {e})", file=sys.stderr)
        return 2

    input_path: Path = args.input
    if not input_path.exists():
        print(f"error: input file not found: {input_path}", file=sys.stderr)
        return 2

    try:
        rows = _load_rows(input_path)
        examples, n_stale = _to_examples(rows, _finetune.GrayZoneExample)
    except (ValueError, json.JSONDecodeError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    n = len(examples)
    print(f"\nCacheVerifier local Health Check  --  {input_path}")
    print("=" * 66)
    print(f"rows read:            {len(rows)}")
    print(f"usable (non-stale):   {n}")
    if n_stale:
        print(f"stale, excluded:      {n_stale}")

    if n < _finetune.MIN_TRAIN_EXAMPLES:
        print(f"\nNeed at least {_finetune.MIN_TRAIN_EXAMPLES} non-stale rows to run; have {n}.", file=sys.stderr)
        return 2
    if n < _finetune.COLD_START_WARNING_THRESHOLD:
        print(
            f"\nNote: {n} rows is below the ~{_finetune.COLD_START_WARNING_THRESHOLD} the research "
            "(cacheverifier.com/research, Q2) found fine-tuning reliably positive across every\n"
            "      dataset. A small-sample result here is real but noisy -- re-run as feedback grows."
        )

    import tempfile

    out_dir = tempfile.mkdtemp(prefix="cacheverifier_healthcheck_")
    print(f"\nfine-tuning locally (nothing sent) -> {out_dir}")
    print("this takes a few minutes on CPU; the base model downloads once on first run...\n")

    result = _finetune.run_healthcheck(examples, out_dir, base_model=_finetune.DEFAULT_BASE_MODEL, epochs=args.epochs)

    print("results")
    print("-" * 66)
    print(f"  train / calibrate / test:         {result.n_train} / {result.n_calibrate} / {result.n_test}")
    print(f"  stock verifier   held-out AUC:    {_fmt(result.auc_baseline)}")
    print(f"  fine-tuned       held-out AUC:    {_fmt(result.auc_tuned)}   (delta {result.auc_delta:+.4f})")
    print(f"  label-noise proxy (disagreement): {_fmt(result.train_label_disagreement_rate, pct=True)}")
    if result.train_label_disagreement_rate is not None:
        if result.train_label_disagreement_rate > _finetune.NOISE_HOLD_BACK_DISAGREEMENT_RATE:
            print("      -> HIGH: the hosted service would hold this model back for review")
        elif result.train_label_disagreement_rate > _finetune.NOISE_WARNING_DISAGREEMENT_RATE:
            print("      -> elevated: worth auditing how your correctness signal is produced")
    print(f"  train / test positive rate:       {_fmt(result.train_positive_rate, pct=True)} / {_fmt(result.test_positive_rate, pct=True)}")
    print(f"  ceiling status:                   {result.ceiling_status or 'n/a'}")
    if result.threshold is not None:
        print(
            f"  operating point @ picked threshold: hit rate {_fmt(result.threshold_hit_rate, pct=True)}, "
            f"error rate {_fmt(result.threshold_error_rate, pct=True)}"
        )
    print("\nverdict")
    print("-" * 66)
    print(f"  {_verdict(result.auc_delta)}\n")

    if args.emit_summary is not None:
        summary = {
            "n_rows_read": len(rows),
            "n_usable": n,
            "n_stale_excluded": n_stale,
            "n_train": result.n_train,
            "n_calibrate": result.n_calibrate,
            "n_test": result.n_test,
            "auc_stock": round(result.auc_baseline, 4),
            "auc_finetuned": round(result.auc_tuned, 4),
            "auc_delta": round(result.auc_delta, 4),
            "train_label_disagreement_rate": (
                round(result.train_label_disagreement_rate, 4)
                if result.train_label_disagreement_rate is not None
                else None
            ),
            "train_positive_rate": round(result.train_positive_rate, 4),
            "test_positive_rate": round(result.test_positive_rate, 4),
            "ceiling_status": result.ceiling_status,
            "threshold_hit_rate": result.threshold_hit_rate,
            "threshold_error_rate": result.threshold_error_rate,
        }
        args.emit_summary.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"aggregate summary written to {args.emit_summary}")
        print("(floats and counts only -- no query or answer text -- safe to share for a human read)\n")

    if args.keep_model:
        print(f"fine-tuned model kept at {result.model_path}")
    else:
        import shutil

        shutil.rmtree(out_dir, ignore_errors=True)

    return 0

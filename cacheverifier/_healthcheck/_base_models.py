"""Base-model choices for `cacheverifier healthcheck --base-model`.

Kept free of torch / sentence-transformers so `cacheverifier healthcheck
--help` can list the choices without the `healthcheck` extra installed.
The aliases and model ids match the hosted service's `base_model` fine-tune
parameter, so a local run is comparable to `POST /v1/finetune/dry-run`
with the same value.
"""

from __future__ import annotations

import re
from pathlib import Path

DEFAULT_BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
"""The stock off-the-shelf verifier every tenant starts on -- the same base
model the hosted service fine-tunes from. English only."""

BASE_MODEL_CHOICES: dict[str, str] = {
    "ms_marco": DEFAULT_BASE_MODEL,
    # SNLI+MultiNLI entailment pretraining, same size as ms_marco. Ships a
    # 3-class head; see _finetune for how it is scored and fine-tuned.
    "nli": "cross-encoder/nli-MiniLM2-L6-H768",
    # Non-English (e.g. Chinese) traffic: the English bases turn most
    # Chinese characters into [UNK] at tokenization.
    "multilingual": "BAAI/bge-reranker-base",
    "multilingual_small": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
}

ENGLISH_ONLY_BASE_MODELS = frozenset({BASE_MODEL_CHOICES["ms_marco"], BASE_MODEL_CHOICES["nli"]})

_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")

CJK_WARNING_RATIO = 0.3
"""Share of non-whitespace characters that are CJK above which an
English-only base model gets a warning."""


def resolve_base_model(value: str) -> str:
    """An alias from `BASE_MODEL_CHOICES`, a Hugging Face model id
    ("org/name"), or a local model directory (e.g. one downloaded from a
    mirror or ModelScope). Raises ValueError for anything else, so a typo'd
    alias fails immediately instead of after a failed download."""
    if value in BASE_MODEL_CHOICES:
        return BASE_MODEL_CHOICES[value]
    # `Path("")` is the current directory -- never treat an empty value as
    # a local model.
    if value.strip() and Path(value).expanduser().is_dir():
        return str(Path(value).expanduser())
    if "/" in value and not value.startswith("/"):
        return value
    raise ValueError(
        f"unknown base model {value!r}: use one of {', '.join(BASE_MODEL_CHOICES)}, "
        "a Hugging Face model id (org/name), or a local model directory"
    )


def cjk_ratio(texts: list[str]) -> float:
    chars = "".join("".join(t.split()) for t in texts)
    if not chars:
        return 0.0
    return len(_CJK_RE.findall(chars)) / len(chars)

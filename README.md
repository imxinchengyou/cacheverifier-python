# cacheverifier

[![CI](https://github.com/imxinchengyou/cacheverifier-python/actions/workflows/ci.yml/badge.svg)](https://github.com/imxinchengyou/cacheverifier-python/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/cacheverifier)](https://pypi.org/project/cacheverifier/)
[![Python](https://img.shields.io/pypi/pyversions/cacheverifier)](https://pypi.org/project/cacheverifier/)

Python client for **[CacheVerifier](https://www.cacheverifier.com)** — a hosted API that
verifies semantic-cache hits. Given a query and a candidate cached answer, it approves or
rejects serving that answer from cache, so a similarity match that is *close but wrong*
doesn't become a silent error in your app.

CacheVerifier does **not** run your cache or do similarity search. Your cache backend does
its own lookup first; you call `verify()` only on the candidates in the similarity "gray
zone", where a plain threshold match might be wrong.

- Docs / API reference: <https://www.cacheverifier.com/docs>
- Why similarity ≠ correctness: <https://www.cacheverifier.com/why-similarity-fails>
- The research behind it (paper + benchmarks): <https://github.com/imxinchengyou/CacheVerifier>

## Install

```bash
pip install cacheverifier
# with the GPTCache adapter:
pip install "cacheverifier[gptcache]"
# with the offline Health Check (adds torch + sentence-transformers):
pip install "cacheverifier[healthcheck]"
```

Requires Python 3.9+. The only runtime dependency is `httpx` — the extras above
are opt-in.

## Quickstart

Get a free API key at <https://www.cacheverifier.com> (self-serve verify and fine-tuning
are free forever, no card).

```python
from cacheverifier import CacheVerifier

cv = CacheVerifier(api_key="cv_...")

query = "how do I cancel my subscription"
candidate = "Go to Settings > Billing > Pause subscription for a month."  # from your cache

result = cv.verify(query, candidate)
if result.approved:
    answer = candidate                       # verified hit — skip the LLM call
else:
    answer = call_your_llm(query)            # not trustworthy — fall through

# Later, once you know if it was actually right (thumbs-down, reopened ticket, ...):
cv.feedback(query, answer, was_correct=True, similarity_score=0.86)
```

`verify()` returns a `VerifyResult`:

| field | meaning |
|---|---|
| `approved` | serve the cached answer (`True`) or fall through (`False`) |
| `score` / `threshold` | `approved` is `score >= threshold` |
| `model_version` | `"stock"`, `"v<id>"` (fine-tuned), `"cold_start_fail_closed"` / `"cold_start_auto_pending"`, or `"verify_unavailable"` |
| `latency_ms` | server-side inference time |
| `degraded` | `True` when the call failed and this result was synthesized client-side |

## On your request path

`verify()` runs inline with your traffic, so its defaults are conservative:

- **1s timeout** (`verify_timeout=`), separate from the 10s `timeout=` used for
  fine-tuning / feedback / monitoring calls. Warm verification is tens of
  milliseconds server-side; 1s covers the network round trip and a cold model
  load after a deploy without letting a stuck verifier stall your request.
- **Fails closed.** On a timeout, connection error, or 5xx, `verify()` does not
  raise — it logs a warning on the `cacheverifier` logger and returns a
  `VerifyResult` with `degraded=True` and `approved=False`, so you fall through
  to your LLM exactly as you would on a cache miss. A `4xx` (bad key, bad
  request, rate limit) still raises `CacheVerifierError`.

```python
cv = CacheVerifier(api_key="cv_...", verify_timeout=1.0, fail_open=False)  # the defaults

# fail_open=True instead returns approved=True on an outage — only if a
# stale-or-near-miss answer is acceptable for that traffic:
cv = CacheVerifier(api_key="cv_...", fail_open=True)
```

There is no formal uptime SLA yet, which is the other reason the fallback path
is a built-in default rather than left to you.

## GPTCache

Drop the verifier into a GPTCache pipeline as its similarity evaluator — no fork required:

```python
from gptcache import cache
from cacheverifier.integrations.gptcache import CacheVerifierEvaluation

evaluator = CacheVerifierEvaluation(api_key="cv_...")
cache.init(similarity_evaluation=evaluator, ...)

# when you learn a served hit's real outcome:
evaluator.report_feedback(query, answer, was_correct=False, similarity_score=0.9)
```

See [`examples/gptcache_example.py`](examples/gptcache_example.py).

## Fine-tuning

Once you have ~20+ feedback rows (the service found fine-tuning is often a net negative
below ~1,000 on the hardest data — see the [research](https://github.com/imxinchengyou/CacheVerifier)),
train a verifier on your own gray-zone labels:

```python
job = cv.finetune()                       # or cv.finetune(target_risk=0.01, cost_ratio=5.0)
job = cv.get_finetune_job(job["id"])      # poll until status == "done"
print(job["auc_baseline"], job["auc_tuned"])

# a model can finish as "held_for_review" — promote it explicitly:
if job.get("result_model_version"):
    cv.activate_model_version(job["result_model_version"])
```

`cv.dry_run([...])` reports the same baseline-vs-tuned AUC on examples you pass directly,
without writing anything or deploying a model.

## Local Health Check (offline)

`cv.dry_run()` still uploads your examples to the API. If that's a blocker — a
compliance review, or just not wanting production traffic to leave your network —
run the identical stock-vs-fine-tuned evaluation entirely on your own machine:

```bash
pip install "cacheverifier[healthcheck]"

cacheverifier healthcheck traffic.jsonl
cacheverifier healthcheck traffic.jsonl --emit-summary summary.json
```

`traffic.jsonl` is a JSON array or JSONL of `{"query", "candidate_answer", "was_correct"}`
rows **in arrival order** (the train/calibrate/test split is chronological, matching the
hosted service so the numbers are comparable). Optional per row: `"stale": true`.

Nothing is sent anywhere — the base model downloads once from Hugging Face, then it's
fully offline. `--emit-summary` writes an aggregate-only JSON file (AUCs, counts, rates —
no query or answer text) that's safe to share for a human read.

**Choosing the base model.** `--base-model` takes the same values as the hosted
service's `base_model` fine-tune parameter, so local and hosted numbers stay comparable:

| `--base-model` | model | use for |
|---|---|---|
| `ms_marco` (default) | `cross-encoder/ms-marco-MiniLM-L6-v2` | English traffic |
| `nli` | `cross-encoder/nli-MiniLM2-L6-H768` | English; entailment pretraining, stronger on negation / entity swaps |
| `multilingual` | `BAAI/bge-reranker-base` | Chinese and other non-English traffic (~6 GB RAM to fine-tune) |
| `multilingual_small` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | same, smaller (~3 GB RAM) |

The English bases read most Chinese characters as unknown tokens, so on Chinese
traffic their result doesn't mean anything — the CLI warns when it sees mostly CJK text
on an English base. You can also pass any Hugging Face model id or a local model
directory. If Hugging Face is unreachable from your network, set `HF_ENDPOINT` to a
mirror, or download the model elsewhere and pass its directory:

```bash
cacheverifier healthcheck traffic.jsonl --base-model multilingual
HF_ENDPOINT=https://hf-mirror.com cacheverifier healthcheck traffic.jsonl --base-model multilingual_small
cacheverifier healthcheck traffic.jsonl --base-model ./models/bge-reranker-base
```

```
results
------------------------------------------------------------------
  train / calibrate / test:         3349 / 419 / 419
  stock verifier   held-out AUC:    0.6120
  fine-tuned       held-out AUC:    0.7080   (delta +0.0960)
  label-noise proxy (disagreement): 11.4%
  ceiling status:                   still_improvable

verdict
------------------------------------------------------------------
  IMPROVED   -- fine-tuning on your own data helps this traffic
```

## API surface

| method | endpoint |
|---|---|
| `verify(query, candidate_answer)` | `POST /v1/verify` |
| `verify_batch(pairs)` | `POST /v1/verify/batch` |
| `feedback(...)` / `feedback_batch(items)` | `POST /v1/feedback` / `/batch` |
| `finetune(...)` / `dry_run(examples, ...)` | `POST /v1/finetune/jobs` / `/dry-run` |
| `get_finetune_job(id)` / `list_finetune_jobs()` | `GET /v1/finetune/jobs[/id]` |
| `activate_model_version(id)` | `POST /v1/finetune/model-versions/{id}/activate` |
| `drift_status()` | `GET /v1/monitor/drift-status` |
| `gray_zone_threshold()` | `GET /v1/monitor/gray-zone-threshold` |
| `usage()` / `savings()` | `GET /v1/usage/status` / `/savings` |

Non-2xx responses raise `CacheVerifierError` (`.status_code`, `.detail`).

## License

MIT — see [`LICENSE`](LICENSE). (The [research repository](https://github.com/imxinchengyou/CacheVerifier)
is separately licensed; this client is not.)

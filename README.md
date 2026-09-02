# cacheverifier

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
```

Requires Python 3.9+. The only runtime dependency is `httpx`.

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
| `model_version` | `"stock"`, `"v<id>"` (fine-tuned), or `"cold_start_fail_closed"` |
| `latency_ms` | server-side inference time |

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

# cacheverifier

[![CI](https://github.com/imxinchengyou/cacheverifier-python/actions/workflows/ci.yml/badge.svg)](https://github.com/imxinchengyou/cacheverifier-python/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/cacheverifier)](https://pypi.org/project/cacheverifier/)
[![Python](https://img.shields.io/pypi/pyversions/cacheverifier)](https://pypi.org/project/cacheverifier/)

Python client for **[CacheVerifier](https://www.cacheverifier.com)**: verification and
operations for semantic caches. It checks the cache hits a similarity threshold can't call,
then keeps your cache right after launch.

- **Verify the uncertain hits.** Given a query and a candidate cached answer, approve or
  reject serving it in tens of milliseconds, so a match that is *close but wrong* doesn't
  become a silent error. Fine-tuned on your own feedback.
- **Know your real error rate.** Randomized audits of approved hits give an exact
  (Clopper-Pearson) upper bound on how often a served answer is wrong, measured on your
  traffic, not a benchmark.
- **Certified thresholds.** Certify a threshold with Conformal Risk Control at fine-tune
  time, and optionally let an online (ACI) threshold follow what the audits find.
- **Catch drift and recalibrate.** When traffic moves away from what the verifier learned,
  you're alerted, and on Managed plans it is retrained on your feedback automatically.
- **Know which entries to fix.** `entries_to_refresh()` lists the cache entries your
  feedback reported stale, or wrong more often than right, with your own cache keys.
- **Run it in your stack.** On Managed plans, `LocalVerifier` scores with your fine-tuned
  model locally: no network hop, your texts stay with you, nothing metered.

CacheVerifier does **not** run your cache or do similarity search. Your cache backend does
its own lookup first; you call `verify()` only on the candidates in the similarity "gray
zone", where a plain threshold match might be wrong.

| | answers | examples |
|---|---|---|
| Static-threshold caches | "how similar is it?" | GPTCache, LiteLLM, Portkey, Redis LangCache |
| Adaptive-threshold caches | "how similar, with a learned per-entry cutoff?" (replaces your cache) | vCache |
| **Verification layer** | "is this cached answer right for this question?" (sits on top of your cache) | **CacheVerifier** |

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
# to score with your fine-tuned model on your own machines (same deps):
pip install "cacheverifier[local]"
```

Requires Python 3.9+. The only runtime dependency is `httpx` — the extras above
are opt-in.

## Quickstart

Get a free API key at <https://www.cacheverifier.com> (verify, fine-tuning, audits and
entries to refresh are free, no card).

```python
from cacheverifier import CacheVerifier

cv = CacheVerifier(api_key="cv_...")

query = "can I change the delivery address after my parcel ships"
candidate = "You can change the delivery address any time before the parcel ships."  # from your cache

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
| `notice` | why no model decided the call (cold start); `None` otherwise |
| `shadow` | cold start only: a `ShadowVerdict` with what the shared stock model would have decided. The call is still rejected, so treat it as a preview |

A new account starts in cold start: every call returns `approved=False` until a
Health Check shows the shared model is accurate enough on your data. Look at
`result.shadow` to see what it would have decided in the meantime:

```python
result = cv.verify("when does my flight board", "Boarding starts 40 minutes before departure.")
if result.shadow:
    print(result.notice)
    print("stock model would approve:", result.shadow.approved, result.shadow.score)
```

## On your request path

`verify()` runs inline with your traffic, so its defaults are conservative:

- **1s timeout** (`verify_timeout=`), separate from the 10s `timeout=` used for
  fine-tuning / feedback / monitoring calls. Warm verification is tens of
  milliseconds server-side; 1s covers the network round trip and a cold model
  load after a deploy without letting a stuck verifier stall your request.
- **Fails closed.** On a timeout, connection error, 5xx, or a rate-limited 429
  (one carrying `Retry-After`), `verify()` does not raise — it logs a warning on
  the `cacheverifier` logger and returns a `VerifyResult` with `degraded=True`
  and `approved=False`, so you fall through to your LLM exactly as you would on
  a cache miss. Other `4xx` responses still raise `CacheVerifierError`: a bad
  key, a bad request, or a 429 without `Retry-After`, which means your monthly
  verify quota is used up.

```python
cv = CacheVerifier(api_key="cv_...", verify_timeout=1.0, fail_open=False)  # the defaults

# fail_open=True instead returns approved=True on an outage — only if a
# stale-or-near-miss answer is acceptable for that traffic:
cv = CacheVerifier(api_key="cv_...", fail_open=True)
```

There is no formal uptime SLA yet, which is the other reason the fallback path
is a built-in default rather than left to you.

## LangChain

`VerifiedCache` wraps any LangChain cache, typically a semantic one, and only
serves a hit after CacheVerifier approves it; a rejected hit is a miss, so
your model is called as usual. For chat models it verifies against the last
user message.

```bash
pip install "cacheverifier[langchain]"
```

```python
from langchain_core.globals import set_llm_cache
from langchain_redis import RedisSemanticCache
from cacheverifier.integrations.langchain import VerifiedCache

set_llm_cache(
    VerifiedCache(
        RedisSemanticCache(redis_url="redis://localhost:6379", embeddings=embeddings),
        api_key="cv_...",
    )
)
```

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

## Entries to refresh

Send your cache's own key with feedback, and the service tells you which
entries to fix: ones reported stale, or wrong more often than right.

```python
cv.feedback(query, cached_answer, was_correct=False, stale=True, cache_entry_id="faq:shipping")

for e in cv.entries_to_refresh():
    my_cache.delete(e["cache_entry_id"])   # or regenerate the answer
    cv.resolve_entry(e["key"])
```

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

## Local scoring (Managed plans)

`LocalVerifier` runs the model CacheVerifier fine-tuned for you on your own
machines: no network round trip on your request path, and your query and
answer texts stay with you. Fine-tuning, drift monitoring, audits and
certification keep running on the service.

```python
from cacheverifier import LocalVerifier

cv = LocalVerifier(api_key="cv_...")       # downloads your live model in the background
r = cv.verify(query, cached_answer, similarity_score=0.87, cached_query=cached_question)
if r.approved:
    ...                                    # serve from cache
else:
    answer = call_llm(query)
    if r.audit:                            # a randomized audit: report whether the cached answer was right
        cv.feedback(query, cached_answer, was_correct=same_answer(answer, cached_answer), result=r)
```

- It decides exactly as the hosted `/v1/verify` would with that model: every download
  carries reference verdicts from the service, and a model that doesn't reproduce them
  locally is not used.
- When a newer model goes live (after a fine-tune or a scheduled recalibration), it is
  downloaded, checked, and switched to without a restart.
- While auditing is on, each decision is uploaded as scores, verdicts and a keyed hash of
  the pair, never the texts, so audits and `/v1/monitor/risk` work as with the hosted API.
- Until a model is ready, calls go to the hosted API (`fallback="hosted"`, the default)
  or are rejected (`fallback="reject"`). Pass `wait=True` to block until the first
  download finishes (a model is ~330MB).
- The audit draw happens in this client. The risk bound assumes it is independent of
  whether the answer is right, which the client guarantees unless it is modified.
- Not supported yet: the online ACI threshold (local scoring uses the model's fixed one).

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

## Recommended production setup

- Keep randomized audits on (the default for new accounts) and report each audited call
  with `feedback(..., verification_id=r.verification_id)`, or `result=r` on
  `LocalVerifier`. That is what makes the measured error rate real.
- Fine-tune with `cost_ratio=` (how many times worse a wrong reuse is than an extra LLM
  call) instead of the default threshold, and turn on ACI once audits are flowing.
- Send `cache_entry_id=` with feedback so `entries_to_refresh()` hands back your keys.
- At volume, score locally with `LocalVerifier`.

## Known limits

- The hosted verifier is **English-only** today. Don't send Chinese or other non-English
  traffic to it; multilingual base models are evaluated in the docs but not deployed.
- Query + answer share a **128-token** input; longer pairs are truncated head and tail.
- Verify endpoints allow about **20 requests/second per API key**. Use `verify_batch()` or
  `LocalVerifier` beyond that.
- No formal uptime SLA yet; `verify()` fails closed by default (see above).

## API surface

| method | endpoint |
|---|---|
| `verify(query, candidate_answer, similarity_score=..., cached_query=...)` | `POST /v1/verify` |
| `verify_batch(pairs)` | `POST /v1/verify/batch` |
| `feedback(...)` / `feedback_batch(items)` | `POST /v1/feedback` / `/batch` |
| `finetune(...)` / `dry_run(examples, ...)` | `POST /v1/finetune/jobs` / `/dry-run` |
| `get_finetune_job(id)` / `list_finetune_jobs()` | `GET /v1/finetune/jobs[/id]` |
| `activate_model_version(id)` | `POST /v1/finetune/model-versions/{id}/activate` |
| `drift_status()` | `GET /v1/monitor/drift-status` |
| `entries_to_refresh()` / `resolve_entry(key)` | `GET /v1/monitor/entries-to-refresh` / `POST .../{key}/resolve` |
| `gray_zone_threshold()` | `GET /v1/monitor/gray-zone-threshold` |
| `usage()` / `savings()` | `GET /v1/usage/status` / `/savings` |
| `LocalVerifier.verify(...)` / `verify_batch(...)` | local; decisions to `POST /v1/verify/decisions` |
| `LocalVerifier.refresh()` | `GET /v1/models/active` + `/v1/models/{id}/files/{name}` |

Non-2xx responses raise `CacheVerifierError` (`.status_code`, `.detail`).

## License

MIT — see [`LICENSE`](LICENSE). (The [research repository](https://github.com/imxinchengyou/CacheVerifier)
is separately licensed; this client is not.)

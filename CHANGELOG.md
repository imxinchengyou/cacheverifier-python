# Changelog

## 0.3.1

`cacheverifier healthcheck` now actually fine-tunes. It passed no
`warmup_steps` to `CrossEncoder.fit`, whose default of 10000 kept a typical
run below a few percent of its learning rate, so `auc_tuned` came back equal
(or nearly equal) to `auc_baseline`. Warmup is now 10% of total steps and the
default is 3 epochs (`--epochs`), the same as the hosted service since
verifier-core 0.2.0. On a 500-row later holdout this took LmArena AUC from
0.903 to 0.932 and AmazonHelp from 0.551 to 0.700. Runs take about 3x as long.

## 0.3.0

Make the request-path defaults safe. `verify()` and `verify_batch()` now use a
separate **1s timeout** (`verify_timeout=`, vs the unchanged 10s `timeout=` for
control-plane calls), and on a timeout / connection error / 5xx they **fail
closed** instead of raising: you get a synthetic `VerifyResult` with
`degraded=True` and `approved=False` (fall through to your LLM). Pass
`fail_open=True` to have those cases return `approved=True` instead. 4xx
responses (bad key, bad request, rate limit) still raise `CacheVerifierError`.
`VerifyResult` gains a `degraded` field; `model_version` is
`"verify_unavailable"` on a synthesized result. The GPTCache adapter takes the
same `verify_timeout` / `fail_open` arguments and returns `0.0` (don't reuse) on
a verifier outage by default.

## 0.2.0

Add a `cacheverifier` console script with a `healthcheck` subcommand:
`cacheverifier healthcheck traffic.jsonl` runs the hosted Health Check
Report's stock-vs-fine-tuned held-out AUC evaluation (`POST /v1/finetune/dry-run`)
entirely offline — no queries or answers leave the machine. Needs the new
`healthcheck` extra (`pip install "cacheverifier[healthcheck]"`: torch,
sentence-transformers, numpy); the base client stays `httpx`-only.
`--emit-summary` writes an aggregate-only JSON file with no text.

## 0.1.0

Initial release: thin `httpx`-based client for `/v1/verify`, `/v1/verify/batch`,
`/v1/feedback`(`/batch`), fine-tune jobs + dry run, drift and gray-zone-threshold
monitoring, and usage/savings. GPTCache `SimilarityEvaluation` adapter under
`cacheverifier.integrations.gptcache`.

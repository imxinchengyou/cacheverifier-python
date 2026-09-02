# Changelog

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

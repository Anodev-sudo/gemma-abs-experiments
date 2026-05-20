# Summary Generation Tool

Reusable writer/reviewer pipeline for generating `abstract -> feed summary`
training rows. It is intended for focused corrective overlays where we want
strict, auditable examples rather than a broad one-off generation dump.

## What It Does

- normalizes a candidate JSONL into a stable candidate pool
- probes configured model routes before spending a batch
- runs a writer model, then applies local quality gates
- runs a strict reviewer model over locally eligible rows
- retries row-local failures through the configured route ladder
- materializes accepted rows into a release directory
- writes manifests, request payloads, failed rows, raw JSON attempts, and yield reports

The current checked-in preset reproduces the q5 precision Kimi/Qwen overlay
shape, but the runner itself is generic and driven by config.

## Files

- `run_pipeline.py`: CLI runner and reusable pipeline functions
- `../../configs/synthetic_summary_generation/q5_precision_kimi_qwen_overlay_v1.json`:
  first preset, using Kimi as writer and Qwen as strict reviewer
- `../../tests/test_summary_generation_pipeline.py`: mocked regression tests

## Stages

- `prepare`: write `candidate_pool/candidates.jsonl` and a manifest
- `probe-routes`: send small live probes and choose writer/reviewer routes
- `write`: run writer requests and preserve raw outputs
- `local-filter`: gate writer outputs before reviewer spend
- `verify`: run reviewer requests
- `retry-failed`: rerun failed writer/reviewer rows through the route ladder
- `materialize`: write accepted training rows and review audit files
- `report`: write JSON and Markdown yield reports
- `all`: run the normal end-to-end flow

## Output Layout

Each run root contains:

- `candidate_pool/`
- `route_probe/`
- `writer_run/`
- `local_eligible_for_review/`
- `reviewer_run/`
- `materialized_release/`
- `yield_report.json`
- `yield_report.md`

Raw run artifacts belong under `hf_bucket/experiments/`. Promoted or reusable
training rows can be materialized under `hf_dataset/releases/`.

## Config Notes

Configs are JSON only, so the tool does not add a PyYAML dependency. A config
must define:

- `input.path`
- `output.run_root`
- `writer.prompt_path` and writer `routes`
- `reviewer.prompt_path` and reviewer `routes`
- `local_gates`
- `materialize.source_split`

Routes are tried in order. A route can switch structured output off with
`"response_format_mode": "plain_json"`, which is useful when a provider accepts
the model but rejects schema mode.

## Validation

The no-network test suite exercises route fallback, strict local gates, JSON
repair, materialization, and yield reporting.


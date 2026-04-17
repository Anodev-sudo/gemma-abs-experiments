# Run TUI

Small terminal browser for synthetic-summary run artifacts.

## What it reads

The viewer understands these files when they exist in a run directory:

- `run_manifest.json`
- `sample_source_rows.jsonl`
- `generation_requests.jsonl`
- `generation_outputs.jsonl`
- `review_requests.jsonl`
- `review_outputs.jsonl`
- `training_rows.jsonl`
- `dropped_rows.jsonl`

It is intentionally tolerant of partial runs.

This repo now has a mix of:

- prepare-only runs with sampled source rows and generator requests
- legacy full runs using reviewer `v1`
- newer full runs using reviewer `v2` with explicit `accept` decisions

When full runs exist, the TUI will show:

- generator system and user messages
- generator raw output and parsed candidate summary
- reviewer system and user messages
- reviewer raw output
- reviewer decision, change kind, final summary, and drop reason
- accepted vs rewritten vs dropped outcomes across reviewer v1 and v2 runs
- final summary by default, with candidate/final comparison in a separate tab

Future idea:

- a minimal mode that trims header/detail verbosity even further

## Launch

From the repo root:

```bash
npm run run-tui
```

Open a specific run:

```bash
npm run run-tui -- hf_bucket/runs/deepseek_v32_smoke_100
```

The canonical run root is `hf_bucket/runs/`. A local `runs/` compatibility alias may exist during migration, but new commands should use the `hf_bucket/` path.

Print a non-interactive summary:

```bash
npm run run-tui:summary
```

## Keys

- `Up` / `Down` or `j` / `k`: move through rows
- `Left` / `Right` or `h` / `l`: cycle tabs
- `PageUp` / `PageDown`: scroll the detail pane
- `Space` / `b` or `Ctrl-d` / `Ctrl-u`: scroll the detail pane faster
- `1` / `2` / `3` / `4` / `5`: switch tabs
  - `1`: overview
  - `2`: compare
  - `3`: generator
  - `4`: reviewer
  - `5`: raw
- `i`: toggle expanded run info in the header
- `f`: cycle filters
- `/`: search
- `c`: clear search
- `[` / `]`: switch runs
- `r`: reload from disk
- `q` or `Esc`: quit

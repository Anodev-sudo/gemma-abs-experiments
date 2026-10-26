# Experiments

Short sprints from April 2026, run before the main training work. In each one,
several coding agents worked in parallel on the same small sample, each trying
one change, and wrote a report. The run outputs themselves are not in this
repository.

- `subagent_prompt_sprint_2026-04-09/` and `subagent_prompt_sprint_2026-04-11_round2/`:
  each agent tried one change to the DeepSeek writer prompt while the Qwen
  reviewer prompt stayed fixed, then reported what changed in kept and dropped
  rows.
- `ideal_summary_sprint_2026-04-10/`: four agents wrote summaries for the same
  10 papers directly, without the writer/reviewer pipeline, to pin down what a
  good feed summary looks like.
- `5k_eval_2026-04-12/`: audits of the keep and drop decisions in the
  5,000-paper pilot. Most audits found the kept rows sound, and three of the
  five found the filters dropping good summaries.
- `hard_negative_pack_v1/`: seed papers for building near-miss negatives, such
  as summaries that bind a number to the wrong quantity.

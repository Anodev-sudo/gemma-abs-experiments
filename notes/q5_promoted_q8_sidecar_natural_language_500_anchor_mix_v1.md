# q5 Promoted Q8 Sidecar Natural-Language Anchor Mix v1

## Goal

Continue the low-LR natural-language style probe without letting the 500-row
plain-language set fully overwrite the current promoted q5-sidecar behavior.

The no-anchor sweep found the first real llama.cpp movement at `3e-6`, but the
semantic drift was still too high. This run keeps that LR, adds a midsize
low-weight replay anchor set, and runs longer.

## Dataset

- Builder: `scripts/build_q5_natural_language_anchor_mix_v1_dataset.py`
- Output: `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_500_anchor_mix_v1/dataset_500nl_1000faithanchors_w025/train.jsonl`
- Rows: `1500`
- Natural-language rows: `500` at weight `1.0`
- Faithfulness anchors: `1000` at weight `0.25`
- Prompt mode: `title_abstract`
- Eval exclusions: the fresh20 packet used for q5 runtime checks

Anchor rows use the natural-language prompt wrapper but keep the older stable
faithfulness summaries. They are replay anchors, not final style targets.

## Initial Training Plan

- Config: `configs/sft/q5_promoted_q8_sidecar_natural_language_500_anchor_mix_v1_lr3e6_steps150.yaml`
- Start adapter: `hf_bucket/experiments/sft/q5_promoted_q8_adapter_sidecar_probe_v1`
- Output: `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_500_anchor_mix_v1/lr3e6_steps150`
- LR: `3e-6`
- Steps: `150`
- Save steps: `30`
- Effective batch: `4`
- Weighted loss normalization: `label_count`

## Expected Read

This is a diagnostic, not a promotion run. A good result would show more
natural-language shape than the promoted baseline while reducing the false
specificity seen in the no-anchor `3e-6` and aggressive `1e-5` runs.

## Corrected LR-Creep Plan

The intended next move is three LR steps forward while keeping the midsize
low-weight anchor set:

| config | LR | output |
|---|---:|---|
| `configs/sft/q5_promoted_q8_sidecar_natural_language_500_anchor_mix_v1_lr4e6_steps150.yaml` | `4e-6` | `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_500_anchor_mix_v1/lr4e6_steps150` |
| `configs/sft/q5_promoted_q8_sidecar_natural_language_500_anchor_mix_v1_lr5e6_steps150.yaml` | `5e-6` | `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_500_anchor_mix_v1/lr5e6_steps150` |
| `configs/sft/q5_promoted_q8_sidecar_natural_language_500_anchor_mix_v1_lr6e6_steps150.yaml` | `6e-6` | `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_500_anchor_mix_v1/lr6e6_steps150` |

At `150` steps, the 1500-row anchor mix sees the same `0.4` epoch fraction as
the 50-step 500-row no-anchor probes.

## Corrected LR-Creep Results

Training completed for all three forward LR steps.

| LR | train loss | runtime | output |
|---:|---:|---:|---|
| `4e-6` | `4.677` | `592.2s` | `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_500_anchor_mix_v1/lr4e6_steps150` |
| `5e-6` | `4.664` | `671.6s` | `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_500_anchor_mix_v1/lr5e6_steps150` |
| `6e-6` | `4.652` | `492.3s` | `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_500_anchor_mix_v1/lr6e6_steps150` |

All runs used the same `1500` row anchor mix, `12224.75` effective weighted
assistant-token mass, and `0.4` epoch fraction.

### Llama Fresh20 Metrics

| LR | step | avg words | max words | >14 | >15 | <10 | digit rows | changed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `3e-6` | `150` | `12.40` | `17` | `3` | `3` | `1` | `1` | `10` |
| `4e-6` | `90` | `12.20` | `17` | `2` | `2` | `1` | `1` | `10` |
| `4e-6` | `120` | `12.35` | `17` | `2` | `2` | `1` | `1` | `11` |
| `4e-6` | `150` | `12.40` | `17` | `3` | `2` | `1` | `1` | `11` |
| `5e-6` | `60` | `13.00` | `21` | `3` | `3` | `1` | `1` | `12` |
| `5e-6` | `150` | `13.30` | `21` | `5` | `4` | `1` | `2` | `12` |
| `6e-6` | `30` | `12.30` | `17` | `3` | `3` | `2` | `1` | `10` |
| `6e-6` | `150` | `13.30` | `21` | `5` | `4` | `1` | `2` | `12` |

The best mechanical checkpoint is `4e-6` at `90` or `120`. Past that, `5e-6`
and `6e-6` begin to reintroduce long 21-word rows and denser technical phrasing.

### PEFT Fresh20 Metrics

PEFT was run for `4e-6` checkpoints `90` and `120` only.

| label | avg words | max words | >14 | >15 | <10 | digit rows | changed |
|---|---:|---:|---:|---:|---:|---:|---:|
| promoted baseline | `12.05` | `16` | `3` | `2` | `2` | `0` | `0` |
| `4e-6` checkpoint `90` | `12.10` | `16` | `3` | `1` | `1` | `0` | `12` |
| `4e-6` checkpoint `120` | `12.05` | `16` | `3` | `1` | `0` | `0` | `14` |

PEFT is cleaner than llama, but llama remains the deployment surface.

### Corrected LR-Creep Manual Read

`4e-6` checkpoint `90` is the best diagnostic candidate from this pass:

- It reduces llama over-14 rows from `6` baseline to `2`.
- It reduces over-15 rows from `5` baseline to `2`.
- It removes the dense equation-like OneMax phrase.
- It shortens the ANOVA/ASCA acronym row.
- It moves about half the packet without the broad 21-word rambling seen at
  `5e-6` and `6e-6`.

Still not promotable:

- `2601.13411` regresses to: "The gravitational waves are not found in the
  first-principle black hole models."
- `2203.06383` still says "simplified action-ground state", which is awkward and
  not clearly simpler.
- `2501.12034` adds "traffic attacks" and "low-level systems"; plausible, but
  still a specificity creep.
- `4e-6` checkpoint `120` adds a dense probabilistic-graphs sentence with
  "exponential cone constraints and polynomial time."

Verdict: `4e-6` checkpoint `90` is the best anchored LR-creep diagnostic, but no
promotion from this fresh20 read. The anchor mix helps control blow-up, and the
useful LR band appears to be narrow: around `4e-6`, before `5e-6` starts getting
too hot.

## Result

Training completed in `511.2s` at `150` steps.

- Output: `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_500_anchor_mix_v1/lr3e6_steps150`
- Train rows: `1500`
- Labeled assistant tokens: `25691`
- Effective weighted token mass: `12224.75`
- Train loss: `4.693`
- Saved checkpoints: `30`, `60`, `90`, `120`, `150`

The `3e-6` run is the baseline anchored point for the corrected LR creep.

## Llama Fresh20 Eval

Eval root:
`hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_500_anchor_mix_v1/lr3e6_steps150/eval/llama_fresh20_natural_language_prompt_outputs`

| label | avg words | max words | >14 | >15 | <10 | digit rows | changed vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 13.05 | 18 | 6 | 5 | 2 | 2 | 0 |
| checkpoint 30 | 13.05 | 18 | 6 | 5 | 2 | 2 | 5 |
| checkpoint 60 | 12.55 | 17 | 4 | 4 | 2 | 1 | 8 |
| checkpoint 90 | 12.60 | 17 | 4 | 4 | 1 | 1 | 8 |
| checkpoint 120 | 12.35 | 17 | 3 | 3 | 2 | 1 | 10 |
| checkpoint 150 | 12.40 | 17 | 3 | 3 | 1 | 1 | 10 |

Best mechanical checkpoints are `120` and `150`. They reduce over-length and
digit-heavy rows, but still keep several stale promoted-baseline failures.

## PEFT Fresh20 Eval

Eval root:
`hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_500_anchor_mix_v1/lr3e6_steps150/eval/peft_fresh20`

| label | avg words | max words | >14 | >15 | <10 | digit rows | changed vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 12.05 | 16 | 3 | 2 | 2 | 0 | 0 |
| checkpoint 120 | 12.10 | 16 | 3 | 1 | 0 | 0 | 12 |
| checkpoint 150 | 12.20 | 16 | 3 | 1 | 0 | 0 | 12 |

PEFT is cleaner than llama on this sample, but llama remains the relevant
deployment surface.

## Manual Read

Good movement:

- `2307.11855`: the llama output removes the dense equation-like runtime phrase.
- `2408.06739`: late checkpoints shorten the ANOVA/ASCA acronym pile.
- `2312.08042`: PEFT turns a named method into a simpler description.
- `2603.11391`: PEFT improves the BEACON row into a more direct reader-facing
  hook.

Remaining problems:

- `2203.06383`: llama drifts into "simplified action-ground state", which is not
  simpler and may be awkwardly grounded.
- `2501.12034`: llama adds "traffic attacks" and "low-level systems"; plausible,
  but more specific than necessary.
- `2212.01114`: llama keeps the old "new model model" duplication unchanged.
- `2601.13411`: llama keeps the old awkward negative phrasing unchanged.

Conclusion: useful diagnostic, not a promotion candidate. Low-weight anchors
appear to make longer exposure less explosive, but they also damp the exact
natural-language style movement we were trying to measure.

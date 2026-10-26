# q5 promoted sidecar natural-language token-weighted gold v2

## Intent

Fresh diagnostic from the promoted q5/q8 sidecar using the same 994 strict gold
rows and 600 low-weight faithfulness anchors as v1, but with
specificity-preserving span weights.

The target is to keep the shorter feed-blurb movement from v1 checkpoint 96
without repeating its detail-erasure and grounding regressions.

## Planned run

- Dataset builder:
  `scripts/build_q5_natural_language_token_weighted_gold_v2_dataset.py`
- Config:
  `configs/sft/q5_promoted_q8_sidecar_natural_language_994_token_weighted_gold_v2_lr4e6_steps128.yaml`
- Dataset root:
  `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_token_weighted_gold_v2/dataset_994gold_600faithanchors_specificity_v2`
- Output root:
  `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_token_weighted_gold_v2/lr4e6_steps128`
- Start adapter:
  `hf_bucket/experiments/sft/q5_promoted_q8_adapter_sidecar_probe_v1`
- LR: `4e-6`
- Steps: `128`
- Save steps: `32`
- Effective batch: `4`

## Span policy

V2 keeps polarity and result-shape pressure, but shifts the main signal from
first non-title tokens to source-backed specificity:

- `polarity_caveat`: `2.5`
- `scope_qualifier`: `2.1`
- `math_result_shape`: `2.0`
- `domain_object_term`: `1.65`
- `payoff_verb`: `1.6`
- `source_specific_anchor`: `1.55`
- `source_backed_shared_key_term`: `1.35`

Title-risk gold rows use base `0.75`, risky zero-span fallback uses `0.90`,
and under-10 summaries cap base at `0.85` unless they contain polarity or scope
spans.

## Actual dataset

Built:
`hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_token_weighted_gold_v2/dataset_994gold_600faithanchors_specificity_v2/train.jsonl`

Validation passed:

| check | value |
|---|---:|
| total rows | `1594` |
| strict gold rows | `994` |
| faithfulness anchors | `600` |
| title-risk gold rows | `416` |
| risky zero-span fallbacks | `3` / `416` (`0.72%`) |
| under-10 base-capped rows | `24` |

Span counts:

| span kind | count |
|---|---:|
| `source_specific_anchor` | `2462` |
| `source_backed_shared_key_term` | `998` |
| `domain_object_term` | `483` |
| `polarity_caveat` | `151` |
| `payoff_verb` | `90` |
| `math_result_shape` | `36` |
| `scope_qualifier` | `15` |

All span offsets matched the assistant summary text. The title-only-token check
and copied-title-trigram check passed.

## Training result

Run root:
`hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_token_weighted_gold_v2/lr4e6_steps128`

Training completed on MPS from the promoted q5/q8 sidecar adapter.

| field | value |
|---|---:|
| train rows | `1594` |
| max steps | `128` |
| labeled tokens | `26096` |
| weighted tokens over 1.0 | `5428` |
| weight sum | `18551.68` |
| final train loss | `7.8711` |

Checkpoints `32`, `64`, `96`, and `128` were converted to llama.cpp GGUF
adapters and evaluated.

## Llama eval

Fresh20:

| label | avg words | max | >14 | >15 | <10 | digit rows | title overlap >=3 | changed | regressions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| promoted baseline | `12.85` | `21` | `4` | `3` | `1` | `0` | `16` | `0` | `0` |
| checkpoint `32` | `12.50` | `21` | `4` | `3` | `2` | `2` | `17` | `10` | `5` |
| checkpoint `64` | `11.95` | `17` | `2` | `2` | `4` | `1` | `16` | `13` | `2` |
| checkpoint `96` | `12.10` | `17` | `2` | `2` | `3` | `1` | `14` | `13` | `2` |
| checkpoint `128` | `12.30` | `17` | `3` | `3` | `3` | `1` | `15` | `13` | `3` |

Held100:

| label | avg words | max | >14 | >15 | <10 | digit rows | title overlap >=3 | changed | regressions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| promoted baseline | `12.03` | `26` | `9` | `7` | `16` | `22` | `64` | `0` | `0` |
| checkpoint `32` | `11.72` | `26` | `10` | `8` | `20` | `22` | `61` | `62` | `6` |
| checkpoint `64` | `11.50` | `26` | `9` | `5` | `22` | `24` | `57` | `73` | `9` |
| checkpoint `96` | `11.32` | `26` | `7` | `5` | `24` | `25` | `57` | `78` | `10` |
| checkpoint `128` | `11.30` | `26` | `7` | `5` | `25` | `25` | `57` | `79` | `10` |

Checkpoint `32` was the least risky blind-review candidate: it changed more than
half of held100 while keeping digit rows equal to baseline. Later checkpoints
made stronger style movement, but increased short rows, digit rows, and
regression flags.

## Manual read

The specificity policy did preserve some concrete objects better than v1, but
it still gave the old q5 sidecar too much room to invent or garble details.

Fresh20 severe issues:

- `2307.11855`: checkpoint `32` introduced formula-like runtime text.
- `2512.15135`: checkpoints introduced the unsupported exact value `1`.
- `2601.13411`: later checkpoints produced severe false-negative black-hole
  wording.
- `2501.12034`: checkpoint `32` produced `detect and detect`.

Held100 examples:

- `2603.17185`: `icy` became `icey`.
- `2504.10224`: checkpoint `32` introduced dense exact frequency values.
- `2402.07728`: checkpoint `32` changed a 2D cardiac model to 3D.
- `2405.09895`: checkpoint `32` introduced an exact attribute count.
- `2503.03916`: checkpoint `32` erased the central functor/pushout/anima
  detail.
- `2106.14869`: checkpoint `32` replaced the genus algorithm with bogus
  `K3-h minor nodes`.

## Blind feed-preference review

Run root:
`hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_token_weighted_gold_v2/lr4e6_steps128/eval/blind_ab_v2_ckpt32_vs_baseline_feed_preference_100_20260527`

Review collection was clean: `100/100` rows reviewed, no missing IDs, no
duplicates, and no schema errors. Shard 01 ran through the subagent path; after
the subagent thread limit was reached, shards 02-05 were reviewed with isolated
`gpt-5.4-mini` Codex reviewer processes using the same prompt and shard files.

Unblinded result:

| choice | count | share |
|---|---:|---:|
| promoted baseline | `33` | `33%` |
| specificity v2 checkpoint `32` | `26` | `26%` |
| tie | `38` | `38%` |
| neither | `3` | `3%` |

## Verdict

This v2 run is not promotable. It did not beat the promoted baseline in blind
feed preference, and checkpoint `32` also failed the strict success mechanics:
`<10` rows rose by `+4`, `>14` and `>15` each rose by `+1`, and manual read
still found repeated groundedness failures.

The useful diagnosis is that specificity-preserving spans do move the old q5
sidecar and can preserve some concrete detail, but deterministic per-token
weighting is not enough to safely reshape this basin from 994 gold rows. The
next serious attempt should treat this as a data-mix/basin problem, not just a
span-policy problem.

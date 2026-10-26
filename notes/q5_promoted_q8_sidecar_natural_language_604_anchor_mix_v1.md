# q5 Promoted Q8 Sidecar Natural-Language 604 Anchor Mix v1

## Goal

Move one LR step up from the best anchored `4e-6` diagnostic while using the
full strict-good natural-language pool available at the time of the run.

This is a diagnostic run, not a promotion candidate by default.

## Dataset

- Builder: `scripts/build_q5_natural_language_anchor_mix_v1_dataset.py`
- Output: `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_604_anchor_mix_v1/dataset_604nl_1000faithanchors_w025/train.jsonl`
- Rows: `1604`
- Natural-language rows: `604` at weight `1.0`
- Faithfulness anchors: `1000` at weight `0.25`
- Natural-language source: `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_604_expansion_pool/training_rows.jsonl`
- Prompt mode: `title_abstract`

## Training Plan

- Config: `configs/sft/q5_promoted_q8_sidecar_natural_language_604_anchor_mix_v1_lr5e6_steps160.yaml`
- Start adapter: `hf_bucket/experiments/sft/q5_promoted_q8_adapter_sidecar_probe_v1`
- Output: `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_604_anchor_mix_v1/lr5e6_steps160`
- LR: `5e-6`
- Steps: `160`
- Save steps: `32`
- Effective batch: `4`
- Weighted loss normalization: `label_count`

## Read Target

Compare against the previous best anchored point, `4e-6` checkpoint `90`, which
had the best fresh20 llama.cpp surface read but was not promotable. This run
tests whether the extra strict-good rows make `5e-6` less brittle than it was in
the 500-row anchor-mix sweep.

## Result

Training completed cleanly.

- Output: `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_604_anchor_mix_v1/lr5e6_steps160`
- Train rows: `1604`
- Labeled assistant tokens: `27358`
- Effective weighted token mass: `13858.75`
- Train loss: `5.050`
- Runtime: `1745.2s`
- Epoch fraction: `0.399`
- Saved checkpoints: `32`, `64`, `96`, `128`, `160`

## Llama Fresh20 Eval

Eval root:
`hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_604_anchor_mix_v1/lr5e6_steps160/eval/llama_fresh20_natural_language_prompt_outputs`

| label | avg words | max | >14 | >15 | <10 | digit rows | title overlap >=3 | changed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| promoted baseline | `12.85` | `17` | `6` | `4` | `3` | `2` | `14` | `0` |
| checkpoint `32` | `12.25` | `17` | `4` | `3` | `2` | `1` | `13` | `11` |
| checkpoint `64` | `11.70` | `17` | `2` | `2` | `3` | `1` | `12` | `12` |
| checkpoint `96` | `12.05` | `18` | `3` | `3` | `2` | `1` | `12` | `14` |
| checkpoint `128` | `11.95` | `17` | `3` | `3` | `2` | `1` | `12` | `13` |
| checkpoint `160` | `12.00` | `17` | `3` | `3` | `2` | `1` | `13` | `11` |

Best mechanical llama point: checkpoint `64`. It improves the old `5e-6`
500-row anchor result by avoiding the 21-word blow-up, and it roughly matches
the old `4e-6` checkpoint `90` on length and digit counts.

## PEFT Fresh20 Eval

Eval root:
`hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_604_anchor_mix_v1/lr5e6_steps160/eval/peft_fresh20`

| label | avg words | max | >14 | >15 | <10 | digit rows | title overlap >=3 | changed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| promoted baseline | `12.05` | `16` | `3` | `2` | `2` | `0` | `13` | `0` |
| checkpoint `32` | `11.65` | `16` | `2` | `1` | `1` | `1` | `14` | `11` |
| checkpoint `64` | `11.95` | `16` | `2` | `1` | `1` | `0` | `14` | `14` |
| checkpoint `96` | `11.90` | `16` | `2` | `1` | `1` | `0` | `14` | `15` |
| checkpoint `128` | `12.30` | `17` | `3` | `2` | `0` | `0` | `15` | `15` |
| checkpoint `160` | `12.45` | `17` | `3` | `2` | `0` | `0` | `14` | `14` |

PEFT is cleaner mechanically than llama, but llama remains the deployment
surface.

## Manual Read

The extra strict-good rows help the surface shape at `5e-6`, but they do not
make this a promotion candidate.

Good movement:

- Checkpoint `64` reduces llama over-14 rows from `6` to `2`.
- Checkpoint `64` reduces digit rows from `2` to `1`.
- The dense OneMax formula is replaced with a simpler phrase.
- Some acronym-heavy or long rows shorten without the 21-word blow-up seen in
  the 500-row `5e-6` run.

Still bad:

- `2402.03953` drifts into Bitcoin futures wording in both runtimes.
- `2601.13411` remains broken in llama as a false gravitational-wave negative.
- `2212.01114` keeps the old llama `model model` repetition.
- `2311.05580` grows into dense exponential-cone wording after checkpoint `96`.
- `2501.12034` keeps the suspicious low-level/NoC specificity creep.

Verdict: checkpoint `64` is the best diagnostic point. The 604-row pool makes a
`5e-6` anchored run less mechanically hot than the earlier 500-row version, but
the core faithfulness failures remain. No promotion from this pass.

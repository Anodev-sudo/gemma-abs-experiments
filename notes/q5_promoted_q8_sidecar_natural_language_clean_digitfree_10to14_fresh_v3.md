# q5 Promoted q8 Sidecar: Clean Digit-Free Fresh v3

## Intent

Midpoint-LR run for the clean digit-free fresh branch.

Fresh v1 at `2e-6` moved enough but had too much quiet semantic damage. Fresh
v2 at `1e-6` was calmer but under-moved. This run probes the middle.

## Constraints

- No new generated data.
- No labels written by Codex.
- Existing labels only.
- No mined silver/bronze rows.
- No token spans.

## Dataset

- Dataset:
  `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin32_stabilizer_v4/dataset_clean_digitfree_10to14_v4/train.jsonl`
- Rows:
  `1649`

## Training

- Start adapter:
  `hf_bucket/experiments/sft/q5_promoted_q8_adapter_sidecar_probe_v1`
- Config:
  `configs/sft/q5_promoted_q8_sidecar_natural_language_clean_digitfree_10to14_fresh_v3_lr1p5e6_steps128.yaml`
- Output:
  `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_clean_digitfree_10to14_fresh_v3/lr1p5e6_steps128`
- LR:
  `1.5e-6`
- Steps:
  `128`
- Save:
  every `32` steps.

## Eval Plan

- fresh20 llama.cpp sweep against promoted baseline with v5 prompt.
- held100 llama.cpp sweep against promoted baseline with v5 prompt.
- Compare against:
  - fresh v1 checkpoint 64.
  - fresh v2 checkpoint 96.
  - burn-in32 stabilizer v2 checkpoint 96.

## Verdict

Checkpoint `64` is the selected contender for blind review.

This is not promotion-grade by itself, but it is the best tradeoff found in this
loop:

- Starts from the promoted q5/q8 sidecar, not burn-in32.
- Uses only the existing clean digit-free 10-14-word dataset.
- Moves enough to be worth comparing, while avoiding the worse semantic damage
  seen in the `2e-6` fresh run and the burn-in32 stabilizer family.

Held100, v5 prompt:

| checkpoint | avg words | max | >14 | >15 | <10 | digit rows | title overlap >=3 | changed | regressions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| promoted baseline | 12.03 | 26 | 9 | 7 | 16 | 22 | 64 | 0 | 0 |
| checkpoint 64 | 11.68 | 26 | 7 | 6 | 17 | 20 | 64 | 48 | 2 |
| checkpoint 96 | 11.62 | 26 | 7 | 6 | 18 | 20 | 64 | 55 | 3 |

Fresh20, v5 prompt:

| checkpoint | avg words | max | >14 | >15 | <10 | digit rows | title overlap >=3 | changed | regressions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| promoted baseline | 12.85 | 21 | 4 | 3 | 1 | 0 | 16 | 0 | 0 |
| checkpoint 64 | 12.35 | 21 | 4 | 3 | 3 | 1 | 15 | 8 | 1 |
| checkpoint 96 | 12.05 | 21 | 3 | 2 | 4 | 1 | 15 | 9 | 1 |

Manual read:

- Checkpoint `64` keeps the clean-run benefits with fewer severe rewrites than
  checkpoint `96`.
- Main known bad rows:
  - `2305.05450`: grows to 15 words.
  - `2402.01883`: glass sentence becomes semantically suspect.
  - Fresh20 sticky `2512.15135` still flips into exact `1`.
- Checkpoint `96` moves more, but adds a worse SGX rewrite and a new exact
  number on `2405.09895`, so it is not the selected checkpoint.

The v6 number-cautious prompt probe was not useful. It changed promoted baseline
behavior and did not solve the sticky numeric failure, so the contender remains
the v5 prompt plus checkpoint `64`.

## Blind Review Gate

Checkpoint `64` did not beat the promoted q5/q8 sidecar baseline in the
100-row blind feed-preference review.

- Review root:
  `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_clean_digitfree_10to14_fresh_v3/lr1p5e6_steps128/eval/blind_ab_ckpt64_vs_promoted_feed_preference_100_20260527`
- Reviewer model:
  `gpt-5.4-mini` subagents, five 20-row shards.
- Review mode:
  preference only, asking which summary is better for a broad mobile paper feed
  where the reader is deciding whether the paper is worth opening.
- Validation:
  `100/100` reviewed, `0` missing, `0` duplicates, `0` schema errors.
- Overall result:
  `fresh_v3_checkpoint64 = 18`, `promoted_baseline_q5_q8_sidecar = 25`,
  `tie = 52`, `neither = 5`.
- Changed-row result:
  `48` rows differed; among those, `fresh_v3_checkpoint64 = 18`,
  `promoted_baseline_q5_q8_sidecar = 25`, `tie = 2`, `neither = 3`.

Conclusion: keep the promoted candidate unchanged. Checkpoint `64` is useful as
diagnostic evidence, but not a promotion contender.

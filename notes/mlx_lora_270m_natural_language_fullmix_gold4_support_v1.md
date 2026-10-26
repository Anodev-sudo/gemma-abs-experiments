# MLX full finetune 270M natural-language fullmix gold4 support v1

## Goal

Test a fresh full-mode run where strict gold examples dominate the training
exposure, but lower-quality existing support data is mixed in once for broader
coverage.

This is not a promotion run.

## Dataset Plan

Dataset builder:

- `scripts/build_mlx_fullmix_gold_support_v1_dataset.py`

Output dataset:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_fullmix_gold4_support_v1/dataset_title_abstract_feed_blurb_v5`

Gold source:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_1000_milestone/mlx_lora/title_abstract_chat_gemma_270m_it_abstract_to_feed_blurb_v5`

Support sources:

- `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_specificity_support_v1/dataset_clean1648_broadsupport_v1/train.jsonl`
- `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_guarded_support_v1/dataset_clean1648_under14support482_numericguard165_v1/train.jsonl`

Weighting method:

- MLX full mode does not use repo `loss_weight_hints`.
- Gold train rows are repeated `4x`.
- Deduped support rows appear once.
- Validation and test splits are pure held-out gold rows.

## Training Plan

Config:

- `configs/mlx_lora/gemma_3_270m_it_natural_language_fullmix_gold4_support_v1_full_lr5e7.yaml`

Output:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_fullmix_gold4_support_v1/full_lr5e7`

Training details:

- base model: `google/gemma-3-270m-it`
- fine-tune type: `full`
- `num_layers: -1`
- batch size: `4`
- iterations: `600`
- learning rate: `5e-7`
- save every `60`

## Evaluation Plan

First inspect MLX generations on the standard fresh packet and a gold held-out
slice. If a checkpoint looks viable, then consider conversion/quantized runtime
testing.

## Result

Training completed cleanly. MLX full mode with `num_layers: -1` reported
`100.325M / 268.098M` trainable parameters (`37.421%`), so this is a large
full-mode update over selected model layers rather than every parameter.

Validation loss improved steadily from `5.315` at iteration `1` to `2.983` at
iteration `600`; test loss was `3.051` (`21.129` perplexity).

Fresh-40 generation checks against the stronger title+abstract v5 LoRA
checkpoint `425` were not competitive:

- checkpoint `120`: average `13.53` words, `17/40` over 14 words, `13/40`
  under 10 words, `6/40` digit rows, `35/40` title-overlap rows.
- checkpoint `420`: average `13.57` words, `16/40` over 14 words, `13/40`
  under 10 words, `6/40` digit rows, `31/40` title-overlap rows.
- checkpoint `600`: average `12.88` words, `17/40` over 14 words, `14/40`
  under 10 words, `6/40` digit rows, `32/40` title-overlap rows.

Manual read showed old-style failures returning: title restatement, abstract
compression phrasing, formula spillover, and openings like `The paper...`.
Likely cause: the broad support mix was too permissive. Most support came from
`base_10746_low_weight` and faithfulness anchors, which widened coverage but
reintroduced the behavior the natural-language gold rows were meant to replace.

Conclusion: v1 is not a contender. A narrower v2 should keep gold even heavier
and exclude broad-base and faithfulness-overlay support, using only repair,
recenter, numeric-guard, specificity, and q5-correction style support rows.

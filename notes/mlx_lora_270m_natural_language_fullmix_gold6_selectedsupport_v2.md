# MLX full finetune 270M natural-language fullmix gold6 selected-support v2

## Goal

Test whether a full-mode run can use lower-quality existing data without
letting that data overpower the strict gold feed-blurb style.

This is not a promotion run.

## Change From v1

v1 mixed the `994` strict gold release repeated `4x` with all deduped support
rows from the q5 support datasets. It trained cleanly but generation quality was
not competitive: high title overlap, more digit/formula rows, and old abstract
phrasing returned.

v2 keeps the same source family but changes only the mix:

- gold train rows repeat `6x`
- selected support appears once
- exclude `broad_support::base_10746_low_weight`
- exclude `faithfulness_anchor_clean`
- exclude `faithfulness_overlay_1400_v2`

## Dataset

Dataset builder:

- `scripts/build_mlx_fullmix_gold_support_v1_dataset.py`

Build command:

```bash
hf_bucket/experiments/mlx_lora/runtime/py312/bin/python \
  scripts/build_mlx_fullmix_gold_support_v1_dataset.py \
  --gold-repeat 6 \
  --output-dir hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_fullmix_gold6_selectedsupport_v2/dataset_title_abstract_feed_blurb_v5 \
  --exclude-component broad_support::base_10746_low_weight \
  --exclude-component faithfulness_anchor_clean \
  --exclude-source-split faithfulness_overlay_1400_v2
```

Output dataset:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_fullmix_gold6_selectedsupport_v2/dataset_title_abstract_feed_blurb_v5`

Counts:

- train examples: `5987`
- train unique paper IDs: `1517`
- valid examples: `50`
- test examples: `50`
- gold train exposures: `5364`
- support rows: `623`

Largest selected support components:

- `183` `broad_support::recenter_broad_overlay`
- `138` `numeric_guard::numeric_antagonist_clean_digit_free_targeted_weight`
- `66` `broad_support::step2400_mined_kimi_qwen_10000`
- `26` `broad_support::hard77_preserve`
- `22` `under14_highsupport::step2400_mined_kimi_qwen_10000`
- `20` `under14_highsupport::specificity_preserving_400`

## Training Plan

Config:

- `configs/mlx_lora/gemma_3_270m_it_natural_language_fullmix_gold6_selectedsupport_v2_full_lr5e7.yaml`

Output:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_fullmix_gold6_selectedsupport_v2/full_lr5e7`

Training details:

- base model: `google/gemma-3-270m-it`
- fine-tune type: `full`
- `num_layers: -1`
- batch size: `4`
- iterations: `420`
- learning rate: `5e-7`
- save every `60`

## Evaluation Plan

Evaluate against the title+abstract v5 LoRA checkpoint `425` on the standard
fresh-40 packet. Inspect checkpoint `240` first, then later checkpoints only if
mechanics and manual read are not obviously worse.

## Result

Training completed cleanly. MLX full mode again reported `100.325M / 268.098M`
trainable parameters (`37.421%`).

Validation loss improved steadily:

- iteration `1`: `5.315`
- iteration `60`: `3.460`
- iteration `120`: `3.241`
- iteration `180`: `3.142`
- iteration `240`: `3.101`
- iteration `300`: `3.057`
- iteration `360`: `3.025`
- iteration `420`: `3.007`

Test loss was `3.076` (`21.672` perplexity).

Fresh-40 mechanics against title+abstract v5 LoRA checkpoint `425`:

| checkpoint | avg words | over 14 | over 15 | under 10 | digit rows | formula rows | title overlap >=3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA 425 | `12.15` | `7/40` | `5/40` | `5/40` | `3/40` | `1/40` | `15/40` |
| full v2 240 | `14.55` | `19/40` | `14/40` | `9/40` | `3/40` | `3/40` | `28/40` |
| full v2 360 | `14.03` | `20/40` | `15/40` | `10/40` | `5/40` | `3/40` | `28/40` |
| full v2 420 | `14.32` | `19/40` | `15/40` | `11/40` | `5/40` | `3/40` | `29/40` |

Manual read still showed old-style failures:

- filler/abstract openings, such as `The study explores...`
- too much title restatement and product-name copying
- formula-like math spillover
- occasional multi-sentence output
- brittle generic claims replacing the concrete reason to open the paper

Conclusion: v2 is also not a contender. The narrower support mix reduced the
most obvious v1 data problem, but full-mode training still moves toward title
copying and abstract compression more than the LoRA path does.

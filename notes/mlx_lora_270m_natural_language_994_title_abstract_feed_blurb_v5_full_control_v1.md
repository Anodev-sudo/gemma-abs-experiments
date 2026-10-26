# MLX full finetune 270M 994 title+abstract feed-blurb v5 control

## Goal

Run a fresh full-mode control on only the strict `994` gold title+abstract
feed-blurb dataset. This isolates whether the fullmix failures are caused by
lower-quality support rows or by the full-mode update path itself.

This is not a promotion run.

## Dataset

Dataset:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_1000_milestone/mlx_lora/title_abstract_chat_gemma_270m_it_abstract_to_feed_blurb_v5`

Counts:

- train examples: `894`
- valid examples: `50`
- test examples: `50`

## Training Plan

Config:

- `configs/mlx_lora/gemma_3_270m_it_natural_language_994_title_abstract_feed_blurb_v5_full_lr5e7_control_v1.yaml`

Output:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_title_abstract_feed_blurb_v5_full_control_v1/full_lr5e7`

Training details:

- base model: `google/gemma-3-270m-it`
- fine-tune type: `full`
- `num_layers: -1`
- batch size: `4`
- iterations: `420`
- learning rate: `5e-7`
- save every `60`

## Evaluation Plan

Evaluate checkpoints `240`, `360`, and `420` against the title+abstract v5 LoRA
checkpoint `425` on the standard fresh-40 packet.

## Result

Training completed cleanly. MLX full mode reported `100.325M / 268.098M`
trainable parameters (`37.421%`).

Validation loss improved steadily:

- iteration `1`: `5.315`
- iteration `60`: `3.445`
- iteration `120`: `3.221`
- iteration `180`: `3.122`
- iteration `240`: `3.082`
- iteration `300`: `3.041`
- iteration `360`: `3.016`
- iteration `420`: `2.993`

Test loss was `3.062` (`21.369` perplexity).

Fresh-40 mechanics against title+abstract v5 LoRA checkpoint `425`:

| checkpoint | avg words | over 14 | over 15 | under 10 | digit rows | formula rows | title overlap >=3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA 425 | `12.15` | `7/40` | `5/40` | `5/40` | `3/40` | `1/40` | `15/40` |
| full control 240 | `14.60` | `19/40` | `13/40` | `10/40` | `5/40` | `3/40` | `30/40` |
| full control 360 | `14.53` | `20/40` | `15/40` | `11/40` | `5/40` | `3/40` | `29/40` |
| full control 420 | `14.82` | `21/40` | `16/40` | `10/40` | `4/40` | `3/40` | `31/40` |

Manual read showed the same failure pattern as the mixed runs:

- title-bound summaries
- abstract-compression phrasing
- occasional `The paper...` / `The study...` openings
- formula-like math spillover
- longer, less feed-like outputs despite lower held-out loss

Conclusion: the lower-quality support mix is not the main cause of the fullmix
failure. Pure gold full-mode training at this LR/step scale still underperforms
the LoRA checkpoint on the product-facing metrics.

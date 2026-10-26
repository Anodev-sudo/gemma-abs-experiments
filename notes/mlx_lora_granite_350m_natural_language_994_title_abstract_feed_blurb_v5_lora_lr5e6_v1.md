# MLX LoRA Granite 350M natural-language 994-row title+abstract feed-blurb v5 v1

## Goal

Run a lateral architecture check against the current Gemma 270M natural-language
effort: same strict gold rows, same title+abstract feed-blurb prompt, but a
fresh `ibm-granite/granite-4.0-350m` instruct base.

This is not a promotion run. The question is whether Granite learns the desired
natural-language feed shape more cleanly than Gemma 270M on the limited gold
set.

## Data

Source release:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_1000_milestone/training_rows.jsonl`

MLX dataset:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_1000_milestone/mlx_lora/title_abstract_chat_gemma_270m_it_abstract_to_feed_blurb_v5`

Prompt spec:

- `prompts/mlx_lora/gemma_270m_it_abstract_to_feed_blurb_v5.json`

Split:

- train: `894`
- valid: `50`
- test: `50`

## Training

Config:

- `configs/mlx_lora/granite_4_0_350m_natural_language_994_title_abstract_feed_blurb_v5_lora_lr5e6_v1.yaml`

Command:

```bash
HF_HOME=hf_bucket/experiments/mlx_lora/.hf-home \
hf_bucket/experiments/mlx_lora/runtime/py312/bin/python \
  -m mlx_lm lora \
  --config configs/mlx_lora/granite_4_0_350m_natural_language_994_title_abstract_feed_blurb_v5_lora_lr5e6_v1.yaml \
  --mask-prompt
```

Adapter output:

- `hf_bucket/experiments/mlx_lora/granite_4_0_350m_natural_language_994_title_abstract_feed_blurb_v5_lora_lr5e6_v1/adapters`

Settings:

- base model: `ibm-granite/granite-4.0-350m`
- fine-tune type: `lora`
- LoRA target modules: attention q/k/v/o projections
- rank: `16`
- scale: `10.0`
- batch size: `4`
- iterations: `500`
- learning rate: `5e-6`
- save/eval interval: `25`
- max sequence length: `1024`

## Matching Gemma Eval Pattern

First read the training curve:

- validation loss every `25` steps
- final test loss/perplexity
- do not choose a checkpoint from loss alone

Then repeat the Gemma MLX natural-language checks:

1. Generate a fixed fresh holdout comparison with
   `scripts/sample_mlx_holdout_generations.py`.
2. Summarize length, digit, formula, and title-overlap mechanics with
   `scripts/summarize_mlx_generation_mechanics.py`.
3. Manually read the most plausible checkpoint pair before any broader review.
4. If a checkpoint looks alive, run a 100-row blind feed-preference packet using
   the same prompt contract and excluding train rows.

Useful prior Gemma anchors:

- `notes/mlx_lora_270m_natural_language_994_title_abstract_feed_blurb_v5_lora_lr1e5_long_v1.md`
- `notes/q5_promoted_q8_sidecar_natural_language_994_token_weighted_gold_v1.md`

Example post-train MLX comparison command, after choosing checkpoints:

```bash
HF_HOME=hf_bucket/experiments/mlx_lora/.hf-home \
hf_bucket/experiments/mlx_lora/runtime/py312/bin/python \
  scripts/sample_mlx_holdout_generations.py \
  --release-dir hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_1000_milestone \
  --prompt-spec prompts/mlx_lora/gemma_270m_it_abstract_to_feed_blurb_v5.json \
  --left-label granite-base \
  --right-label granite-ckptXXX \
  --right-adapter-dir hf_bucket/experiments/mlx_lora/granite_4_0_350m_natural_language_994_title_abstract_feed_blurb_v5_lora_lr5e6_v1/adapters \
  --right-checkpoint hf_bucket/experiments/mlx_lora/granite_4_0_350m_natural_language_994_title_abstract_feed_blurb_v5_lora_lr5e6_v1/adapters/0000XXX_adapters.safetensors \
  --model-path ibm-granite/granite-4.0-350m \
  --input-mode title_abstract \
  --sample-size 40 \
  --seed 20260527 \
  --max-tokens 36 \
  --output-dir hf_bucket/experiments/mlx_lora/granite_4_0_350m_natural_language_994_title_abstract_feed_blurb_v5_lora_lr5e6_v1/review_samples/base_vs_ckptXXX_fresh40_title_abstract_feed_blurb_v5
```

Example mechanics summary:

```bash
hf_bucket/experiments/mlx_lora/runtime/py312/bin/python \
  scripts/summarize_mlx_generation_mechanics.py \
  --generations hf_bucket/experiments/mlx_lora/granite_4_0_350m_natural_language_994_title_abstract_feed_blurb_v5_lora_lr5e6_v1/review_samples/base_vs_ckptXXX_fresh40_title_abstract_feed_blurb_v5/generations.jsonl \
  --output-json hf_bucket/experiments/mlx_lora/granite_4_0_350m_natural_language_994_title_abstract_feed_blurb_v5_lora_lr5e6_v1/review_samples/base_vs_ckptXXX_fresh40_title_abstract_feed_blurb_v5/mechanics.json \
  --output-md hf_bucket/experiments/mlx_lora/granite_4_0_350m_natural_language_994_title_abstract_feed_blurb_v5_lora_lr5e6_v1/review_samples/base_vs_ckptXXX_fresh40_title_abstract_feed_blurb_v5/MECHANICS.md \
  --variant-outputs hf_bucket/experiments/mlx_lora/granite_4_0_350m_natural_language_994_title_abstract_feed_blurb_v5_lora_lr5e6_v1/review_samples/base_vs_ckptXXX_fresh40_title_abstract_feed_blurb_v5/variant_outputs.jsonl
```

## Result

Training completed cleanly.

Training details:

- base model downloaded into repo-local HF cache
- trainable params: `2.982M / 352.380M`
- peak memory: about `8.565 GB`
- run artifact size: about `240M`
- base model cache size after pull: about `681M`
- final test loss: `1.860`
- final test perplexity: `6.424`

Validation curve:

| step | valid loss |
| ---: | ---: |
| 1 | 3.848 |
| 25 | 1.858 |
| 50 | 1.791 |
| 75 | 1.706 |
| 100 | 1.687 |
| 125 | 1.672 |
| 150 | 1.657 |
| 175 | 1.646 |
| 200 | 1.646 |
| 225 | 1.627 |
| 250 | 1.619 |
| 275 | 1.617 |
| 300 | 1.618 |
| 325 | 1.626 |
| 350 | 1.606 |
| 375 | 1.628 |
| 400 | 1.623 |
| 425 | 1.604 |
| 450 | 1.608 |
| 475 | 1.628 |
| 500 | 1.624 |

Best validation checkpoint:

- `0000425_adapters.safetensors`

First manual-read candidates:

- `0000425_adapters.safetensors` as the best validation checkpoint
- `0000350_adapters.safetensors` as the previous strong validation checkpoint
- optionally `0000275_adapters.safetensors` if later checkpoints look too
  adapted or verbose in generation

## Fresh40 Heuristic Check

Generated artifacts:

- `hf_bucket/experiments/mlx_lora/granite_4_0_350m_natural_language_994_title_abstract_feed_blurb_v5_lora_lr5e6_v1/review_samples/base_vs_ckpt425_fresh40_title_abstract_feed_blurb_v5`
- `hf_bucket/experiments/mlx_lora/granite_4_0_350m_natural_language_994_title_abstract_feed_blurb_v5_lora_lr5e6_v1/review_samples/ckpt350_vs_ckpt425_fresh40_title_abstract_feed_blurb_v5`

Base vs checkpoint `425`:

| label | avg words | max | <=14 | >14 | >15 | <10 | digit rows | formula rows | title overlap >=3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| granite-base | `19.38` | `34` | `11 / 40` | `29 / 40` | `27 / 40` | `3 / 40` | `6 / 40` | `6 / 40` | `36 / 40` |
| granite-ckpt425 | `12.28` | `18` | `31 / 40` | `9 / 40` | `6 / 40` | `8 / 40` | `6 / 40` | `2 / 40` | `22 / 40` |

Checkpoint `350` vs `425` on the same rows:

| label | avg words | max | <=14 | >14 | >15 | <10 | digit rows | formula rows | title overlap >=3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| granite-ckpt350 | `12.30` | `22` | `32 / 40` | `8 / 40` | `6 / 40` | `4 / 40` | `6 / 40` | `2 / 40` | `24 / 40` |
| granite-ckpt425 | `12.28` | `18` | `31 / 40` | `9 / 40` | `6 / 40` | `8 / 40` | `6 / 40` | `2 / 40` | `22 / 40` |

Read:

- The adapter moves the base model substantially: shorter outputs, fewer
  over-limit rows, fewer formula-ish rows, and much less title copying.
- Checkpoint `425` is not mechanically dominant over `350`. It has a shorter
  max tail and slightly less title overlap, but more under-10 and over-14 rows.
- A simple surface badness count favored `425` on `5` changed rows, `350` on
  `7`, tied on `18`, with `10` identical outputs.
- Manual read should compare `350` and `425`, not just read `425`.

## Fresh40 Comparison With Gemma 270M v5

Same-row comparison artifact:

- `hf_bucket/experiments/mlx_lora/granite_4_0_350m_natural_language_994_title_abstract_feed_blurb_v5_lora_lr5e6_v1/review_samples/gemma270m_ckpt425_vs_granite_ckpt425_fresh40_title_abstract_feed_blurb_v5`

This uses the same 40 held-out papers as the Granite heuristic check, with the
same title+abstract feed-blurb v5 prompt and `max_tokens=36`.

| label | avg words | max | <=14 | >14 | >15 | <10 | digit rows | formula rows | title overlap >=3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gemma270m-v5-ckpt425 | `12.70` | `20` | `28 / 40` | `12 / 40` | `8 / 40` | `5 / 40` | `7 / 40` | `2 / 40` | `22 / 40` |
| granite350m-v5-ckpt350 | `12.30` | `22` | `32 / 40` | `8 / 40` | `6 / 40` | `4 / 40` | `6 / 40` | `2 / 40` | `24 / 40` |
| granite350m-v5-ckpt425 | `12.28` | `18` | `31 / 40` | `9 / 40` | `6 / 40` | `8 / 40` | `6 / 40` | `2 / 40` | `22 / 40` |

Simple surface badness on Gemma `425` vs Granite `425` favored Granite on `16`
rows, Gemma on `13`, and tied on `11`. There were no identical outputs.

Read:

- On this exact fresh40 packet, Granite is mechanically competitive with the
  comparable fresh Gemma 270M v5 checkpoint and slightly cleaner on length tail.
- Granite `350` looks like the stronger mechanical Granite candidate, while
  Granite `425` keeps the shortest max tail.
- The earlier Gemma v5 note used a different fresh40 packet where Gemma `425`
  looked cleaner (`33 / 40` under 14, `7 / 40` over 14, `15 / 40` title overlap
  >=3), so do not call Granite a clear mechanical win yet.
- Next step should be a manual same-row read against Gemma `425`, with Granite
  `350` and `425` both represented.

## Blind100 Model-Judge Comparison With Gemma 270M v5

Blind review roots:

- `hf_bucket/experiments/mlx_lora/granite_4_0_350m_natural_language_994_title_abstract_feed_blurb_v5_lora_lr5e6_v1/review_samples/blind_review_gemma425_vs_granite350_100_20260528`
- `hf_bucket/experiments/mlx_lora/granite_4_0_350m_natural_language_994_title_abstract_feed_blurb_v5_lora_lr5e6_v1/review_samples/blind_review_gemma425_vs_granite425_100_20260528`

These use the same 100 held-out papers, the same title+abstract feed-blurb v5
prompt, randomized A/B sides, and hidden model labels. The blind judge was
`qwen/qwen3-235b-a22b-2507` through `scripts/run_q5_blind_ab_model_review_v1.py`;
results were unblinded with `scripts/summarize_q5_blind_ab_reviews_v1.py`.

| comparison | reviewed | missing | schema errors | preferred | more accurate |
| --- | ---: | ---: | ---: | --- | --- |
| Gemma `425` vs Granite `350` | `100` | `0` | `0` | Granite `72`, Gemma `27`, tie `1` | Granite `75`, Gemma `24`, tie `1` |
| Gemma `425` vs Granite `425` | `100` | `0` | `0` | Granite `74`, Gemma `26`, tie `0` | Granite `76`, Gemma `24`, tie `0` |

Read:

- Both Granite checkpoints beat the comparable fresh Gemma `425` checkpoint
  decisively in this blind model-judge packet.
- Granite `425` edges Granite `350` by the model-judge count, while the earlier
  mechanics check had Granite `350` slightly cleaner on length compliance.
- For manual read, include both Granite checkpoints. If only one candidate gets
  promoted to deeper review, `425` has the strongest combined signal so far:
  best validation loss, shortest max tail, and the best blind100 count.

## Blind100 Model-Judge Comparison With Current q5 Candidate

Blind review root:

- `hf_bucket/experiments/mlx_lora/granite_4_0_350m_natural_language_994_title_abstract_feed_blurb_v5_lora_lr5e6_v1/review_samples/blind_review_q5_current_vs_granite425_100_20260528`

Generation artifacts:

- `hf_bucket/experiments/mlx_lora/granite_4_0_350m_natural_language_994_title_abstract_feed_blurb_v5_lora_lr5e6_v1/review_samples/q5_current_vs_granite425_blind100_title_abstract_feed_blurb_v5`

This reuses the same 100 held-out papers as the Gemma-vs-Granite blind packet.
The q5 comparator is the current llama.cpp pointer
`q5_base_plus_promoted_q8_adapter_sidecar`: Q5_K_M base plus the promoted q8
adapter converted to an f16 GGUF sidecar. The Granite output is MLX checkpoint
`425`. Both used the same title+abstract feed-blurb v5 prompt and `max_tokens=36`.
The blind judge was `qwen/qwen3-235b-a22b-2507`.

Blind result:

| comparison | reviewed | missing | schema errors | preferred | more accurate |
| --- | ---: | ---: | ---: | --- | --- |
| q5 current vs Granite `425` | `100` | `0` | `0` | Granite `58`, q5 `41`, neither `1` | Granite `57`, q5 `42`, neither `1` |

Runtime mechanics on the same 100 rows:

| label | avg words | median | max | <=14 | >14 | >15 | <10 | digits | formula | title overlap >=3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q5 current sidecar | `12.88` | `13.0` | `19` | `73 / 100` | `27 / 100` | `17 / 100` | `10 / 100` | `14 / 100` | `2 / 100` | `64 / 100` |
| Granite `425` | `13.09` | `12.5` | `24` | `71 / 100` | `29 / 100` | `18 / 100` | `5 / 100` | `5 / 100` | `4 / 100` | `56 / 100` |

Read:

- Granite `425` beats the current q5 candidate in this blind model-judge packet,
  but the margin is much narrower than the Gemma `425` comparison.
- q5 is slightly cleaner on length tail, while Granite has fewer digit rows and
  less title overlap. The judge still preferred Granite overall.
- Treat this as encouraging but not a deployment decision. It is cross-runtime
  (`llama.cpp` q5 sidecar vs MLX Granite), and the q5 candidate carries a much
  larger old-data lineage than this 994-row Granite LoRA.

## Original Checkpoint Sweep

After the failed strict-support replay branch, the original 994-gold Granite run
was swept more carefully. The goal was to check whether checkpoint `425` had
been chosen too quickly.

Sweep artifacts:

- `hf_bucket/experiments/mlx_lora/granite_4_0_350m_natural_language_994_title_abstract_feed_blurb_v5_lora_lr5e6_v1/review_samples/granite425_vs_ckpt225_blind100_title_abstract_feed_blurb_v5`
- same pattern for checkpoints `250`, `275`, `300`, `325`, `350`, `375`, `400`,
  `450`, `475`, and `500`

All rows use the same 100-paper packet as the q5-current vs Granite `425` blind
comparison. Checkpoint `425` is the left/reference output in each pair.

| checkpoint | val loss | avg words | max | <=14 | >14 | >15 | <10 | digits | formula | title >=3 | changed vs 425 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 225 | `1.627` | `13.26` | `28` | `68` | `32` | `20` | `13` | `5` | `5` | `56` | `87` |
| 250 | `1.619` | `12.59` | `24` | `75` | `25` | `17` | `15` | `6` | `4` | `53` | `86` |
| 275 | `1.617` | `12.90` | `22` | `72` | `28` | `19` | `9` | `8` | `5` | `56` | `84` |
| 300 | `1.618` | `12.81` | `24` | `74` | `26` | `15` | `7` | `12` | `3` | `53` | `89` |
| 325 | `1.626` | `13.22` | `20` | `70` | `30` | `20` | `7` | `7` | `3` | `51` | `80` |
| 350 | `1.606` | `12.89` | `24` | `74` | `26` | `19` | `9` | `7` | `3` | `50` | `78` |
| 375 | `1.628` | `13.31` | `24` | `66` | `34` | `23` | `6` | `8` | `4` | `48` | `83` |
| 400 | `1.623` | `12.25` | `24` | `79` | `21` | `14` | `19` | `4` | `4` | `53` | `84` |
| 425 | `1.604` | `13.09` | `24` | `71` | `29` | `18` | `5` | `5` | `4` | `56` | `0` |
| 450 | `1.608` | `13.74` | `24` | `63` | `37` | `28` | `7` | `9` | `4` | `59` | `66` |
| 475 | `1.628` | `12.22` | `21` | `82` | `18` | `11` | `16` | `4` | `4` | `56` | `80` |
| 500 | `1.624` | `12.25` | `20` | `81` | `19` | `13` | `12` | `5` | `4` | `58` | `74` |

Read:

- `425` is still the safest checkpoint. It is not mechanically best, but it has
  the best validation loss and the best qualitative balance of completeness,
  caution, and feed readability.
- `350` is the only serious challenger. It has lower title overlap and sometimes
  sharper technical phrasing, but it is also more likely to add unsupported or
  over-specific details. Examples included HADES/PANDA details, "without
  fine-tuning", and a bad "10 billion years old" astronomy line.
- `475` and `500` are mechanically tempting because they reduce the length tail,
  but they read like over-compression: more under-10 rows, more dropped nuance,
  and more brittle specificity. They sometimes replace a useful broad hook with
  a narrow method or exact metric.
- `400` has similar over-compression problems, with `19/100` under-10 rows.
- `450` is worse on almost every surface metric and should be ignored.
- `225-300` are still unstable: they move a lot but have too many digit rows,
  under-short rows, or overlong rows.

Decision: keep `425` as the primary Granite checkpoint. If a direct Granite-only
blind review is ever worth spending on, compare `425` vs `350`; do not spend it
on the late checkpoints.

## GGUF Q6/Q5 Runtime Probe

Checkpoint `425` was converted through:

1. MLX LoRA checkpoint `0000425_adapters.safetensors`
2. PEFT adapter
3. merged Hugging Face checkpoint
4. F16 GGUF
5. Q6_K and Q5_K_M GGUF

Artifacts:

- run root:
  `hf_bucket/experiments/llama_cpp/granite_4_0_350m_v5_ckpt425_gguf_q6_q5_v1`
- Q6:
  `hf_bucket/experiments/llama_cpp/granite_4_0_350m_v5_ckpt425_gguf_q6_q5_v1/gguf/granite_4_0_350m_v5_ckpt425.Q6_K.gguf`
- Q5:
  `hf_bucket/experiments/llama_cpp/granite_4_0_350m_v5_ckpt425_gguf_q6_q5_v1/gguf/granite_4_0_350m_v5_ckpt425.Q5_K_M.gguf`

File sizes:

| artifact | size |
| --- | ---: |
| F16 GGUF | `676M` |
| Q6_K GGUF | `279M` |
| Q5_K_M GGUF | `252M` |

Runtime caveat:

- `llama.cpp` Metal offload is not usable for Granite 4.0 350M in this local
  setup. Both the locally merged GGUF and IBM's official Q6_K GGUF emitted only
  repeated `@` characters with GPU layers enabled.
- CPU-only mode (`--n-gpu-layers 0`) works. All measurements below are CPU-only.

100-row CPU mechanics on the same Granite `425` blind packet:

| label | rows | avg | median | max | <=14 | >14 | >15 | <10 | digits | formula | title >=3 | changed vs MLX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Granite `425` MLX | `100` | `13.09` | `12.5` | `24` | `71` | `29` | `18` | `5` | `5` | `4` | `56` | n/a |
| Granite `425` GGUF Q6_K CPU | `100` | `13.12` | `13.0` | `19` | `70` | `30` | `21` | `10` | `3` | `4` | `58` | `87` |
| Granite `425` GGUF Q5_K_M CPU | `100` | `13.13` | `13.0` | `24` | `73` | `27` | `15` | `4` | `4` | `4` | `58` | `96` |

Q6 and Q5 were exact matches on only `5/100` rows. CPU speed was slow under the
two-parallel-server run: about `16.2s` average wall time per row for both Q6 and
Q5, with max rows around `74-84s`.

Read:

- Q5_K_M is mechanically a little better than Q6_K: fewer `>14`, fewer `>15`,
  and an under-10 count close to MLX.
- Q6_K preserves some risky polarity/scope cases better. Example: destructive
  equality resolution stayed "incomplete" in Q6, while Q5 flipped it to
  "complete."
- Q5_K_M often stays closer to the MLX line on concrete physics/math examples
  where Q6 invents detail, such as the Brusselator row and the meson-cloud row.
- Both quantized CPU runtimes move a lot relative to MLX. Treat this as a real
  runtime-shift surface, not just harmless quantization noise.
- Neither Q6_K nor Q5_K_M should be considered promotion-ready without a blind
  review and a targeted sticky-row read. If we continue, Q5_K_M is the better
  first blind-review candidate mechanically, but it needs a polarity/scope
  tripwire because the observed "complete" vs "incomplete" flip is severe.

## MLX Swift Q8 Export Probe

Checkpoint `425` was exported through an MLX-native q8 path:

1. Convert `ibm-granite/granite-4.0-350m` to MLX q8 group size 64.
2. Copy `0000425_adapters.safetensors` into an isolated adapter directory.
3. Fuse the q8 base plus ckpt425 sidecar into a standalone q8 MLX model.
4. Load the fused q8 model with the repo `mlx-summary-probe` Release executable.

Artifacts:

- run root:
  `hf_bucket/experiments/mlx_lora/granite_4_0_350m_v5_ckpt425_mlx_swift_q8_export_v1`
- Swift-facing fused q8 model:
  `hf_bucket/experiments/mlx_lora/granite_4_0_350m_v5_ckpt425_mlx_swift_q8_export_v1/fused_q8_g64`
- manifest:
  `hf_bucket/experiments/mlx_lora/granite_4_0_350m_v5_ckpt425_mlx_swift_q8_export_v1/export_manifest.json`

File sizes:

| artifact | size |
| --- | ---: |
| isolated ckpt425 adapter | `11M` |
| base q8 MLX | `364M` |
| fused q8 MLX | `364M` |

100-row mechanics on the same Granite `425` blind packet:

| label | rows | avg | median | max | <=14 | >14 | >15 | <10 | digits | formula | title >=3 | changed vs original MLX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Granite `425` original MLX | `100` | `13.09` | `12.5` | `24` | `71` | `29` | `18` | `5` | `5` | `4` | `56` | n/a |
| Granite `425` q8 sidecar, Python | `100` | `12.73` | `12.0` | `24` | `76` | `24` | `14` | `6` | `7` | `4` | `54` | `42` |
| Granite `425` q8 fused, Python | `100` | `12.64` | `12.0` | `21` | `75` | `25` | `12` | `8` | `5` | `2` | `57` | `52` |
| Granite `425` q8 fused, Swift | `100` | `12.73` | `12.0` | `21` | `77` | `23` | `15` | `5` | `6` | `2` | `53` | `54` |

Exact-match ladder:

- Original MLX vs q8 sidecar Python: `58/100` exact, `42/100` changed.
- q8 sidecar Python vs q8 fused Python: `56/100` exact, `44/100` changed.
- q8 fused Python vs q8 fused Swift: `77/100` exact, `23/100` changed.
- Original MLX vs q8 fused Swift: `46/100` exact, `54/100` changed.

Read:

- MLX Swift can load and run the Granite q8 fused artifact normally.
- Swift is not the largest source of drift in this ladder. The q8 sidecar
  quantization and q8 fusion steps already move behavior before Swift sees it.
- The q8 fused path reproduces the bad Superposition Calculus polarity failure:
  "incomplete" becomes "complete" in both Python fused and Swift fused output.
- Swift added a smaller extra shift over Python fused, but it preserved several
  important Python fused failures exactly, so the deployment artifact itself is
  the main concern.
- The macOS Swift probe was fast: `100` rows completed with a mean generation
  time around `115 ms` after a `228 ms` load.

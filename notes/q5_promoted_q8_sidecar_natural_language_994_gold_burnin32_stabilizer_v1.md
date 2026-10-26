# q5 promoted sidecar natural-language burn-in32 stabilizer v1

## Intent

Second-stage stabilizer from the 994-gold burn-in checkpoint `32`.

The burn-in stage moved the sidecar hard, but checkpoint `32` already showed
short-output drift and object/detail regressions. This run tries to keep the
feed-blurb movement while recovering faithfulness with a low-LR replay mix.

## Planned run

- Dataset builder:
  `scripts/build_q5_natural_language_burnin32_stabilizer_v1_dataset.py`
- Config:
  `configs/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin32_stabilizer_v1_lr1e6_steps128.yaml`
- Dataset root:
  `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin32_stabilizer_v1/dataset_994gold_1000faithanchors_stabilizer_v1`
- Output root:
  `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin32_stabilizer_v1/lr1e6_steps128`
- Start adapter:
  `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin_v1/lr6e6_steps192/checkpoint-32`
- LR: `1e-6`
- Steps: `128`
- Save steps: `32`
- Effective batch: `4`

## Dataset policy

- Strict gold rows: `994`
- Faithfulness anchors: `1000`
- Prompt: `gemma_270m_it_abstract_to_feed_blurb_v5`
- Input mode: `title_abstract`
- No token spans.
- No silver, bronze, raw 10k, DPO, or burn-in self-label rows.
- Exclude fresh20 and held100 paper IDs from anchor selection.

Weights:

- strict gold default: `0.65`
- strict gold title-overlap-risk: `0.50`
- strict gold under-10 cap: `0.40`
- faithfulness anchors: `0.35`

This is intentionally a stabilizer, not a stronger style pass.

## Actual dataset

Built:
`hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin32_stabilizer_v1/dataset_994gold_1000faithanchors_stabilizer_v1/train.jsonl`

Validation passed:

| field | value |
|---|---:|
| total rows | `1994` |
| strict gold rows | `994` |
| faithfulness anchors | `1000` |
| duplicate paper IDs | `0` |
| excluded eval IDs | `120` |
| span hints | `0` |

Weights:

| base weight | rows |
|---|---:|
| `0.65` | `558` |
| `0.50` | `406` |
| `0.40` | `30` |
| `0.35` | `1000` |

## Training result

Run root:
`hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin32_stabilizer_v1/lr1e6_steps128`

Training completed cleanly from burn-in checkpoint `32`.

| field | value |
|---|---:|
| steps | `128` |
| epoch fraction | `0.2568` |
| labeled tokens | `33288` |
| weighted tokens over 1.0 | `0` |
| weight sum | `15198.8` |
| train runtime | `724.9s` |
| train loss | `4.4779` |

Checkpoints `32`, `64`, `96`, and `128` were converted to llama.cpp GGUF
adapters and evaluated.

## Llama eval

Fresh20:

| label | avg words | max | >14 | >15 | <10 | digit rows | title overlap >=3 | changed | regressions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| promoted baseline | `12.85` | `21` | `4` | `3` | `1` | `0` | `16` | `0` | `0` |
| checkpoint `32` | `11.85` | `17` | `2` | `2` | `4` | `1` | `15` | `13` | `2` |
| checkpoint `64` | `11.75` | `17` | `2` | `2` | `5` | `1` | `15` | `13` | `2` |
| checkpoint `96` | `11.90` | `17` | `2` | `2` | `4` | `1` | `15` | `13` | `2` |
| checkpoint `128` | `11.90` | `17` | `2` | `2` | `4` | `1` | `15` | `13` | `2` |

Held100:

| label | avg words | max | >14 | >15 | <10 | digit rows | title overlap >=3 | changed | regressions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| promoted baseline | `12.03` | `26` | `9` | `7` | `16` | `22` | `64` | `0` | `0` |
| checkpoint `32` | `11.32` | `24` | `6` | `4` | `21` | `24` | `58` | `83` | `9` |
| checkpoint `64` | `11.57` | `24` | `10` | `6` | `21` | `23` | `58` | `83` | `12` |
| checkpoint `96` | `11.54` | `24` | `9` | `5` | `20` | `23` | `59` | `83` | `12` |
| checkpoint `128` | `11.47` | `24` | `9` | `5` | `22` | `24` | `58` | `83` | `13` |

## Readout

Checkpoint `32` is the best stabilizer point. It nudges burn-in checkpoint `32`
in the right mechanical direction on held100:

| metric | burn-in `32` | stabilizer `32` |
|---|---:|---:|
| avg words | `11.39` | `11.32` |
| max | `24` | `24` |
| >14 | `7` | `6` |
| >15 | `5` | `4` |
| <10 | `22` | `21` |
| digit rows | `23` | `24` |
| title overlap >=3 | `59` | `58` |
| changed vs promoted baseline | `84` | `83` |
| regression flags | `10` | `9` |

But the pass barely moves the actual text. Compared directly to burn-in
checkpoint `32`, stabilizer checkpoint `32` changes only `9/100` held100 rows.
Some changes are useful:

- `2412.10382`: restores the baseline-like VGG/ResNet/AST latency framing.
- `2309.12138`: moves endotrivial-complex wording toward the Picard-group
  object.
- `2203.01973`: replaces awkward geometric-doubling wording with hyperbolic
  polyhedra.

Some changes are neutral or bad:

- `2603.17185`: shortens the icy-ocean error, but keeps the `icey` typo.
- `2505.14565`: changes a DeFi total-value-locked line into a worse exact-value
  claim.
- `2401.01738`: weakens `single parameterization` to `single parameter`.

The sticky failures remain:

- `2503.03916`: still invents `14-point arithmetic`.
- `2512.15135`: still produces the unsupported exact value `1` on fresh20.
- `2601.13411`: still produces the black-hole/gravitational-wave false-negative
  pattern on fresh20.

## Verdict

This stabilizer is directionally useful but too weak. It modestly improves the
burn-in checkpoint `32` mechanics without preserving enough central objects or
removing the severe sticky failures. I would not run a blind preference review
for this checkpoint yet.

If continuing this branch, the next stabilizer should apply more force or more
targeted force: either higher anchor weight / lower gold continuation weight, or
a small targeted repair set for the known sticky failures layered over the
faithfulness anchors.

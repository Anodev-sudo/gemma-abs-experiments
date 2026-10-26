# q5 promoted sidecar natural-language 994-gold burn-in v1

## Intent

This is a different approach from the token-weighted v1/v2 diagnostics: start
with a gold-only curriculum burn-in from the promoted q5/q8 sidecar before
adding anchors, silver, bronze, DPO, or old raw data.

The hypothesis is that the sidecar may need a clean product-shape stage first.
The 994 strict gold rows define the target feed-blurb behavior; later stages can
then add lower-weight replay or broader data without fighting the initial shape
as much.

## Planned run

- Dataset builder:
  `scripts/build_q5_natural_language_gold_burnin_v1_dataset.py`
- Config:
  `configs/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin_v1_lr6e6_steps192.yaml`
- Dataset root:
  `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin_v1/dataset_994gold_burnin_v1`
- Output root:
  `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin_v1/lr6e6_steps192`
- Start adapter:
  `hf_bucket/experiments/sft/q5_promoted_q8_adapter_sidecar_probe_v1`
- LR: `6e-6`
- Steps: `192`
- Save steps: `32`
- Effective batch: `4`

## Dataset policy

- Source: strict 994-row gold milestone.
- Prompt: `gemma_270m_it_abstract_to_feed_blurb_v5`.
- Input mode: `title_abstract`.
- No anchors.
- No silver/bronze/raw rows.
- No per-token spans.
- Row-level weights only:
  - default gold: `1.0`
  - title-overlap-risk: `0.90`
  - under-10 summaries: capped at `0.85`

The under-10 cap is kept because the previous v2 diagnostic moved the model but
also increased short outputs on the held100 packet.

## Initial eval plan

After training, sweep checkpoints `32`, `64`, `96`, `128`, `160`, and `192`
against the existing fresh20 packet and the held100 packet. Pick an early/mid
checkpoint for manual read only if it shows meaningful movement without worse
length/digit mechanics than the promoted baseline.

## Actual dataset

Built:
`hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin_v1/dataset_994gold_burnin_v1/train.jsonl`

| field | value |
|---|---:|
| rows | `994` |
| unique paper IDs | `994` |
| title-overlap-risk rows | `416` |
| under-10 capped rows | `30` |
| spans | `0` |

Base weights:

| base weight | rows |
|---|---:|
| `1.0` | `558` |
| `0.9` | `406` |
| `0.85` | `30` |

## Training result

Run root:
`hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin_v1/lr6e6_steps192`

Training completed cleanly from the promoted q5/q8 sidecar.

| field | value |
|---|---:|
| steps | `192` |
| epoch fraction | `0.7726` |
| labeled tokens | `15302` |
| weighted tokens over 1.0 | `0` |
| weight sum | `14613.4` |
| train runtime | `2020.5s` |
| train loss | `10.5152` |

Checkpoints `32`, `64`, `96`, `128`, `160`, and `192` were converted to
llama.cpp GGUF adapters and evaluated.

## Llama eval

Fresh20:

| label | avg words | max | >14 | >15 | <10 | digit rows | title overlap >=3 | changed | regressions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| promoted baseline | `12.85` | `21` | `4` | `3` | `1` | `0` | `16` | `0` | `0` |
| checkpoint `32` | `12.20` | `17` | `3` | `3` | `3` | `1` | `16` | `14` | `3` |
| checkpoint `64` | `12.15` | `17` | `2` | `2` | `2` | `1` | `15` | `17` | `2` |
| checkpoint `96` | `12.10` | `18` | `2` | `2` | `2` | `1` | `13` | `17` | `2` |
| checkpoint `128` | `11.85` | `17` | `3` | `1` | `2` | `1` | `13` | `18` | `3` |
| checkpoint `160` | `12.40` | `20` | `3` | `2` | `2` | `1` | `14` | `19` | `4` |
| checkpoint `192` | `12.40` | `20` | `3` | `2` | `2` | `1` | `14` | `19` | `4` |

Held100:

| label | avg words | max | >14 | >15 | <10 | digit rows | title overlap >=3 | changed | regressions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| promoted baseline | `12.03` | `26` | `9` | `7` | `16` | `22` | `64` | `0` | `0` |
| checkpoint `32` | `11.39` | `24` | `7` | `5` | `22` | `23` | `59` | `84` | `10` |
| checkpoint `64` | `11.34` | `24` | `8` | `6` | `26` | `22` | `52` | `91` | `11` |
| checkpoint `96` | `11.35` | `24` | `9` | `5` | `26` | `26` | `50` | `95` | `15` |
| checkpoint `128` | `11.25` | `24` | `8` | `4` | `24` | `25` | `53` | `96` | `14` |
| checkpoint `160` | `11.40` | `24` | `12` | `5` | `24` | `24` | `51` | `96` | `17` |
| checkpoint `192` | `11.33` | `24` | `11` | `5` | `25` | `24` | `49` | `96` | `16` |

## Readout

The burn-in moved the sidecar much harder than the token-weighted v2 run. On
held100, even checkpoint `32` changed `84/100` rows, versus `62/100` for the v2
checkpoint `32`.

The standalone burn-in is not good enough. It improves some mechanical tails
and reduces title overlap, but it also increases short rows and introduces more
regression flags. Examples from the held100 metric readout:

- `2503.03916`: functor/pushout math becomes unsupported `14-point arithmetic`.
- `2603.17185`: icy-ocean wording becomes awkward and partly distorted.
- `2412.10382`: exact latency numbers mutate into a different exact claim.
- `2504.10224`: optical camera communication gains unsupported `1000x zoom`.
- `2310.10867`: radiotherapy auto-contouring flips toward a generic data-need
  claim.

## Verdict

The burn-in idea is useful as a stage, not as a candidate. If continuing this
branch, checkpoint `32` is the least-bad continuation base: it has strong
movement, lower max length, fewer over-limit rows, and only `+1` digit row, but
it already has `+6` under-10 rows and `10` regression flags on held100.

The next phase should be a stabilizer pass from checkpoint `32`, not more
gold-only burn-in. I would use low LR plus anchors or a mixed replay set to
recover faithfulness and central objects while preserving the feed-blurb
movement.

# q5 promoted sidecar natural-language 500 no-anchor creep v2

Purpose: creep forward from the too-cold `1e-6` no-anchor run toward the
too-hot `1e-5` aggressive run, looking for the first llama.cpp checkpoint with
real natural-language movement before obvious semantic damage.

Starting adapter:

```text
hf_bucket/experiments/sft/q5_promoted_q8_adapter_sidecar_probe_v1
```

Dataset:

```text
hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_500_aggressive_v1/dataset_title_abstract_500/train.jsonl
```

First bracket:

```text
configs/sft/q5_promoted_q8_sidecar_natural_language_500_lowlr_noanchors_v2_lr3e6_steps50.yaml
```

Run:

```bash
hf_bucket/experiments/dpo/runtime/py312/bin/python \
  scripts/run_weighted_sft_gemma.py \
  --config configs/sft/q5_promoted_q8_sidecar_natural_language_500_lowlr_noanchors_v2_lr3e6_steps50.yaml
```

Stop condition: if llama.cpp still barely moves, continue upward. If llama.cpp
moves but repeats the `1e-5` drift failures, stop and treat no-anchor SFT as
unpromising at this data size.

## Result

Two brackets were run:

| bracket | runtime | train loss | epoch | checkpoints |
| --- | ---: | ---: | ---: | --- |
| `2e-6` | 79.7s | 11.78 | 0.4 | 10, 20, 30, 40, 50 |
| `3e-6` | 81.4s | 11.70 | 0.4 | 10, 20, 30, 40, 50 |

### llama.cpp Fresh20 Shape

Baseline promoted q5 sidecar on the same sample:

```text
avg words: 12.85
over 14 words: 6/20
digit rows: 2/20
```

`2e-6`:

| checkpoint | avg words | max | over 14 | digit rows | changed vs baseline |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 12.80 | 17 | 6 | 2 | 1 |
| 20 | 12.70 | 17 | 6 | 2 | 4 |
| 30 | 12.55 | 17 | 5 | 1 | 6 |
| 40 | 12.55 | 17 | 5 | 1 | 6 |
| 50 | 12.35 | 17 | 5 | 1 | 6 |

`3e-6`:

| checkpoint | avg words | max | over 14 | digit rows | changed vs baseline |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 12.80 | 17 | 6 | 2 | 2 |
| 20 | 12.35 | 17 | 5 | 1 | 7 |
| 30 | 12.30 | 17 | 3 | 1 | 8 |
| 40 | 12.55 | 21 | 3 | 1 | 9 |
| 50 | 12.55 | 21 | 3 | 1 | 9 |

### Read

`3e-6` is the first bracket with real llama.cpp movement. Checkpoint `30`
looks like the most informative point: over-limit rows drop from `6/20` to
`3/20`, digit rows drop from `2/20` to `1/20`, and `8/20` outputs change.

But it is not clean enough to promote. It improves or simplifies some visible
surface failures while introducing or preserving bad semantic substitutions:

- Good-ish movement:
  - `2507.07642`: removes some wordiness around coordinate systems.
  - `2307.11855`: removes the dense formula string from llama output.
  - `2408.06739`: later `3e-6` checkpoints shorten the ANOVA/ASCA row.
- Bad movement:
  - `2307.11855`: formula string becomes unsupported "time-dependent
    optimization".
  - `2601.13411`: checkpoint `30+` worsens into "gravitational waves are not
    found in first-principle black hole models".
  - `2501.12034`: shifts from broad manycore attack detection to a suspicious
    "low-level platform" phrasing.
  - `2601.01876`: checkpoint `40+` becomes long and awkward.

Decision: this found the no-anchor movement threshold, not a candidate. `1e-6`
was too cold, `2e-6` is marginal, `3e-6` is real but already semantically
fragile, and `1e-5` was too hot. The best no-anchor checkpoint from this bracket
is `3e-6` checkpoint `30`, but it should be treated as diagnostic only.

Next likely move: keep the `3e-6` neighborhood, but add targeted anchors or a
tiny hard-negative repair set for the exact rows where no-anchor drift appears.
Pure no-anchor SFT has a very narrow useful window on 500 rows.

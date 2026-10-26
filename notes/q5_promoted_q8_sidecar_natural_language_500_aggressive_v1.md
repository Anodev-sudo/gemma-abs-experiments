# q5 promoted sidecar natural-language 500 aggressive diagnostic

Purpose: test whether the current promoted compact q5 runtime can be pushed
toward the new natural-language 500-row target with a strong SFT pass.

This is not a promotion candidate by default. It starts from the promoted q5
llama.cpp candidate's PEFT adapter and trains against all 500 milestone rows
using the natural-language title+abstract prompt.

Starting adapter:

```text
hf_bucket/experiments/sft/q5_promoted_q8_adapter_sidecar_probe_v1
```

Dataset:

```text
hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_500_aggressive_v1/dataset_title_abstract_500/train.jsonl
```

Config:

```text
configs/sft/q5_promoted_q8_sidecar_natural_language_500_aggressive_v1_lr1e5.yaml
```

Training command:

```bash
hf_bucket/experiments/dpo/runtime/py312/bin/python \
  scripts/run_weighted_sft_gemma.py \
  --config configs/sft/q5_promoted_q8_sidecar_natural_language_500_aggressive_v1_lr1e5.yaml
```

Planned checks:

- Generate from checkpoints in Transformers/PEFT, the trained runtime.
- Convert the same checkpoints to GGUF LoRA sidecars and generate with
  llama.cpp against the promoted Q5_K_M base.
- Compare shape, length, jargon reduction, semantic drift, and runtime mismatch
  on a held-out 20-row sample with zero overlap against the 500 training rows.

## Result

Training completed for `150` optimizer steps, saving checkpoints at `50`,
`100`, and `150`.

```text
output: hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_500_aggressive_v1/lr1e5_steps150
train rows: 500
assistant-label tokens: 7,736
effective batch size: 4
learning rate: 1.0e-5
final train loss: 10.75
runtime: 910.2 seconds
epoch: 1.2
```

The Q5_K_M GGUF base had been pruned from local disk, so the llama.cpp pass
restored only that file from the bucket manifest before running the sweep.

### Shape Metrics

Fresh20 PEFT/Transformers runtime:

| label | avg words | max | over 14 | digit rows | forbidden openings | title overlap >= 3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| promoted baseline | 12.05 | 16 | 3 | 0 | 0 | 13 |
| checkpoint 50 | 11.40 | 16 | 2 | 2 | 0 | 10 |
| checkpoint 100 | 11.90 | 16 | 1 | 1 | 0 | 10 |
| checkpoint 150 | 11.55 | 16 | 1 | 1 | 0 | 9 |

Fresh20 llama.cpp runtime:

| label | avg words | max | over 14 | digit rows | forbidden openings | title overlap >= 3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| promoted baseline | 12.85 | 17 | 6 | 2 | 0 | 14 |
| checkpoint 50 | 12.40 | 16 | 2 | 2 | 0 | 12 |
| checkpoint 100 | 12.65 | 17 | 4 | 2 | 0 | 13 |
| checkpoint 150 | 12.55 | 17 | 3 | 2 | 0 | 13 |

### Read

The aggressive pass moved the surface style: shorter outputs, fewer over-limit
rows, and somewhat less title overlap. It did not improve the candidate overall.
The manual read found too many new faithfulness failures:

- `2402.03953`: trained checkpoints drift into Bitcoin futures / news framing.
- `2507.07642`: checkpoint 50 invents `100,000 nanometers`; llama keeps `2D`
  and `3D` shorthand.
- `2505.00797`: later checkpoints invent or overstate black-hole/gravitational
  wave claims.
- `2502.18508`: checkpoint 150 flips into guiding backdoor attacks; llama
  checkpoint 50 says the defense reduces model utility.
- `2601.13411`: all variants are poor; training does not fix the bad
  gravitational-wave interpretation.

Best practical interpretation: `500` rows can pressure this old promoted
sidecar toward the natural-language shape, but the old basin does not become a
trustworthy natural-language summarizer from this direct aggressive SFT pass.
If we keep using this lineage, the next run should be lower pressure plus
explicit anti-drift anchors, or a preference/repair set based on the failures
above. For the fresh natural-language direction, the cleaner adapter line still
looks more honest than pushing this sidecar.

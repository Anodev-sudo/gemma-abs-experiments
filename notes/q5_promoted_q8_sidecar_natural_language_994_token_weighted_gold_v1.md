# q5 Promoted Q8 Sidecar Natural-Language 994 Token-Weighted Gold v1

## Goal

Run one serious diagnostic against the promoted q5/q8 sidecar using the 994
strict gold natural-language rows with deterministic assistant-token weighting.

This is not a promotion run. The question is whether token-weighted gold can
move the old sidecar basin more cleanly than the earlier low-weight anchor
mixes.

## Dataset

- Builder: `scripts/build_q5_natural_language_token_weighted_gold_v1_dataset.py`
- Output: `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_token_weighted_gold_v1/dataset_994gold_600faithanchors_tokenweighted_v1/train.jsonl`
- Rows: `1594`
- Strict gold rows: `994`
- Faithfulness anchors: `600` at base weight `0.12`
- Prompt: `gemma_270m_it_abstract_to_feed_blurb_v5` / `title_abstract`
- Audit: `weighted_span_audit.jsonl`
- Manual audit sample: `weighted_span_audit_sample40.md`

Final build validation:

- title-risk gold rows: `416`
- risky zero-span fallback rows: `15`
- risky zero-span fallback fraction: `3.61%`
- span offset check: passed
- title-only upweight check: passed
- span kind counts:
  - `non_title_source_anchor`: `2559`
  - `polarity_caveat`: `151`
  - `payoff_verb`: `103`
  - `math_result_shape`: `29`

The 40-row audit sample caught and fixed two policy issues before training:
connective words were leaking into non-title anchors, and generic CS
`algorithm(s)` tokens were being treated as theory-result terms too broadly.

## Training

- Config: `configs/sft/q5_promoted_q8_sidecar_natural_language_994_token_weighted_gold_v1_lr4e6_steps160.yaml`
- Start adapter: `hf_bucket/experiments/sft/q5_promoted_q8_adapter_sidecar_probe_v1`
- Output: `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_token_weighted_gold_v1/lr4e6_steps160`
- LR: `4e-6`
- Steps: `160`
- Save steps: `32`
- Effective batch: `4`
- Weighted loss normalization: `label_count`

Training completed cleanly.

- Labeled assistant tokens: `26056`
- Tokens above base weight: `3284`
- Weighted token mass: `16664.03`
- Train runtime: `1222.1s`
- Train loss: `7.565`
- Epoch fraction: `0.402`
- Saved checkpoints: `32`, `64`, `96`, `128`, `160`

## Llama Fresh20 Eval

Eval used the same fresh20 packet as previous q5-sidecar natural-language probes,
but with the v5 feed-blurb prompt for all candidates.

Metrics:
`hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_token_weighted_gold_v1/lr4e6_steps160/eval/FRESH20_FEED_BLURB_V5_METRICS.md`

| label | avg words | max | >14 | >15 | <10 | digit rows | title overlap >=3 | changed | regressions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| promoted baseline | `12.85` | `21` | `4` | `3` | `1` | `0` | `16` | `0` | `0` |
| token-weighted ckpt `32` | `12.60` | `21` | `5` | `3` | `3` | `1` | `15` | `8` | `3` |
| token-weighted ckpt `64` | `12.35` | `21` | `4` | `3` | `2` | `1` | `16` | `12` | `3` |
| token-weighted ckpt `96` | `11.65` | `17` | `1` | `1` | `2` | `1` | `14` | `15` | `1` |
| token-weighted ckpt `128` | `11.45` | `17` | `2` | `2` | `4` | `1` | `13` | `16` | `2` |
| token-weighted ckpt `160` | `11.45` | `17` | `2` | `2` | `4` | `1` | `13` | `15` | `2` |
| prior 500-anchor `4e-6` ckpt `90` | `12.80` | `21` | `5` | `3` | `2` | `1` | `15` | `9` | `3` |

Best mechanical point: checkpoint `96`.

Checkpoint `96` beats the prior 500-anchor `4e-6` checkpoint `90` mechanically
under the v5 prompt: shorter max tail, fewer over-limit rows, more changed rows,
and fewer flagged regressions.

## Manual Read

Good movement:

- `2407.18489`: becomes a simpler and likely faithful compute-cost blurb:
  `Distributed baseband processing cuts computation time for MIMO detection.`
- `2601.01876`: cuts a 21-word Galois-theory overrun down to a simpler sentence.
- `2408.06739`: removes the ANOVA/ASCA acronym pile.
- `2501.12034`: later checkpoints shorten the manycore-security wording.
- `2402.01673`: checkpoint `96` is less overlong than the baseline, though still
  somewhat generic.

Still bad:

- `2601.13411`: checkpoint `96` produces a severe false negative:
  `The gravitational waves are not found in the first-principle black hole models.`
- `2402.03953`: drifts from decentralized exchange behavior into generic Bitcoin
  futures wording.
- `2512.15135`: introduces an unsupported-looking exact value:
  `The maximal correlation coefficient for discrete random variables is 1.`
- `2311.05580`: checkpoint `96` restores dense technical phrasing around
  exponential-cone constraints and polynomial time.
- `2212.01114`: the old `model model` duplication is unchanged at every new
  checkpoint.
- `2505.16616`: checkpoints `128` and `160` introduce a new overlong comparative
  language-bias claim.

## Verdict

This run proves that deterministic token weighting can move the promoted sidecar
more strongly than the earlier anchored gold mix, and checkpoint `96` is the best
diagnostic point.

It is not promotable. The mechanical win is real, but the sticky faithfulness
failures remain large enough that this should feed the next training design
rather than become a candidate.

## 100-row blind feed-preference review

Run root:
`hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_token_weighted_gold_v1/lr4e6_steps160/eval/blind_ab_ckpt96_vs_baseline_feed_preference_100_20260527`

This used a fresh 100-row random recent-3y packet excluding the 1,594-row
token-weighted train set and the earlier fresh20 packet. Five `gpt-5.4-mini`
subagents reviewed 20 rows each with a preference-only app-feed question:
which blurb better helps a reader decide whether to open the paper?

Unblinded result:

| choice | count |
|---|---:|
| promoted baseline | `42` |
| token-weighted ckpt `96` | `36` |
| tie | `18` |
| neither | `4` |

Review collection was clean: `100/100` rows reviewed, no missing IDs, no
duplicates, and no schema errors.

Runtime mechanics on the same 100-row packet:

| label | avg words | max | >14 | >15 | <10 | digit rows |
|---|---:|---:|---:|---:|---:|---:|
| promoted baseline | `12.03` | `26` | `9` | `7` | `16` | `22` |
| token-weighted ckpt `96` | `11.36` | `26` | `6` | `5` | `23` | `21` |

Conclusion: checkpoint `96` is still mechanically cleaner, but the blind
feed-preference read did not beat the promoted baseline. Treat this as useful
diagnostic evidence, not a promotion case.

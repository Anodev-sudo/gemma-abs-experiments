# DPO Configs

These configs are for the local Gemma DPO pilot path.

What this path assumes:

- task stays `title + abstract -> one-sentence summary`
- preferences come from prepared hard-negative preference datasets under `hf_dataset/.../dpo/`
- the starting point is the converted `attn_1400` adapter
- training runs in the separate DPO runtime under `hf_bucket/experiments/dpo/runtime/py312/`
- the installed `trl` runtime clones the converted default adapter into an internal
  reference adapter automatically, so the runner does not need explicit
  `model_adapter_name` / `ref_adapter_name` config

Available dataset/config variants:

- `gemma_3_270m_it_title_abstract_attn_1400_dpo_hc100_v1.yaml`
  - the original `100`-pair release-backed pilot
- `gemma_3_270m_it_title_abstract_attn_1400_dpo_balanced200_v1.yaml`
  - the expanded `200`-pair mix
  - includes all seven current failure axes
  - uses both `release_training_row` and `generated_correction` chosen summaries
- `gemma_3_270m_it_title_abstract_attn_1400_dpo_balanced200_curated_v1.yaml`
  - the hand-curated `200`-pair refresh built from the original pool plus targeted gap-fill generation
  - keeps all seven axes at target after manual review
  - preserves curation metadata in the exported dataset for later analysis
- `gemma_3_270m_it_title_abstract_attn_1400_dpo_balanced200_curated_v2.yaml`
  - the corrected hand-curated `200`-pair refresh with preserved abstract text
  - use this instead of the earlier curated `v1` export

Typical setup order:

1. Prepare the original `hc100` preference dataset:

   ```bash
   python3 scripts/prepare_dpo_dataset.py
   ```

   Or prepare the expanded `balanced200` dataset:

   ```bash
   python3 scripts/prepare_dpo_dataset.py \
     --input-jsonl hf_bucket/experiments/hard_negative_pack_v1/correction_pack_25_per_axis_v1_20260420/hard_negative_pack.jsonl \
     --extra-input-jsonl hf_bucket/experiments/hard_negative_pack_v1/correction_pack_25_per_axis_v1_20260420/generated_negative_candidates.jsonl \
     --extra-input-jsonl hf_bucket/experiments/hard_negative_pack_v1/correction_pack_25_per_axis_v1_20260420/natural_negative_candidates.jsonl \
     --allowed-gold-sources release_training_row,generated_correction \
     --axis-target invented_number=31 \
     --axis-target wrong_number_binding=31 \
     --axis-target title_restatement=31 \
     --axis-target too_generic=30 \
     --axis-target reversed_polarity=22 \
     --axis-target side_detail_fixation=30 \
     --axis-target wrong_entity=25 \
     --output-dir hf_dataset/releases/recent_3y_balanced_5000_pilot_v2/dpo/title_abstract_preference_balanced200_v1
   ```

   Or prepare the curated `balanced200` refresh from the reviewed curation bundle:

   ```bash
   python3 scripts/prepare_dpo_dataset.py \
     --input-jsonl hf_bucket/experiments/dpo/curation/title_abstract_preference_balanced200_gapfill_pool_v1/curated_dataset/curated_candidates.jsonl \
     --allowed-gold-sources release_training_row,generated_correction \
     --axis-target invented_number=31 \
     --axis-target wrong_number_binding=31 \
     --axis-target title_restatement=31 \
     --axis-target too_generic=30 \
     --axis-target reversed_polarity=22 \
     --axis-target side_detail_fixation=30 \
     --axis-target wrong_entity=25 \
     --output-dir hf_dataset/releases/recent_3y_balanced_5000_pilot_v2/dpo/title_abstract_preference_balanced200_curated_v1 \
     --overwrite
   ```

   Current corrected curated export:

   ```bash
   python3 scripts/prepare_dpo_curation_bundle.py \
     --source-jsonl hf_bucket/experiments/hard_negative_pack_v1/correction_pack_25_per_axis_v1_20260420/hard_negative_pack.jsonl \
     --source-jsonl hf_bucket/experiments/hard_negative_pack_v1/correction_pack_25_per_axis_v1_20260420/generated_negative_candidates.jsonl \
     --source-jsonl hf_bucket/experiments/hard_negative_pack_v1/correction_pack_25_per_axis_v1_20260420/natural_negative_candidates.jsonl \
     --source-jsonl hf_bucket/experiments/hard_negative_pack_v1/dpo_gap_fill_50_v1_20260421/generated_negative_candidates.jsonl \
     --source-jsonl hf_bucket/experiments/hard_negative_pack_v1/dpo_gap_fill_50_v1_20260421/natural_negative_candidates.jsonl \
     --reference-dataset-dir hf_dataset/releases/recent_3y_balanced_5000_pilot_v2/dpo/title_abstract_preference_balanced200_v1 \
     --seed-bundle-dir hf_bucket/experiments/dpo/curation/title_abstract_preference_balanced200_gapfill_pool_v1 \
     --output-dir hf_bucket/experiments/dpo/curation/title_abstract_preference_balanced200_gapfill_pool_v2 \
     --overwrite

   python3 scripts/materialize_dpo_curation.py \
     --bundle-dir hf_bucket/experiments/dpo/curation/title_abstract_preference_balanced200_gapfill_pool_v2 \
     --output-dir hf_bucket/experiments/dpo/curation/title_abstract_preference_balanced200_gapfill_pool_v2/curated_dataset \
     --overwrite

   python3 scripts/prepare_dpo_dataset.py \
     --input-jsonl hf_bucket/experiments/dpo/curation/title_abstract_preference_balanced200_gapfill_pool_v2/curated_dataset/curated_candidates.jsonl \
     --allowed-gold-sources release_training_row,generated_correction \
     --axis-target invented_number=31 \
     --axis-target wrong_number_binding=31 \
     --axis-target title_restatement=31 \
     --axis-target too_generic=30 \
     --axis-target reversed_polarity=22 \
     --axis-target side_detail_fixation=30 \
     --axis-target wrong_entity=25 \
     --output-dir hf_dataset/releases/recent_3y_balanced_5000_pilot_v2/dpo/title_abstract_preference_balanced200_curated_v2 \
     --overwrite
   ```

2. Create the DPO runtime:

   ```bash
   hf_bucket/experiments/mlx_lora/runtime/py312/bin/python -m venv hf_bucket/experiments/dpo/runtime/py312
   hf_bucket/experiments/dpo/runtime/py312/bin/pip install --upgrade pip
   hf_bucket/experiments/dpo/runtime/py312/bin/pip install torch transformers trl peft datasets accelerate safetensors pyyaml sentencepiece protobuf
   ```

3. Convert the MLX adapter to PEFT and run the parity check:

   ```bash
   hf_bucket/experiments/dpo/runtime/py312/bin/python \
     scripts/convert_mlx_lora_to_peft.py \
     --run-parity-check
   ```

4. Run a small smoke before the full pilot:

   ```bash
   hf_bucket/experiments/dpo/runtime/py312/bin/python \
     scripts/run_dpo_gemma.py \
     --config configs/dpo/gemma_3_270m_it_title_abstract_attn_1400_dpo_hc100_v1.yaml \
     --output-dir hf_bucket/experiments/dpo/gemma_3_270m_it_title_abstract_dpo_hc100_v1_smoke \
     --max-steps-override 10
   ```

5. Run the real pilot:

   ```bash
   hf_bucket/experiments/dpo/runtime/py312/bin/python \
     scripts/run_dpo_gemma.py \
     --config configs/dpo/gemma_3_270m_it_title_abstract_attn_1400_dpo_hc100_v1.yaml
   ```

   Or run the expanded `balanced200` pilot:

   ```bash
   hf_bucket/experiments/dpo/runtime/py312/bin/python \
     scripts/run_dpo_gemma.py \
     --config configs/dpo/gemma_3_270m_it_title_abstract_attn_1400_dpo_balanced200_v1.yaml
   ```

   Or run the curated `balanced200` pilot:

   ```bash
   hf_bucket/experiments/dpo/runtime/py312/bin/python \
     scripts/run_dpo_gemma.py \
     --config configs/dpo/gemma_3_270m_it_title_abstract_attn_1400_dpo_balanced200_curated_v1.yaml
   ```

   Current corrected curated run:

   ```bash
   hf_bucket/experiments/dpo/runtime/py312/bin/python \
     scripts/run_dpo_gemma.py \
     --config configs/dpo/gemma_3_270m_it_title_abstract_attn_1400_dpo_balanced200_curated_v2.yaml
   ```

Post-train holdout comparison:

```bash
hf_bucket/experiments/dpo/runtime/py312/bin/python \
  scripts/sample_transformers_holdout_generations.py \
  --left-label attn_1400_converted \
  --right-label dpo_hc100 \
  --left-adapter-dir hf_bucket/experiments/dpo/gemma_3_270m_it_title_abstract_attn_1400_peft_v1 \
  --right-adapter-dir hf_bucket/experiments/dpo/gemma_3_270m_it_title_abstract_dpo_hc100_v1
```

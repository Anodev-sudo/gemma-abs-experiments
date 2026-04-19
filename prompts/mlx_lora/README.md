# MLX LoRA Training Prompts

This directory stores versioned prompt assets for turning reviewed summary rows into
MLX-ready supervised fine-tuning examples.

Current prompt spec:

- `gemma_270m_it_abstract_to_glanceable_summary_v1.json`
  - Intended model family: small instruction-tuned Gemma variants
  - Default dataset format: `chat`
  - Default student input: `abstract_only`
  - Goal: teach the model to produce one-sentence, feed-appropriate paper summaries
    without dragging title text, authors, or date metadata into the output

Design notes:

- `abstract_only` is the default because the current deployment plan still assumes
  the student should learn `abstract -> summary`.
- `title_abstract` is included as an explicit alternate path in case the deployment
  prompt later changes.
- The prompt stays short on purpose. The source rows already encode the target style,
  so the prep prompt should mainly preserve the mobile-feed framing and output shape.

Generate MLX-ready splits from the current promoted 5k release with:

```bash
python3 scripts/prepare_mlx_lora_dataset.py \
  --source hf_dataset/releases/recent_3y_balanced_5000_pilot_v2
```

That writes a versioned output directory containing `train.jsonl`, `valid.jsonl`,
`test.jsonl`, and `manifest.json`.

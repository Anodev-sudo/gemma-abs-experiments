# MLX LoRA Training Configs

This directory stores repo-local MLX LoRA configs for the current Gemma summary
fine-tuning workflow.

Current configs:

- `gemma_3_270m_it_abstract_only_v1.yaml`
  - Base model: `google/gemma-3-270m-it`
  - Dataset: reviewed 5k pilot release, converted to MLX `chat` format
  - Student input: `abstract_only`
  - Goal: get a first clean local LoRA run on the promoted pilot set before
    deciding whether to scale beyond 5k rows
- `gemma_3_270m_it_title_abstract_v1.yaml`
  - Base model: `google/gemma-3-270m-it`
  - Dataset: reviewed 5k pilot release, converted to MLX `chat` format
  - Student input: `title_abstract`
  - Goal: current preferred setup when the student should explicitly see the
    title during fine-tuning
- `gemma_3_270m_it_title_abstract_attn_mlp_v1.yaml`
  - Base model: `google/gemma-3-270m-it`
  - Dataset: reviewed 5k pilot release, converted to MLX `chat` format
  - Student input: `title_abstract`
  - Goal: isolate whether broadening LoRA targets from attention-only to
    attention + MLP reduces the remaining semantic failure modes
- `gemma_3_1b_it_title_abstract_v1.yaml`
  - Base model: `google/gemma-3-1b-it`
  - Dataset: reviewed 5k pilot release, converted to MLX `chat` format
  - Student input: `title_abstract`
  - Goal: apples-to-apples follow-up to the 270M run when we want to test
    whether the remaining errors are mostly model-size limited

Why this setup:

- The dataset format is `chat`, so training should be launched with
  `--mask-prompt` to put loss on the final assistant summary instead of the
  instruction text.
- The title-aware config uses `num_layers: -1`, which is MLX's explicit
  "all layers" setting and avoids guessing the block count.
- The adapter output path lives under `hf_bucket/experiments/` so training
  artifacts stay outside the main repo history.

Before running:

1. Install MLX LM training support:

```bash
python3 -m pip install "mlx-lm[train]"
```

2. Make sure you have accepted the Gemma license for the base model on Hugging
   Face:

```text
google/gemma-3-270m-it
google/gemma-3-1b-it
```

Train with:

```bash
python3 -m mlx_lm.lora \
  --config configs/mlx_lora/gemma_3_270m_it_title_abstract_v1.yaml \
  --mask-prompt
```

For the next 270M attention + MLP comparison run:

```bash
python3 -m mlx_lm.lora \
  --config configs/mlx_lora/gemma_3_270m_it_title_abstract_attn_mlp_v1.yaml \
  --mask-prompt
```

For the next 1B run:

```bash
python3 -m mlx_lm.lora \
  --config configs/mlx_lora/gemma_3_1b_it_title_abstract_v1.yaml \
  --mask-prompt
```

The 1B config keeps the same dataset, LoRA target modules, and learning rate as
the 270M run, but starts more conservatively on memory:

- `batch_size: 4`
- `grad_accumulation_steps: 2`
- `grad_checkpoint: true`

That preserves an effective batch size of 8 while reducing the risk of an
overnight OOM on the first launch.

If you want a faster first smoke run, override iterations on the command line:

```bash
python3 -m mlx_lm.lora \
  --config configs/mlx_lora/gemma_3_270m_it_title_abstract_v1.yaml \
  --mask-prompt \
  --iters 200
```

The prepared title-aware dataset lives at:

```text
hf_dataset/releases/recent_3y_balanced_5000_pilot_v2/mlx_lora/title_abstract_chat_gemma_270m_it_abstract_to_glanceable_summary_v1
```

Note:

- `google/gemma-3-1b-it` is not cached in the current local HF store yet, so the
  first 1B launch will download the base weights before training begins.

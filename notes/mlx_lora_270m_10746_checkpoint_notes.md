# MLX LoRA 270M 10.7k Checkpoint Notes

## 10.7k continuation from `0001400`

- Run: `gemma_3_270m_it_title_abstract_10746_continue_from_1400_v1`.
- Parent checkpoint: `gemma_3_270m_it_title_abstract_v1/adapters/0001400_adapters.safetensors`.
- Best validation checkpoint: `0003200_adapters.safetensors`.
- Training metrics looked healthy, but the fixed 100-paper holdout review did not support promotion over `attn_1400`.
- Two blinded reviewers preferred `attn_1400` by clear margins:
  - reviewer 1: `48` left wins, `28` right wins, `17` ties, `7` both bad
  - reviewer 2: `44` left wins, `29` right wins, `20` ties, `7` both bad
- Unblinded mapping: left was `attn_1400`, right was `attn_10746_ckpt3200`.

## Weaknesses Observed In `attn_10746_ckpt3200`

- Often more readable at a glance, but smoother by becoming more generic.
- Sometimes drops the paper's actual point, constraint, or mechanism.
- Occasionally flips polarity or reverses the key conclusion.
- Can replace an awkward but faithful summary with a cleaner but less faithful one.
- Still makes unsupported claim-strength errors, especially on papers with numeric results or benchmark-style framing.

Representative failure patterns:

- `2308.14989`: flipped compatibility to incompatibility.
- `2504.20147`: changed a gravitational-wave foreground that could obscure other sources into one that could reveal them.
- `2602.08965`: lost the no-communication quantum-advantage angle and became generic MARL language.
- `2305.08009`: sounded polished but garbled the uncertainty-quantification result.
- `2406.13831`: became smoother while dropping useful survey specificity.

Representative wins:

- `2503.12976`: better captured variable-based abstraction for real-time MAS model checking.
- `2406.14901`: better captured IDL locality/no-collision hashing.
- `2507.19295`: fixed a baseline reversal about the CB-cPIR vulnerability.
- `2507.13123`: avoided an unsupported numeric improvement claim.

## Next Experiment Rationale

- The continuation may have inherited a bad basin from `0001400`.
- The continuation learning rate, `5e-6`, may have been too conservative to find a better readability/faithfulness tradeoff.
- The next run should start from the raw base model on the same merged 10.7k dataset and keep the LoRA target modules unchanged.
- New run config: `configs/mlx_lora/gemma_3_270m_it_title_abstract_10746_from_base_aggressive_v1.yaml`.
- Key settings:
  - `resume_adapter_file: null`
  - `learning_rate: 1.5e-5`
  - `iters: 3600`
  - `batch_size: 4`
  - `grad_accumulation_steps: 2`
  - `grad_checkpoint: true`

Promotion gate for the next run:

- Pick the best validation checkpoint from the new base-model run.
- Compare it only against `attn_1400` on the fixed 100-paper MLX holdout.
- Do not promote on readability alone; require a net blinded-review win without repeating the cleaner-but-less-faithful failure pattern.

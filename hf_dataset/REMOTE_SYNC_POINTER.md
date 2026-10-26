# Remote dataset pointer

The local dataset staging payload was cleared on 2026-07-14 after checksum
verification confirmed all 507 local files matched the Hub dataset.

Dataset:

`https://huggingface.co/datasets/peering-latent/abstract-summaries`

Restore from the repository root:

```sh
hf download peering-latent/abstract-summaries \
  --type dataset \
  --local-dir ./hf_dataset
```

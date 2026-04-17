# HF Dataset Staging

This directory is reserved for files that should live in a separate Hugging Face dataset repo rather than the main source repo.

## Intended Contents

- `corpora/`: source abstract exports that are stable enough to version as dataset inputs
- `releases/`: promoted training-set snapshots that are ready to keep as named dataset outputs

The main git repo intentionally ignores the data files here. Only lightweight markdown docs stay visible to the source repo.

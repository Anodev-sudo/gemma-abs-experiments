# HF Bucket Staging

This directory is reserved for raw run artifacts and machine-generated experiment payloads that belong in mutable object storage rather than git history.

Typical contents:

- request and response JSONL
- dropped-row logs
- review outputs
- experiment audit packs
- ad hoc analysis payloads

Use this as the local staging root before syncing to a Hugging Face bucket.

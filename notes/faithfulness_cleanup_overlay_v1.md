# Faithfulness Cleanup Overlay v1

Purpose:
- Build an 800-row correction-only SFT overlay from high-risk rows in `recent_3y_balanced_10746_v1`.
- Teach the model to avoid unsupported numbers, named labels, mechanisms, causal direction, and overstrong claims.
- Keep this separate from the 10.7k release so the same `paper_id` does not appear with conflicting targets.

Implemented workflow:
- `scripts/prepare_faithfulness_cleanup_candidates.py` scores the 10.7k release and selects all high-risk rows plus a deterministic medium-risk top-up.
- `scripts/run_faithfulness_cleanup_rewriter.py` drafts cleaned summaries with the strict rewriter prompt and supports `--prepare-only` plus append/resume live runs.
- `scripts/prepare_faithfulness_cleanup_review_bundle.py` creates primary reviewer shards and a deterministic QA overlap shard.
- `scripts/prepare_faithfulness_cleanup_qa_followup.py` creates a second QA shard for low-confidence primary keeps/edits after primary decisions are filled.
- `scripts/materialize_faithfulness_cleanup_overlay.py` turns completed manual decisions into the final overlay release.

Current candidate artifact:
- Root: `hf_bucket/experiments/faithfulness_cleanup/recent_3y_balanced_10746_faithfulness_overlay_800_v1`
- Selected candidates: `candidate_selection/selected_candidates.jsonl`
- Candidate count: `1600`
- Final target: `800`

Review policy:
- Every candidate receives one primary manual-agent review.
- A deterministic 20% overlap goes through QA.
- Low-confidence keeps must be QA-audited before materialization unless explicitly overridden.
- More than 800 surviving rows are ranked by reviewer confidence, QA agreement, QA presence, risk tier, risk score, and `paper_id`.

Live rewrite command:

```bash
python3 scripts/run_faithfulness_cleanup_rewriter.py \
  2>&1 | tee hf_bucket/experiments/faithfulness_cleanup/recent_3y_balanced_10746_faithfulness_overlay_800_v1/rewrite_run/rewrite.log
```

Manual review bundle command after rewrites finish:

```bash
python3 scripts/prepare_faithfulness_cleanup_review_bundle.py
```

Materialization command after manual decisions are complete:

```bash
python3 scripts/prepare_faithfulness_cleanup_qa_followup.py \
  --review-bundle-dir hf_bucket/experiments/faithfulness_cleanup/recent_3y_balanced_10746_faithfulness_overlay_800_v1/manual_review

python3 scripts/materialize_faithfulness_cleanup_overlay.py \
  --review-bundle-dir hf_bucket/experiments/faithfulness_cleanup/recent_3y_balanced_10746_faithfulness_overlay_800_v1/manual_review
```

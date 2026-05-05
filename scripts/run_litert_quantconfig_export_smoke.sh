#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/workspace/hf_bucket/experiments/litert_exports/q4_hardening_combined_v1_lr_1e_6_300/q4_fused_dequantized_hf}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/q4_litert_quantconfig}"
CACHE_DIR="${CACHE_DIR:-/tmp/litert-cache}"
RECIPE="${RECIPE:-dynamic_int4_blockwise32}"
PREFILL_LENGTH="${PREFILL_LENGTH:-1024}"
CACHE_LENGTH="${CACHE_LENGTH:-1024}"
EXTERNALIZE_FLAG="${EXTERNALIZE_FLAG:---externalize-embedder}"

rm -rf "${OUTPUT_DIR}" "${CACHE_DIR}"
mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}"

python /workspace/scripts/run_litert_quantconfig_export_smoke.py \
  "${MODEL_DIR}" \
  "${OUTPUT_DIR}" \
  --recipe "${RECIPE}" \
  --prefill-length "${PREFILL_LENGTH}" \
  --cache-length "${CACHE_LENGTH}" \
  ${EXTERNALIZE_FLAG}

ls -lh "${OUTPUT_DIR}/model.litertlm"

python /workspace/scripts/smoke_litertlm_generation.py \
  "${OUTPUT_DIR}/model.litertlm" \
  --cache-dir "${CACHE_DIR}" \
  --fail-on-degenerate

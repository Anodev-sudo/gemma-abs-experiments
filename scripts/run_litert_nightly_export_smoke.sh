#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/workspace/hf_bucket/experiments/litert_exports/q4_hardening_combined_v1_lr_1e_6_300/q4_fused_dequantized_hf}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/q4_litert_nightly_wi4}"
CACHE_DIR="${CACHE_DIR:-/tmp/litert-cache}"
PREFILL_LENGTH="${PREFILL_LENGTH:-1024}"
CACHE_LENGTH="${CACHE_LENGTH:-1024}"
QUANTIZATION_RECIPE="${QUANTIZATION_RECIPE:-dynamic_wi4_afp32}"

python - <<'PY'
import importlib.metadata as metadata

for name in ("litert-torch", "litert-torch-nightly", "litert-lm"):
    try:
        print(f"before {name}={metadata.version(name)}", flush=True)
    except metadata.PackageNotFoundError:
        print(f"before {name}=not-installed", flush=True)
PY

python -m pip uninstall -y litert-torch
python -m pip install --no-cache-dir --upgrade litert-torch-nightly

python - <<'PY'
import importlib.metadata as metadata

for name in ("litert-torch", "litert-torch-nightly", "litert-lm"):
    try:
        print(f"after {name}={metadata.version(name)}", flush=True)
    except metadata.PackageNotFoundError:
        print(f"after {name}=not-installed", flush=True)
PY

rm -rf "${OUTPUT_DIR}" "${CACHE_DIR}"
mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}"

python -m litert_torch.generative.export_hf \
  "${MODEL_DIR}" \
  "${OUTPUT_DIR}" \
  --prefill_lengths="${PREFILL_LENGTH}," \
  --cache_length="${CACHE_LENGTH}" \
  --quantization_recipe="${QUANTIZATION_RECIPE}" \
  --externalize_embedder=True \
  --use_jinja_template=True \
  --bundle_litert_lm=True \
  --trust_remote_code=True

ls -lh "${OUTPUT_DIR}/model.litertlm"

python /workspace/scripts/smoke_litertlm_generation.py \
  "${OUTPUT_DIR}/model.litertlm" \
  --cache-dir "${CACHE_DIR}" \
  --fail-on-degenerate

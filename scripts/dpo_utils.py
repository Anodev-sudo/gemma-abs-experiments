from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_MODEL_REPO_ID = "google/gemma-3-270m-it"
DEFAULT_PROMPT_SPEC = (
    "prompts/mlx_lora/gemma_270m_it_abstract_to_glanceable_summary_v1.json"
)
DEFAULT_INPUT_MODE = "title_abstract"
DEFAULT_REUSE_SAMPLE_JSONL = (
    "hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_v1/"
    "review_samples/holdout_100_20260419-075505/generations.jsonl"
)
DEFAULT_MLX_HF_HOME = "hf_bucket/experiments/mlx_lora/.hf-home"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def load_prompt_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    required = ("prompt_version", "system_prompt", "user_prompt_templates")
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(
            f"Prompt spec {path} is missing required keys: {', '.join(missing)}"
        )
    return spec


def build_prompt_texts(
    row: dict[str, Any], prompt_spec: dict[str, Any], input_mode: str
) -> tuple[str, str]:
    templates = prompt_spec.get("user_prompt_templates", {})
    if input_mode not in templates:
        raise ValueError(
            f"Prompt spec {prompt_spec.get('prompt_version', '<unknown>')} "
            f"does not define input mode {input_mode!r}"
        )
    system_prompt = str(prompt_spec["system_prompt"]).strip()
    user_prompt = str(templates[input_mode]).format(
        title=str(row.get("title", "")).strip(),
        abstract=str(row.get("abstract", "")).strip(),
    )
    return system_prompt, user_prompt


def build_prompt_messages(
    row: dict[str, Any], prompt_spec: dict[str, Any], input_mode: str
) -> list[dict[str, str]]:
    system_prompt, user_prompt = build_prompt_texts(row, prompt_spec, input_mode)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def normalize_summary(value: str) -> str:
    return " ".join(value.strip().split())


def resolve_model_path(
    *,
    model_path: str | None,
    hf_home: Path,
    repo_id: str,
) -> str:
    if model_path:
        return model_path
    cache_dir = hf_home / "hub" / f"models--{repo_id.replace('/', '--')}"
    ref_path = cache_dir / "refs" / "main"
    if ref_path.exists():
        snapshot = ref_path.read_text(encoding="utf-8").strip()
        snapshot_path = cache_dir / "snapshots" / snapshot
        if snapshot_path.exists():
            return str(snapshot_path)
    return repo_id


def ensure_new_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        existing = list(path.iterdir())
        if existing and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite non-empty output directory: {path}. "
                "Pass --overwrite to replace it."
            )
    path.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower())
    return normalized.strip("_") or "item"


def utc_timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

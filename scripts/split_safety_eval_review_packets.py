#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_GENERATIONS_JSONL = (
    "hf_bucket/experiments/mlx_lora/safety_eval_generations/"
    "faith_overlay_1400_v2_final_600_safety_representative_2500_v1/"
    "generations.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    "hf_bucket/experiments/mlx_lora/safety_eval_generations/"
    "faith_overlay_1400_v2_final_600_safety_representative_2500_v1/"
    "manual_review_packets"
)
DEFAULT_CHUNK_SIZE = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split safety-eval generations into readable manual-review packets."
    )
    parser.add_argument("--generations-jsonl", default=DEFAULT_GENERATIONS_JSONL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def packet_name(packet_index: int, rows: list[dict[str, Any]]) -> str:
    first = rows[0].get("eval_id", f"row_{packet_index:04d}")
    last = rows[-1].get("eval_id", f"row_{packet_index:04d}")
    return f"packet_{packet_index:02d}_{first}_to_{last}"


def render_markdown(rows: list[dict[str, Any]], packet_label: str) -> str:
    lines = [
        f"# {packet_label}",
        "",
        "Reviewer notes:",
        "",
        "- Safety failure rows:",
        "- Borderline rows:",
        "- Quality-only patterns:",
        "",
    ]
    for row in rows:
        topics = ", ".join(row.get("edge_topics", [])) or "none"
        lines.extend(
            [
                f"## {row['eval_id']} · {row['paper_id']}",
                "",
                f"- Split: `{row.get('split', '')}`",
                f"- Category: `{row.get('primary_category', '')}`",
                f"- Edge topics: `{topics}`",
                f"- Adversarial variant: `{row.get('adversarial_variant', 'none')}`",
                "",
                f"Title: {row['title']}",
                "",
                "Summary:",
                "",
                f"> {row['summary']}",
                "",
                "Abstract:",
                "",
                row["abstract"],
                "",
                "Verdict:",
                "",
                "- [ ] OK",
                "- [ ] Borderline",
                "- [ ] Safety failure",
                "",
                "Notes:",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = load_jsonl(Path(args.generations_jsonl))
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{output_dir} already exists and is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    packets: list[dict[str, Any]] = []
    for packet_index, start in enumerate(range(0, len(rows), args.chunk_size), start=1):
        packet_rows = rows[start : start + args.chunk_size]
        label = packet_name(packet_index, packet_rows)
        jsonl_name = f"{label}.jsonl"
        markdown_name = f"{label}.md"
        write_jsonl(output_dir / jsonl_name, packet_rows)
        (output_dir / markdown_name).write_text(
            render_markdown(packet_rows, label),
            encoding="utf-8",
        )
        packets.append(
            {
                "packet": packet_index,
                "label": label,
                "rows": len(packet_rows),
                "first_eval_id": packet_rows[0]["eval_id"],
                "last_eval_id": packet_rows[-1]["eval_id"],
                "jsonl": jsonl_name,
                "markdown": markdown_name,
            }
        )

    manifest = {
        "generations_jsonl": args.generations_jsonl,
        "chunk_size": args.chunk_size,
        "packet_count": len(packets),
        "row_count": len(rows),
        "packets": packets,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(packets)} packets for {len(rows)} rows to {output_dir}")


if __name__ == "__main__":
    main()

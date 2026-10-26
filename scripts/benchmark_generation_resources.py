#!/usr/bin/env python3
"""Run a generation command while sampling local process-tree resources."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--stdout-log", required=True)
    parser.add_argument("--stderr-log", required=True)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("provide command after --")
    return args


def read_ps() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss=,pcpu=,command="],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 4)
        if len(parts) < 5:
            continue
        try:
            rows.append(
                {
                    "pid": int(parts[0]),
                    "ppid": int(parts[1]),
                    "rss_kb": int(parts[2]),
                    "cpu_percent": float(parts[3]),
                    "command": parts[4],
                }
            )
        except ValueError:
            continue
    return rows


def descendant_pids(root_pid: int, rows: list[dict[str, Any]]) -> set[int]:
    by_parent: dict[int, list[int]] = {}
    for row in rows:
        by_parent.setdefault(int(row["ppid"]), []).append(int(row["pid"]))
    seen = {root_pid}
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        for child in by_parent.get(pid, []):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {
            "samples": 0,
            "peak_tree_rss_mb": None,
            "median_tree_rss_mb": None,
            "mean_tree_rss_mb": None,
            "peak_process_rss_mb": None,
            "peak_tree_cpu_percent": None,
            "mean_tree_cpu_percent": None,
        }
    tree_rss = [sample["tree_rss_kb"] / 1024 for sample in samples]
    process_rss = [sample["peak_process_rss_kb"] / 1024 for sample in samples]
    tree_cpu = [sample["tree_cpu_percent"] for sample in samples]
    return {
        "samples": len(samples),
        "peak_tree_rss_mb": round(max(tree_rss), 2),
        "median_tree_rss_mb": round(statistics.median(tree_rss), 2),
        "mean_tree_rss_mb": round(statistics.fmean(tree_rss), 2),
        "peak_process_rss_mb": round(max(process_rss), 2),
        "peak_tree_cpu_percent": round(max(tree_cpu), 2),
        "mean_tree_cpu_percent": round(statistics.fmean(tree_cpu), 2),
    }


def main() -> None:
    args = parse_args()
    output_json = Path(args.output_json)
    stdout_log = Path(args.stdout_log)
    stderr_log = Path(args.stderr_log)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    samples: list[dict[str, Any]] = []
    with stdout_log.open("w", encoding="utf-8") as stdout, stderr_log.open("w", encoding="utf-8") as stderr:
        proc = subprocess.Popen(args.command, stdout=stdout, stderr=stderr)
        try:
            while proc.poll() is None:
                rows = read_ps()
                pids = descendant_pids(proc.pid, rows)
                tree_rows = [row for row in rows if row["pid"] in pids]
                if tree_rows:
                    samples.append(
                        {
                            "elapsed_s": round(time.monotonic() - start, 3),
                            "process_count": len(tree_rows),
                            "tree_rss_kb": sum(int(row["rss_kb"]) for row in tree_rows),
                            "peak_process_rss_kb": max(int(row["rss_kb"]) for row in tree_rows),
                            "tree_cpu_percent": sum(float(row["cpu_percent"]) for row in tree_rows),
                            "commands": [
                                {
                                    "pid": row["pid"],
                                    "rss_kb": row["rss_kb"],
                                    "cpu_percent": row["cpu_percent"],
                                    "command": str(row["command"])[:160],
                                }
                                for row in sorted(tree_rows, key=lambda value: int(value["rss_kb"]), reverse=True)[:5]
                            ],
                        }
                    )
                time.sleep(args.sample_interval)
        finally:
            return_code = proc.wait()
    wall_time_s = time.monotonic() - start

    payload = {
        "label": args.label,
        "command": args.command,
        "return_code": return_code,
        "wall_time_s": round(wall_time_s, 3),
        "sample_interval_s": args.sample_interval,
        "summary": summarize_samples(samples),
        "samples": samples,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
    }
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ["label", "return_code", "wall_time_s", "summary"]}, indent=2))


if __name__ == "__main__":
    main()

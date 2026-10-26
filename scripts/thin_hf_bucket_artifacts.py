#!/usr/bin/env python3
"""Replace synced heavy bucket artifacts with local restore manifests."""
from __future__ import annotations

import argparse
import json
import shlex
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_REMOTE_ROOT = "hf://buckets/peering-latent/gemma-abs-experiments"
DEFAULT_MANIFEST_NAME = "REMOTE_ARTIFACTS_MANIFEST.json"
EXCLUDED_PARTS = {".hf-home", "runtime", "__pycache__"}
EXCLUDED_NAMES = {".DS_Store", DEFAULT_MANIFEST_NAME, "LOCAL_CACHE_MANIFEST.json"}


@dataclass(frozen=True)
class Candidate:
    path: Path
    rel_path: str
    size_bytes: int
    mtime_utc: str
    remote_uri: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Thin local hf_bucket experiment artifacts after verifying that they "
            "are already present in the Hugging Face bucket."
        )
    )
    parser.add_argument("--root", default="hf_bucket", help="Local bucket staging root.")
    parser.add_argument(
        "--remote-root",
        default=DEFAULT_REMOTE_ROOT,
        help="Remote HF bucket URI matching the local root.",
    )
    parser.add_argument(
        "--sync-plan",
        required=True,
        help="JSONL sync plan proving files are already remote-matching skips.",
    )
    parser.add_argument(
        "--min-size-mib",
        type=int,
        default=10,
        help="Prune files larger than this size under experiments/.",
    )
    parser.add_argument(
        "--manifest-name",
        default=DEFAULT_MANIFEST_NAME,
        help="Manifest filename to leave in each affected directory.",
    )
    parser.add_argument("--apply", action="store_true", help="Actually write manifests and delete files.")
    return parser.parse_args()


def utc_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat().replace("+00:00", "Z")


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def load_verified_skips(plan_path: Path) -> tuple[dict[str, Any], set[str]]:
    header: dict[str, Any] = {}
    skips: set[str] = set()
    uploads: list[str] = []
    deletes: list[str] = []
    with plan_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            item_type = item.get("type")
            if item_type == "header":
                header = item
                continue
            if item_type != "operation":
                continue
            action = item.get("action")
            path = item.get("path")
            if not isinstance(path, str):
                continue
            if action == "skip":
                skips.add(path)
            elif action == "upload":
                uploads.append(path)
            elif action == "delete":
                deletes.append(path)
    if uploads or deletes:
        raise SystemExit(
            "Refusing to prune: sync plan still has "
            f"{len(uploads)} uploads and {len(deletes)} deletes."
        )
    return header, skips


def is_candidate(path: Path, root: Path, min_size_bytes: int, manifest_name: str) -> bool:
    rel = path.relative_to(root)
    if rel.parts[:1] != ("experiments",):
        return False
    if path.name in EXCLUDED_NAMES or path.name == manifest_name:
        return False
    if any(part in EXCLUDED_PARTS for part in rel.parts):
        return False
    return path.stat().st_size > min_size_bytes


def collect_candidates(root: Path, remote_root: str, min_size_mib: int, manifest_name: str) -> list[Candidate]:
    min_size_bytes = min_size_mib * 1024 * 1024
    remote_root = remote_root.rstrip("/")
    candidates: list[Candidate] = []
    experiments_root = root / "experiments"
    if not experiments_root.exists():
        raise SystemExit(f"Missing experiments directory: {experiments_root}")
    for path in sorted(experiments_root.rglob("*")):
        if not path.is_file():
            continue
        if not is_candidate(path, root, min_size_bytes, manifest_name):
            continue
        stat = path.stat()
        rel_path = path.relative_to(root).as_posix()
        candidates.append(
            Candidate(
                path=path,
                rel_path=rel_path,
                size_bytes=stat.st_size,
                mtime_utc=utc_timestamp(stat.st_mtime),
                remote_uri=f"{remote_root}/{rel_path}",
            )
        )
    return candidates


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(
    manifest_path: Path,
    candidates: list[Candidate],
    root: Path,
    remote_root: str,
    header: dict[str, Any],
    min_size_mib: int,
    manifest_name: str,
) -> None:
    existing = load_manifest(manifest_path)
    previous_files = existing.get("removed_files", [])
    by_path: dict[str, dict[str, Any]] = {}
    if isinstance(previous_files, list):
        for item in previous_files:
            if isinstance(item, dict) and isinstance(item.get("relative_path"), str):
                by_path[item["relative_path"]] = item

    for candidate in candidates:
        local_path = candidate.path.relative_to(Path.cwd()).as_posix()
        local_parent = Path(local_path).parent.as_posix()
        by_path[candidate.rel_path] = {
            "name": candidate.path.name,
            "relative_path": candidate.rel_path,
            "remote_uri": candidate.remote_uri,
            "size_bytes": candidate.size_bytes,
            "local_mtime_utc": candidate.mtime_utc,
            "restore_command": (
                f"mkdir -p {shlex.quote(local_parent)} && "
                f"hf buckets cp {shlex.quote(candidate.remote_uri)} {shlex.quote(local_path)}"
            ),
        }

    removed_files = [by_path[key] for key in sorted(by_path)]
    manifest = {
        "schema_version": 1,
        "created_at_utc": existing.get("created_at_utc") or now_utc(),
        "updated_at_utc": now_utc(),
        "remote_root": remote_root.rstrip("/"),
        "local_directory": manifest_path.parent.relative_to(root).as_posix(),
        "manifest_name": manifest_name,
        "prune_policy": {
            "scope": "hf_bucket/experiments",
            "min_size_mib_exclusive": min_size_mib,
            "excluded_path_parts": sorted(EXCLUDED_PARTS),
            "excluded_names": sorted(EXCLUDED_NAMES | {manifest_name}),
        },
        "sync_plan": {
            "path": str(header.get("_plan_path", "")),
            "timestamp": header.get("timestamp"),
            "source": header.get("source"),
            "dest": header.get("dest"),
            "summary": header.get("summary"),
        },
        "removed_file_count": len(removed_files),
        "removed_size_bytes": sum(int(item.get("size_bytes", 0)) for item in removed_files),
        "removed_files": removed_files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_summary(candidates: list[Candidate]) -> None:
    total_size = sum(candidate.size_bytes for candidate in candidates)
    print(f"Candidates: {len(candidates)} files, {total_size / 1024 / 1024 / 1024:.2f} GiB")
    grouped: dict[str, int] = defaultdict(int)
    grouped_size: dict[str, int] = defaultdict(int)
    for candidate in candidates:
        parts = candidate.rel_path.split("/")
        key = "/".join(parts[:3]) if len(parts) >= 3 else candidate.rel_path
        grouped[key] += 1
        grouped_size[key] += candidate.size_bytes
    print("Top affected directories:")
    for key in sorted(grouped, key=lambda item: grouped_size[item], reverse=True)[:20]:
        print(f"  {grouped[key]:4d} files  {grouped_size[key] / 1024 / 1024:.1f} MiB  {key}")


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    plan_path = Path(args.sync_plan).resolve()
    if not root.exists():
        raise SystemExit(f"Missing root: {root}")
    if not plan_path.exists():
        raise SystemExit(f"Missing sync plan: {plan_path}")

    header, verified_skips = load_verified_skips(plan_path)
    header["_plan_path"] = str(plan_path)
    candidates = collect_candidates(root, args.remote_root, args.min_size_mib, args.manifest_name)
    missing = [candidate.rel_path for candidate in candidates if candidate.rel_path not in verified_skips]
    if missing:
        print("Files missing from remote-matching sync plan:")
        for rel_path in missing[:100]:
            print(f"  {rel_path}")
        if len(missing) > 100:
            print(f"  ... and {len(missing) - 100} more")
        raise SystemExit(f"Refusing to prune {len(missing)} unverified files.")

    print_summary(candidates)
    if not args.apply:
        print("Dry run only. Re-run with --apply to write manifests and delete candidates.")
        return

    by_dir: dict[Path, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_dir[candidate.path.parent].append(candidate)

    for directory, directory_candidates in sorted(by_dir.items()):
        write_manifest(
            directory / args.manifest_name,
            directory_candidates,
            root,
            args.remote_root,
            header,
            args.min_size_mib,
            args.manifest_name,
        )

    deleted = 0
    deleted_bytes = 0
    for candidate in candidates:
        if candidate.path.exists() or candidate.path.is_symlink():
            deleted_bytes += candidate.size_bytes
            candidate.path.unlink()
            deleted += 1
    print(f"Deleted: {deleted} files, {deleted_bytes / 1024 / 1024 / 1024:.2f} GiB")


if __name__ == "__main__":
    main()

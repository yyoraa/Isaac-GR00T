# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, metadata-only RoboCasa365 subset selection."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


REQUIRED_CAMERAS = frozenset({"side_0", "side_1", "wrist_0"})
REQUIRED_SPLITS = ("pretrain", "composite_seen", "composite_unseen")


@dataclass(frozen=True)
class RoboCasa365Manifest:
    episodes: list[dict[str, Any]]
    total_bytes: int
    max_frames: int
    budget_bytes: int
    reserve_bytes: int


def _eligible(record: dict[str, Any]) -> bool:
    return (
        record.get("source") == "human"
        and record.get("task_type") == "composite"
        and record.get("split") in REQUIRED_SPLITS
        and REQUIRED_CAMERAS.issubset(record.get("cameras", []))
        and int(record.get("bytes", 0)) > 0
        and int(record.get("frames", 0)) > 0
    )


def _rank(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(record.get("subtask_count", 0)),
        -int(record.get("frames", 0)),
        str(record.get("task_id", "")),
        str(record.get("episode_id", "")),
    )


def build_manifest(
    registry: Iterable[dict[str, Any]], budget_bytes: int, reserve_bytes: int
) -> RoboCasa365Manifest:
    """Select genuine long human-composite episodes without downloading payloads."""
    if budget_bytes <= 0 or reserve_bytes < 0 or reserve_bytes >= budget_bytes:
        raise ValueError("budget_bytes must be positive and exceed reserve_bytes")
    available = budget_bytes - reserve_bytes
    candidates = [dict(record) for record in registry if _eligible(record)]

    selected = []
    selected_ids = set()
    total = 0
    for split in REQUIRED_SPLITS:
        split_candidates = sorted(
            (record for record in candidates if record["split"] == split), key=_rank
        )
        if not split_candidates:
            raise ValueError(f"no eligible RoboCasa365 episode for required split {split}")
        record = split_candidates[0]
        size = int(record["bytes"])
        if total + size > available:
            raise ValueError(
                f"data budget cannot cover all required RoboCasa365 splits within {available} bytes"
            )
        selected.append(record)
        selected_ids.add(record["episode_id"])
        total += size

    for record in sorted(candidates, key=_rank):
        if record["episode_id"] in selected_ids:
            continue
        size = int(record["bytes"])
        if total + size <= available:
            selected.append(record)
            selected_ids.add(record["episode_id"])
            total += size

    return RoboCasa365Manifest(
        episodes=selected,
        total_bytes=total,
        max_frames=max(int(record["frames"]) for record in selected),
        budget_bytes=budget_bytes,
        reserve_bytes=reserve_bytes,
    )


def write_manifest(manifest: RoboCasa365Manifest, path: str | Path, dataset_revision: str) -> Path:
    """Write a self-checking JSON manifest before any payload download."""
    if not dataset_revision:
        raise ValueError("dataset_revision must be non-empty")
    payload = {"dataset_revision": dataset_revision, **asdict(manifest)}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output

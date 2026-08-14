#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Materialize disk-safe, episode-complete RoboCasa365 file-000 subsets."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.parquet as pq


CAMERAS = (
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        raise FileExistsError(destination)
    os.link(source, destination)


def materialize(source: Path, destination: Path, split: str) -> dict:
    episode_path = source / "meta/episodes/chunk-000/file-000.parquet"
    rows = pq.read_table(episode_path).to_pylist()
    selected = [
        row
        for row in rows
        if row["data/chunk_index"] == 0
        and row["data/file_index"] == 0
        and all(row[f"videos/{camera}/chunk_index"] == 0 for camera in CAMERAS)
        and all(row[f"videos/{camera}/file_index"] == 0 for camera in CAMERAS)
    ]
    if not selected:
        raise ValueError(f"no complete file-000 episodes in {source}")
    if [row["episode_index"] for row in selected] != list(range(len(selected))):
        raise ValueError("file-000 episodes must be a contiguous prefix")

    destination.mkdir(parents=True, exist_ok=True)
    for name in ("tasks.parquet", "stats.json"):
        target = destination / "meta" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "meta" / name, target)

    episode_target = destination / "meta/episodes/chunk-000/file-000.parquet"
    episode_target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(selected), episode_target)

    info = json.loads((source / "meta/info.json").read_text())
    info["total_episodes"] = len(selected)
    info["total_frames"] = int(selected[-1]["dataset_to_index"])
    info["splits"] = {"train": f"0:{len(selected)}"}
    (destination / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")

    payloads = [source / "data/chunk-000/file-000.parquet"] + [
        source / f"videos/{camera}/chunk-000/file-000.mp4" for camera in CAMERAS
    ]
    linked = []
    for payload in payloads:
        relative = payload.relative_to(source)
        target = destination / relative
        _link(payload, target)
        linked.append(
            {
                "path": str(relative),
                "bytes": payload.stat().st_size,
                "sha256": _sha256(payload),
            }
        )
    return {
        "split": split,
        "path": str(destination),
        "episodes": len(selected),
        "frames": info["total_frames"],
        "max_episode_frames": max(int(row["length"]) for row in selected),
        "files": linked,
        "statistics_scope": "regenerate_for_materialized_subset",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=4,
        metavar=("SPLIT", "REPO_ID", "REVISION", "SOURCE"),
        required=True,
    )
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    datasets = []
    for split, repo_id, revision, source in args.dataset:
        record = materialize(Path(source), output_root / split, split)
        record.update({"repo_id": repo_id, "revision": revision})
        datasets.append(record)
    manifest = {
        "format": "robottt-robocasa365-file0-v1",
        "datasets": datasets,
        "total_bytes": sum(item["bytes"] for dataset in datasets for item in dataset["files"]),
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

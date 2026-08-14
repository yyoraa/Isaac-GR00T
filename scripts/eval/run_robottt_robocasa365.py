#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Record a reproducible RoboTTT RoboCasa365 evaluation result."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal

from gr00t.eval.robottt_metrics import aggregate_robottt_metrics
import tyro


EvaluationMode = Literal["base", "history", "gdn", "robottt_update_off", "robottt_full"]
EVALUATION_MODES: tuple[EvaluationMode, ...] = (
    "base",
    "history",
    "gdn",
    "robottt_update_off",
    "robottt_full",
)
CONTEXT_SWEEP = (128, 512, 1024, 2048, "real_max")


@dataclass
class EvalConfig:
    checkpoint: str
    manifest: str
    episode_records: str
    output: str
    mode: EvaluationMode = "robottt_full"
    context_length: int = 1024
    model_revision: str = "public-gr00t-n1.7-ga"
    data_revision: str = "unknown"


def _hash_path(path: str | Path) -> str:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    files = (
        [source]
        if source.is_file()
        else sorted(item for item in source.rglob("*") if item.is_file())
    )
    for item in files:
        name = item.relative_to(source) if source.is_dir() else Path(item.name)
        digest.update(str(name).encode())
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def main(config: EvalConfig) -> None:
    records = [
        json.loads(line) for line in Path(config.episode_records).read_text().splitlines() if line
    ]
    result = {
        "config": asdict(config),
        "checkpoint_sha256": _hash_path(config.checkpoint),
        "manifest_sha256": _hash_path(config.manifest),
        "metrics": aggregate_robottt_metrics(records),
    }
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(tyro.cli(EvalConfig, description=__doc__))

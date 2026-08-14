# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Scientific aggregation for RoboTTT RoboCasa365 rollouts."""

from collections import Counter
from collections.abc import Iterable, Mapping
from statistics import fmean
from typing import Any


_REQUIRED_NUMERIC = (
    "completed_stages",
    "context_length",
    "subtask_count",
    "latency_ms",
    "gpu_peak_bytes",
    "cpu_peak_bytes",
)
_VALID_SPLITS = {"seen", "unseen"}


def _summary(records: list[Mapping[str, Any]]) -> dict[str, float | int]:
    return {
        "episodes": len(records),
        "success_rate": fmean(float(record["success"]) for record in records),
        "mean_completed_stages": fmean(float(record["completed_stages"]) for record in records),
        "mean_subtask_count": fmean(float(record["subtask_count"]) for record in records),
        "mean_latency_ms": fmean(float(record["latency_ms"]) for record in records),
        "mean_gpu_peak_bytes": fmean(float(record["gpu_peak_bytes"]) for record in records),
        "mean_cpu_peak_bytes": fmean(float(record["cpu_peak_bytes"]) for record in records),
    }


def aggregate_robottt_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    episodes = list(records)
    if not episodes:
        raise ValueError("at least one evaluation episode is required")
    for record in episodes:
        split = record.get("split")
        if split not in _VALID_SPLITS:
            raise ValueError(f"split must be seen or unseen, got {split!r}")
        missing = {"success", *_REQUIRED_NUMERIC}.difference(record)
        if missing:
            raise ValueError(f"evaluation record is missing: {sorted(missing)}")

    by_split = {
        split: _summary([record for record in episodes if record["split"] == split])
        for split in sorted(_VALID_SPLITS)
        if any(record["split"] == split for record in episodes)
    }
    context_counts = Counter(str(int(record["context_length"])) for record in episodes)
    return {
        "overall": _summary(episodes),
        "by_split": by_split,
        "context_counts": dict(sorted(context_counts.items(), key=lambda item: int(item[0]))),
    }

import json

from gr00t.data.robocasa365_manifest import build_manifest, write_manifest


def _registry():
    return [
        {
            "episode_id": "p-long",
            "task_id": "p",
            "split": "pretrain",
            "source": "human",
            "task_type": "composite",
            "subtask_count": 8,
            "frames": 800,
            "bytes": 30,
            "cameras": ["side_0", "side_1", "wrist_0"],
        },
        {
            "episode_id": "s-long",
            "task_id": "s",
            "split": "composite_seen",
            "source": "human",
            "task_type": "composite",
            "subtask_count": 7,
            "frames": 700,
            "bytes": 20,
            "cameras": ["side_0", "side_1", "wrist_0"],
        },
        {
            "episode_id": "u-long",
            "task_id": "u",
            "split": "composite_unseen",
            "source": "human",
            "task_type": "composite",
            "subtask_count": 9,
            "frames": 900,
            "bytes": 25,
            "cameras": ["side_0", "side_1", "wrist_0"],
        },
        {
            "episode_id": "atomic",
            "task_id": "a",
            "split": "pretrain",
            "source": "mimicgen",
            "task_type": "atomic",
            "subtask_count": 1,
            "frames": 1000,
            "bytes": 1,
            "cameras": ["side_0", "side_1", "wrist_0"],
        },
        {
            "episode_id": "missing-camera",
            "task_id": "m",
            "split": "pretrain",
            "source": "human",
            "task_type": "composite",
            "subtask_count": 10,
            "frames": 1000,
            "bytes": 1,
            "cameras": ["side_0", "wrist_0"],
        },
    ]


def test_manifest_keeps_real_human_composites_and_all_required_splits():
    manifest = build_manifest(_registry(), budget_bytes=80, reserve_bytes=5)

    assert [item["episode_id"] for item in manifest.episodes] == [
        "p-long",
        "s-long",
        "u-long",
    ]
    assert manifest.total_bytes == 75
    assert manifest.max_frames == 900


def test_manifest_is_deterministic_under_registry_reordering():
    first = build_manifest(_registry(), budget_bytes=80, reserve_bytes=5)
    second = build_manifest(list(reversed(_registry())), budget_bytes=80, reserve_bytes=5)

    assert first.episodes == second.episodes


def test_manifest_rejects_budget_that_cannot_cover_required_splits():
    try:
        build_manifest(_registry(), budget_bytes=79, reserve_bytes=5)
    except ValueError as exc:
        assert "required RoboCasa365 splits" in str(exc)
    else:
        raise AssertionError("insufficient data budget was accepted")


def test_write_manifest_records_revision_and_checksum(tmp_path):
    manifest = build_manifest(_registry(), budget_bytes=80, reserve_bytes=5)
    path = tmp_path / "manifest.json"

    write_manifest(manifest, path, dataset_revision="rev-123")
    payload = json.loads(path.read_text())

    assert payload["dataset_revision"] == "rev-123"
    assert len(payload["manifest_sha256"]) == 64
    assert payload["episodes"][0]["episode_id"] == "p-long"

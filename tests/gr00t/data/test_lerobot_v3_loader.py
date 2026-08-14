import json

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.types import ModalityConfig
import pandas as pd


def test_v3_packed_parquet_metadata_and_episode_filtering(tmp_path):
    (tmp_path / "meta/episodes/chunk-000").mkdir(parents=True)
    (tmp_path / "data/chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "PandaOmron",
        "total_episodes": 1,
        "total_frames": 2,
        "chunks_size": 1000,
        "fps": 20,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {},
    }
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    pd.DataFrame(
        [
            {
                "episode_index": 0,
                "length": 2,
                "data/chunk_index": 0,
                "data/file_index": 0,
            }
        ]
    ).to_parquet(tmp_path / "meta/episodes/chunk-000/file-000.parquet", index=False)
    pd.DataFrame([{"task_index": 0, "task": "move object"}]).to_parquet(
        tmp_path / "meta/tasks.parquet", index=False
    )
    stats = {
        "observation.state": {
            name: [0.0] * 16 for name in ("min", "max", "mean", "std", "q01", "q99")
        },
        "action": {name: [0.0] * 12 for name in ("min", "max", "mean", "std", "q01", "q99")},
    }
    (tmp_path / "meta/stats.json").write_text(json.dumps(stats))
    pd.DataFrame(
        {
            "episode_index": [0, 0, 1],
            "observation.state": [[0.0] * 16, [1.0] * 16, [9.0] * 16],
            "action": [[0.0] * 12, [1.0] * 12, [9.0] * 12],
            "annotation.human.task_description": [0, 0, 0],
        }
    ).to_parquet(tmp_path / "data/chunk-000/file-000.parquet", index=False)
    configs = {
        "state": ModalityConfig(delta_indices=[0], modality_keys=["base_position"]),
        "action": ModalityConfig(delta_indices=[0], modality_keys=["base_motion"]),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.task_description"]
        ),
    }

    loader = LeRobotEpisodeLoader(tmp_path, configs)
    episode = loader._load_parquet_data(0)

    assert loader.is_lerobot_v3 is True
    assert loader.episode_lengths == [2]
    assert episode["language.annotation.human.task_description"].tolist() == [
        "move object",
        "move object",
    ]
    assert len(episode) == 2
    assert episode["state.base_position"].iloc[1].tolist() == [1.0, 1.0, 1.0]

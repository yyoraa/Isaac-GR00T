from gr00t.data.dataset.trajectory_sequence_dataset import (
    TrajectoryCollator,
    TrajectorySequenceDataset,
)
import torch


class _Loader:
    def __init__(self):
        self.episodes = [list(range(5)), list(range(100, 104))]
        self.episode_lengths = [len(episode) for episode in self.episodes]

    def __getitem__(self, index):
        return self.episodes[index]

    def get_dataset_statistics(self):
        return {"state": {}}


def _step_factory(episode, step):
    return {"value": torch.tensor(episode[step]), "action": torch.tensor([episode[step]])}


def test_windows_never_cross_episode_boundaries():
    dataset = TrajectorySequenceDataset(
        episode_loader=_Loader(),
        step_factory=_step_factory,
        context_length=3,
        stride=2,
        action_horizon=1,
    )

    samples = [dataset.get_shard(index)[0] for index in range(len(dataset))]

    assert [(sample["episode_id"], sample["start"], sample["length"]) for sample in samples] == [
        (0, 0, 3),
        (0, 2, 3),
        (0, 4, 1),
        (1, 0, 3),
        (1, 2, 2),
    ]
    for sample in samples:
        values = [int(step["value"]) for step in sample["trajectory_steps"]]
        assert all(value < 100 for value in values) or all(value >= 100 for value in values)


def test_action_horizon_reduces_valid_window_starts():
    dataset = TrajectorySequenceDataset(
        episode_loader=_Loader(),
        step_factory=_step_factory,
        context_length=8,
        stride=1,
        action_horizon=3,
    )

    samples = [dataset.get_shard(index)[0] for index in range(len(dataset))]
    assert [(sample["episode_id"], sample["length"]) for sample in samples] == [(0, 3), (1, 2)]


class _BaseCollator:
    def __call__(self, steps):
        return {"inputs": {key: torch.stack([step[key] for step in steps]) for key in steps[0]}}


def test_collator_pads_time_and_emits_loss_masks():
    dataset = TrajectorySequenceDataset(
        episode_loader=_Loader(),
        step_factory=_step_factory,
        context_length=3,
        stride=2,
        action_horizon=1,
    )
    long_sample = dataset.get_shard(0)[0]
    short_sample = dataset.get_shard(2)[0]

    batch = TrajectoryCollator(_BaseCollator())([long_sample, short_sample])["inputs"]

    assert batch["value"].shape == (2, 3)
    assert batch["valid_mask"].tolist() == [[True, True, True], [True, False, False]]
    assert batch["action_loss_mask"].tolist() == [[True, True, True], [True, False, False]]
    assert batch["episode_reset_mask"].tolist() == [[True, False, False], [True, False, False]]
    assert batch["temporal_positions"].tolist() == [[0, 1, 2], [4, 4, 4]]


def test_collator_applies_runtime_curriculum_context():
    dataset = TrajectorySequenceDataset(
        episode_loader=_Loader(),
        step_factory=_step_factory,
        context_length=5,
        stride=5,
        action_horizon=1,
    )
    collator = TrajectoryCollator(_BaseCollator())
    collator.set_context_length(2)

    batch = collator([dataset.get_shard(0)[0]])["inputs"]

    assert batch["value"].shape == (1, 2)
    assert batch["valid_mask"].tolist() == [[True, True]]
    assert batch["temporal_positions"].tolist() == [[0, 1]]

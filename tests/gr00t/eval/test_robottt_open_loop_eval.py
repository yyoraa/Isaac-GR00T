import json
from types import SimpleNamespace

from gr00t.eval.open_loop_eval import (
    ArgsConfig,
    configure_robottt_evaluation_mode,
    get_robottt_observation_count,
    reset_policy_for_trajectory,
    write_evaluation_json,
)
import pytest


class _StatefulActionHead:
    def __init__(self):
        self.online_updates = None
        self.robottt_observation_count = 0

    def set_robottt_online_updates(self, enabled):
        self.online_updates = enabled


class _StatefulPolicy:
    def __init__(self):
        self.model = SimpleNamespace(action_head=_StatefulActionHead())
        self.state = 0
        self.reset_count = 0

    def reset(self):
        self.state = 0
        self.reset_count += 1


@pytest.mark.parametrize(
    ("mode", "expected"),
    (("robottt_update_off", False), ("robottt_full", True)),
)
def test_configure_robottt_evaluation_mode_controls_online_updates(mode, expected):
    policy = _StatefulPolicy()

    configure_robottt_evaluation_mode(policy, mode)

    assert policy.model.action_head.online_updates is expected


def test_every_trajectory_reset_starts_from_initial_state():
    policy = _StatefulPolicy()

    for _ in range(2):
        policy.state = 17
        reset_policy_for_trajectory(policy)
        assert policy.state == 0

    assert policy.reset_count == 2


def test_robottt_observation_count_is_available_for_result_auditing():
    policy = _StatefulPolicy()
    policy.model.action_head.robottt_observation_count = 3

    assert get_robottt_observation_count(policy) == 3


def test_write_evaluation_json_preserves_literal_metrics(tmp_path):
    config = ArgsConfig(
        mode="robottt_full",
        model_path="checkpoint-100",
        dataset_path="composite_unseen",
        traj_ids=[1, 4],
        steps=32,
        execution_horizon=16,
        denoising_steps=2,
        seed=7,
    )
    records = [
        {"trajectory_id": 1, "mse": 1.0, "mae": 0.5},
        {"trajectory_id": 4, "mse": 3.0, "mae": 1.5},
    ]
    output = tmp_path / "nested" / "result.json"

    written = write_evaluation_json(output, config, records)
    payload = json.loads(written.read_text())

    assert payload["per_trajectory"] == records
    assert payload["aggregate"] == {"mse": 2.0, "mae": 1.0, "num_trajectories": 2}
    assert payload["mode"] == "robottt_full"
    assert payload["seed"] == 7

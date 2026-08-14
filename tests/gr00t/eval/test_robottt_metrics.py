from gr00t.eval.robottt_metrics import aggregate_robottt_metrics
import pytest


def test_metrics_aggregate_exact_split_means_and_context_counts():
    records = [
        {
            "split": "seen",
            "success": True,
            "completed_stages": 4,
            "context_length": 128,
            "subtask_count": 4,
            "latency_ms": 10.0,
            "gpu_peak_bytes": 100,
            "cpu_peak_bytes": 1000,
        },
        {
            "split": "unseen",
            "success": False,
            "completed_stages": 2,
            "context_length": 512,
            "subtask_count": 5,
            "latency_ms": 30.0,
            "gpu_peak_bytes": 300,
            "cpu_peak_bytes": 3000,
        },
    ]

    result = aggregate_robottt_metrics(records)

    assert result["overall"]["success_rate"] == pytest.approx(0.5)
    assert result["overall"]["mean_completed_stages"] == pytest.approx(3.0)
    assert result["overall"]["mean_latency_ms"] == pytest.approx(20.0)
    assert result["by_split"]["seen"]["success_rate"] == pytest.approx(1.0)
    assert result["by_split"]["unseen"]["success_rate"] == pytest.approx(0.0)
    assert result["context_counts"] == {"128": 1, "512": 1}


def test_metrics_reject_empty_or_unknown_split():
    with pytest.raises(ValueError):
        aggregate_robottt_metrics([])
    with pytest.raises(ValueError):
        aggregate_robottt_metrics([{"split": "private"}])

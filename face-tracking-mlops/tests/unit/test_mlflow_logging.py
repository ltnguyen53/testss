"""Unit test cho src/training/mlflow_logging.py bằng fake client (không cần
MLflow server thật, không cần torch — module này không phụ thuộc torch)."""

from dataclasses import dataclass, field
from typing import Any

from src.training.mlflow_logging import TrainRunLogger, flatten_config_for_logging


@dataclass
class FakeMlflowClient:
    params: dict[str, Any] = field(default_factory=dict)
    metrics_log: list[tuple[dict[str, float], int]] = field(default_factory=list)
    tags: dict[str, Any] = field(default_factory=dict)

    def log_params(self, params: dict[str, Any]) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.metrics_log.append((dict(metrics), step))

    def set_tag(self, key: str, value: Any) -> None:
        self.tags[key] = value


def test_flatten_config_nested_keys_joined_by_dot():
    cfg = {"optimizer": {"head_lr": 1e-3, "nested": {"deep": 1}}, "seed": 42}

    flat = flatten_config_for_logging(cfg)

    assert flat == {"optimizer.head_lr": 1e-3, "optimizer.nested.deep": 1, "seed": 42}


def test_flatten_config_includes_checkpoint_and_mlflow_sections():
    """Khác compute_config_hash (checkpoint.py) — logging KHÔNG loại section
    nào, xem docstring module."""
    cfg = {"checkpoint": {"dir": "/drive/ckpt"}, "mlflow": {"experiment_name": "run1"}}

    flat = flatten_config_for_logging(cfg)

    assert flat == {"checkpoint.dir": "/drive/ckpt", "mlflow.experiment_name": "run1"}


def test_log_config_calls_client_with_flattened_params():
    client = FakeMlflowClient()
    logger = TrainRunLogger(client)

    logger.log_config({"a": {"b": 1}})

    assert client.params == {"a.b": 1}


def test_log_freeze_report_computes_trainable_fraction():
    client = FakeMlflowClient()
    logger = TrainRunLogger(client)

    class FakeReport:
        total_params = 1000
        trainable_params = 250
        bn_affine_params = 50

    logger.log_freeze_report(FakeReport())

    assert client.params["freeze.trainable_fraction"] == 0.25
    assert client.params["freeze.total_params"] == 1000


def test_log_step_metrics_passes_step_through():
    client = FakeMlflowClient()
    logger = TrainRunLogger(client)

    logger.log_step_metrics({"loss": 0.5}, step=42)

    assert client.metrics_log == [({"loss": 0.5}, 42)]


def test_log_resume_sets_expected_tags():
    client = FakeMlflowClient()
    logger = TrainRunLogger(client)

    logger.log_resume(
        resumed=True, config_hash="abc123", checkpoint_dir="/drive/ckpt/ckpt_epoch0005"
    )

    assert client.tags == {
        "resumed_from_checkpoint": "True",
        "config_hash": "abc123",
        "resumed_checkpoint_dir": "/drive/ckpt/ckpt_epoch0005",
    }


def test_log_resume_omits_checkpoint_dir_tag_when_starting_fresh():
    client = FakeMlflowClient()
    logger = TrainRunLogger(client)

    logger.log_resume(resumed=False, config_hash="abc123", checkpoint_dir=None)

    assert "resumed_checkpoint_dir" not in client.tags
    assert client.tags["resumed_from_checkpoint"] == "False"

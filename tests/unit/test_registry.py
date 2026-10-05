"""Unit test cho src/evaluation/registry.py. Thuần Python — fake client, không
cần MLflow server thật."""

import math

from src.evaluation.registry import register_if_better, should_register


def test_should_register_true_when_no_production_model_exists_yet():
    assert should_register(new_metric=0.5, current_production_metric=None) is True


def test_should_register_true_when_strictly_better():
    assert should_register(new_metric=0.9, current_production_metric=0.8) is True


def test_should_register_false_when_worse_or_equal():
    assert should_register(new_metric=0.7, current_production_metric=0.8) is False
    assert should_register(new_metric=0.8, current_production_metric=0.8) is False


def test_should_register_false_when_new_metric_is_nan():
    """Không so sánh an toàn được với NaN — từ chối đăng ký thay vì để
    `nan > x` (luôn False trong Python) âm thầm quyết định sai lý do."""
    assert should_register(new_metric=math.nan, current_production_metric=0.5) is False


class _FakeRegistryClient:
    def __init__(self, production_metric: float | None) -> None:
        self.production_metric = production_metric
        self.registered: list[tuple[str, str, str, float]] = []

    def get_production_metric(self, registered_model_name: str) -> float | None:
        return self.production_metric

    def register_model(
        self, registered_model_name: str, run_id: str, config_hash: str, metric: float
    ) -> str:
        self.registered.append((registered_model_name, run_id, config_hash, metric))
        return f"v{len(self.registered)}"


def test_register_if_better_registers_first_model():
    client = _FakeRegistryClient(production_metric=None)

    version = register_if_better(
        client,
        registered_model_name="face-recognition-occlusion",
        run_id="run1",
        config_hash="hash1",
        new_test_metric=0.7,
    )

    assert version == "v1"
    assert client.registered == [("face-recognition-occlusion", "run1", "hash1", 0.7)]


def test_register_if_better_skips_when_not_an_improvement():
    client = _FakeRegistryClient(production_metric=0.9)

    version = register_if_better(
        client,
        registered_model_name="face-recognition-occlusion",
        run_id="run2",
        config_hash="hash2",
        new_test_metric=0.7,
    )

    assert version is None
    assert client.registered == []


def test_register_if_better_registers_when_improved():
    client = _FakeRegistryClient(production_metric=0.7)

    version = register_if_better(
        client,
        registered_model_name="face-recognition-occlusion",
        run_id="run3",
        config_hash="hash3",
        new_test_metric=0.85,
    )

    assert version == "v1"

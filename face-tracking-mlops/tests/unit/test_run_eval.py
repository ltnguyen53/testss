"""Integration test cho src/evaluation/run_eval.py — chạy THẬT (không mock
logic eval, chỉ thay backbone/mlflow/registry client bằng fake, cùng ranh giới
đã dùng ở test_train.py: thay dependency BÊN NGOÀI, giữ nguyên orchestration
code của chính ta).

Bỏ qua sạch nếu torch chưa cài."""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402
from PIL import Image  # noqa: E402

from src.data.labels import ImageRecord, write_manifest_csv  # noqa: E402
from src.evaluation.run_eval import load_inference_config, run_eval  # noqa: E402
from src.training.checkpoint import (  # noqa: E402
    CheckpointManager,
    CheckpointMeta,
    compute_config_hash,
)
from src.training.head import ArcFaceHead  # noqa: E402
from src.training.model import FaceModel  # noqa: E402

EMBED_DIM = 8
REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_EXPORT_CONFIG = REPO_ROOT / "configs" / "export.yaml"


class _TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(self.conv(x)).flatten(1))


class _FakeMlflow:
    def __init__(self) -> None:
        self.params: dict = {}
        self.metrics: dict = {}
        self.experiments_set: list[str] = []
        self._run_id = "fake-run-id-001"

    def set_experiment(self, name: str) -> None:
        self.experiments_set.append(name)

    def start_run(self):
        return _FakeRunContext(self)

    def log_param(self, key: str, value: Any) -> None:
        self.params[key] = value

    def log_metric(self, key: str, value: float) -> None:
        self.metrics[key] = value

    def active_run(self):
        return _ActiveRun(self._run_id)


@dataclass
class _ActiveRun:
    class _Info:
        def __init__(self, run_id: str) -> None:
            self.run_id = run_id

    run_id_holder: str

    def __post_init__(self):
        self.info = self._Info(self.run_id_holder)


class _FakeRunContext:
    def __init__(self, parent: _FakeMlflow) -> None:
        self.parent = parent

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@dataclass
class _FakeRegistryClient:
    registered: list[tuple] = field(default_factory=list)

    def get_production_metric(self, registered_model_name: str):
        return None  # chưa có Production -> luôn đăng ký lần đầu

    def register_model(self, registered_model_name, run_id, config_hash, metric) -> str:
        self.registered.append((registered_model_name, run_id, config_hash, metric))
        return "v1"


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), color=color).save(path)


def _build_manifest(root: Path, csv_path: Path, identities: list[str]) -> None:
    """Mỗi identity: 1 unmasked + 1 masked thật — đủ để tạo genuine pair ở cả
    3 nhóm occlusion."""
    records = []
    for identity in identities:
        _write_image(root / identity / "u.jpg", (10, 0, 0))
        _write_image(root / identity / "m.jpg", (0, 10, 0))
        records.append(ImageRecord(f"{identity}/u.jpg", identity, False, "rmfd_unmasked"))
        records.append(ImageRecord(f"{identity}/m.jpg", identity, True, "rmfd_masked"))
    write_manifest_csv(csv_path, records)


def _base_cfg(train_csv: Path, val_csv: Path, checkpoint_dir: Path) -> dict:
    return {
        "seed": 1,
        "data": {"train_csv": str(train_csv), "val_csv": str(val_csv)},
        "backbone": {
            "arch": "unused-monkeypatched",
            "num_features": EMBED_DIM,
            "dropout": 0.0,
            "arcface_torch_dir": "unused",
            "pretrained_weights": "unused",
            "weights_source": "unused",
        },
        "freeze": {
            "trainable_blocks": ["conv"],
            "train_embedding_fc": False,
            "head_module": "head",
            "train_bn_affine_everywhere": False,
            "bn_affine_exclude_modules": [],
        },
        "head": {"arcface_margin": 0.3, "arcface_scale": 32.0},
        "optimizer": {
            "head_lr": 1.0e-2,
            "backbone_unfrozen_lr": 1.0e-3,
            "bn_affine_lr": 1.0e-3,
            "weight_decay": 0.0,
            "bn_affine_weight_decay": 0.0,
        },
        "sampler": {"identities_per_batch": 2, "images_per_identity": 2, "masked_per_identity": 1},
        "training": {"optimizer_type": "sgd", "momentum": 0.9, "epochs": 1, "amp": False},
        "checkpoint": {"save_every_n_steps": 1, "keep_last_n": 2, "dir": str(checkpoint_dir)},
        "mlflow": {"experiment_name": "test-eval-exp", "registered_model_name": "test-model"},
    }


def _write_inference_cfg(path: Path, test_csv: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "threshold": {"target_far": 0.5, "value": None, "swept_on_config_hash": None},
                "eval": {"max_impostor_pairs": 20, "pair_sampling_seed": 42},
                "data": {"test_csv": str(test_csv)},
            }
        )
    )


@pytest.fixture
def _patched_env(monkeypatch):
    monkeypatch.setattr(
        "src.evaluation.run_eval.load_pretrained_backbone", lambda cfg: _TinyBackbone()
    )
    fake_mlflow = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    fake_registry = _FakeRegistryClient()
    monkeypatch.setattr("src.evaluation.run_eval.build_real_registry_client", lambda: fake_registry)
    return fake_mlflow, fake_registry


def _save_fake_checkpoint(cfg: dict, checkpoint_dir: Path) -> None:
    """Mô phỏng đúng những gì train.py để lại trên Drive — dùng CÙNG
    CheckpointManager/CheckpointMeta thật, không tự bịa format riêng."""
    backbone = _TinyBackbone()
    head = ArcFaceHead(EMBED_DIM, num_classes=2, margin=0.3, scale=32.0)
    model = FaceModel(backbone, head)
    manager = CheckpointManager(checkpoint_dir, keep_last_n=2)
    manager.save(
        {"model": model.state_dict(), "optimizer": {}, "rng": {}},
        CheckpointMeta(
            timestamp="2026-01-01T00:00:00+00:00",
            config_hash=compute_config_hash(cfg),
            git_commit_hash="deadbeef",
            epoch=1,
            step=0,
        ),
    )


def test_load_inference_config_validates_required_keys(tmp_path):
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(yaml.safe_dump({"threshold": {"target_far": 0.1}}))

    with pytest.raises(ValueError, match="eval"):
        load_inference_config(bad_path)


def test_run_eval_end_to_end_writes_report_and_updates_threshold(tmp_path, _patched_env):
    fake_mlflow, fake_registry = _patched_env
    images_root = tmp_path / "images"
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"
    _build_manifest(images_root, train_csv, ["id_train_0", "id_train_1"])
    _build_manifest(images_root, val_csv, ["id_val_0", "id_val_1"])
    _build_manifest(images_root, test_csv, ["id_test_0", "id_test_1"])

    checkpoint_dir = tmp_path / "ckpt"
    cfg = _base_cfg(train_csv, val_csv, checkpoint_dir)
    _save_fake_checkpoint(cfg, checkpoint_dir)

    inference_cfg_path = tmp_path / "inference.yaml"
    _write_inference_cfg(inference_cfg_path, test_csv)
    inference_cfg = load_inference_config(inference_cfg_path)

    report_path = tmp_path / "reports" / "eval_metrics.json"
    run_eval(
        cfg,
        inference_cfg,
        images_root=images_root,
        checkpoint_dir=checkpoint_dir,
        export_config=REAL_EXPORT_CONFIG,
        report_path=report_path,
        inference_config_path=inference_cfg_path,
    )

    # Report ghi ra đúng file, đọc lại được bằng JSON chuẩn.
    assert report_path.exists()
    loaded = json.loads(report_path.read_text())
    assert loaded["config_hash"] == compute_config_hash(cfg)
    assert "overall" in loaded["groups"]
    assert "masked_masked" in loaded["groups"]
    # `report` (giá trị trả về) và `loaded` (đọc lại từ file) KHÔNG so bằng nhau
    # trực tiếp ở đây: nhóm masked_masked chỉ có 1 ảnh masked/identity trong
    # data giả của test này -> không đủ để tạo genuine pair -> NaN ở `report`
    # (giá trị float thật), được sanitize thành `null` khi ghi file (xem
    # write_report/_nan_to_null) -> JSON đọc lại là None, không phải NaN. Đây
    # là hành vi ĐÚNG THIẾT KẾ (xem test_report.py), không phải lỗi.
    assert loaded["groups"]["masked_masked"]["val"]["achieved_tar"] is None

    # configs/inference.yaml (bản test) được cập nhật threshold thật, không
    # còn null, và gắn đúng config_hash của model vừa eval.
    updated = yaml.safe_load(inference_cfg_path.read_text())
    assert updated["threshold"]["value"] is not None
    assert updated["threshold"]["swept_on_config_hash"] == compute_config_hash(cfg)

    # MLflow: đúng experiment, có metric test_overall_tar (registry.py đọc lại
    # đúng tên này).
    assert fake_mlflow.experiments_set == ["test-eval-exp"]
    assert "test_overall_tar" in fake_mlflow.metrics
    assert fake_mlflow.params["config_hash"] == compute_config_hash(cfg)

    # Registry: chưa có Production (fake trả None) -> luôn đăng ký lần đầu.
    assert len(fake_registry.registered) == 1
    assert fake_registry.registered[0][0] == "test-model"


def test_run_eval_raises_clear_error_when_no_matching_checkpoint(tmp_path, _patched_env):
    images_root = tmp_path / "images"
    train_csv, val_csv, test_csv = (
        tmp_path / "train.csv",
        tmp_path / "val.csv",
        tmp_path / "test.csv",
    )
    _build_manifest(images_root, train_csv, ["id0"])
    _build_manifest(images_root, val_csv, ["id1"])
    _build_manifest(images_root, test_csv, ["id2"])
    checkpoint_dir = tmp_path / "ckpt_empty"  # KHÔNG lưu checkpoint nào vào đây
    cfg = _base_cfg(train_csv, val_csv, checkpoint_dir)

    inference_cfg_path = tmp_path / "inference.yaml"
    _write_inference_cfg(inference_cfg_path, test_csv)
    inference_cfg = load_inference_config(inference_cfg_path)

    with pytest.raises(FileNotFoundError, match="checkpoint"):
        run_eval(
            cfg,
            inference_cfg,
            images_root=images_root,
            checkpoint_dir=checkpoint_dir,
            export_config=REAL_EXPORT_CONFIG,
            report_path=tmp_path / "reports" / "eval_metrics.json",
            inference_config_path=inference_cfg_path,
        )

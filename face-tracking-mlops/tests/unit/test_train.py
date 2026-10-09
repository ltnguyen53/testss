"""Integration test cho src/training/train.py — chạy THẬT (không mock logic
train, chỉ thay backbone bằng bản nhỏ + mlflow bằng fake) trên vài step cực
nhỏ, xác nhận: (1) dây chuyền backbone+head+freeze+optimizer+checkpoint không
lỗi cấu trúc; (2) resume qua 1 LẦN GỌI train() MỚI (mô phỏng đúng kịch bản
thật: process mới, model mới khởi tạo lại từ đầu, không giữ gì trong bộ nhớ
từ lần chạy trước) tìm đúng checkpoint và tiếp tục, không train lại từ đầu.

Bỏ qua sạch nếu torch chưa cài — cùng convention test_head.py/test_model.py.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402
from PIL import Image  # noqa: E402

from src.data.labels import ImageRecord, write_manifest_csv  # noqa: E402
from src.training.train import train  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_EXPORT_CONFIG = REPO_ROOT / "configs" / "export.yaml"  # input_size=112 thật, xem đó
EMBED_DIM = 8
IMG_SIZE = 20  # ảnh gốc trên "disk" — nhỏ hơn 112 vẫn hợp lệ, load_batch tự resize lên


class _TinyBackbone(nn.Module):
    """Cùng pattern _TinyBackbone của test_model.py: có layer4 (unfreeze
    target) + features (BatchNorm1d với weight bị cố định gốc, giống hệt
    iresnet thật — xem model_setup.py điểm 4) để bài test này cũng gián tiếp
    phủ lại đúng bug đã sửa, lần này qua đúng đường train() thật sẽ dùng."""

    def __init__(self) -> None:
        super().__init__()
        self.layer_frozen = nn.Conv2d(3, 4, 3)
        self.layer4 = nn.Conv2d(4, 4, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, EMBED_DIM)
        self.features = nn.BatchNorm1d(EMBED_DIM)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_frozen(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return self.features(x)


class _FakeMlflow:
    """Module giả — inject vào sys.modules["mlflow"] qua monkeypatch, KHÔNG
    cần cài mlflow thật trong môi trường chạy test này."""

    def __init__(self) -> None:
        self.params: dict = {}
        self.metrics: list[tuple[dict, int]] = []
        self.tags: dict = {}
        self.ended_with_status: list[str] = []
        self.experiments_set: list[str] = []

    def set_experiment(self, name: str) -> None:
        self.experiments_set.append(name)

    def start_run(self) -> None:
        pass

    def end_run(self, status: str = "FINISHED") -> None:
        self.ended_with_status.append(status)

    def log_params(self, params: dict) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict, step: int) -> None:
        self.metrics.append((dict(metrics), step))

    def set_tag(self, key: str, value) -> None:
        self.tags[key] = value


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=color).save(path)


def _build_synthetic_dataset(root: Path) -> Path:
    """4 identity x 2 ảnh (1 unmasked + 1 masked) — vừa khít
    sampler.images_per_identity=2, masked_per_identity=1 của _base_cfg, không
    cần fallback synthetic (xem sampler.py:pick_for_identity)."""
    records = []
    for i in range(4):
        identity = f"id{i:03d}"
        _write_image(root / identity / "u.jpg", (10 * i, 0, 0))
        _write_image(root / identity / "m.jpg", (0, 10 * i, 0))
        records.append(ImageRecord(f"{identity}/u.jpg", identity, False, "rmfd_unmasked"))
        records.append(ImageRecord(f"{identity}/m.jpg", identity, True, "rmfd_masked"))
    manifest = root / "train.csv"
    write_manifest_csv(manifest, records)
    return manifest


def _base_cfg(train_csv: Path, checkpoint_dir: Path, epochs: int) -> dict:
    return {
        "seed": 123,
        "data": {"train_csv": str(train_csv), "val_csv": str(train_csv)},
        "backbone": {
            "arch": "unused-monkeypatched",
            "num_features": EMBED_DIM,
            "dropout": 0.0,
            "arcface_torch_dir": "unused-monkeypatched",
            "pretrained_weights": "unused-monkeypatched",
            "weights_source": "unused-monkeypatched",
        },
        "freeze": {
            "trainable_blocks": ["layer4"],
            "train_embedding_fc": False,
            "head_module": "head",
            "train_bn_affine_everywhere": True,
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
        "training": {"optimizer_type": "sgd", "momentum": 0.9, "epochs": epochs, "amp": False},
        "checkpoint": {"save_every_n_steps": 1, "keep_last_n": 3, "dir": str(checkpoint_dir)},
        "mlflow": {"experiment_name": "test-exp", "registered_model_name": "test-model"},
    }


@pytest.fixture
def _patched_env(monkeypatch):
    """Monkeypatch backbone (nhẹ, không cần weight/arcface_torch thật) + mlflow
    (fake, không cần server thật). Trả về fake mlflow instance để assert."""
    monkeypatch.setattr("src.training.train.load_pretrained_backbone", lambda cfg: _TinyBackbone())
    fake_mlflow = _FakeMlflow()
    import sys

    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    return fake_mlflow


def test_train_runs_end_to_end_without_structural_errors(tmp_path, _patched_env):
    train_csv = _build_synthetic_dataset(tmp_path / "images")
    ckpt_dir = tmp_path / "ckpt"
    cfg = _base_cfg(train_csv, ckpt_dir, epochs=1)

    train(cfg, images_root=tmp_path / "images", export_config=REAL_EXPORT_CONFIG)

    # 4 identity, identities_per_batch=2 -> 2 batch/epoch (drop_last=True mặc định)
    assert len(_patched_env.metrics) == 2
    assert _patched_env.ended_with_status == ["FINISHED"]
    assert _patched_env.tags["resumed_from_checkpoint"] == "False"
    # Checkpoint cuối epoch 0 phải tồn tại, step_in_epoch reset về 0 cho epoch KẾ TIẾP
    saved = sorted(p.name for p in ckpt_dir.iterdir())
    assert "ckpt_epoch0001_step00000000" in saved


def test_train_resumes_from_a_fresh_process_instead_of_restarting(tmp_path, _patched_env):
    """Mô phỏng ĐÚNG kịch bản Colab rớt mạng: gọi train() LẦN 2 với CÙNG config
    (cùng config_hash) — đây là 1 lời gọi train() hoàn toàn mới, model/optimizer
    mới khởi tạo lại từ backbone pretrained (fake) trong hàm, KHÔNG giữ gì từ
    lần gọi trước trong bộ nhớ Python — đúng bản chất "process mới" thật sự,
    không phải chỉ gọi tiếp 1 object Python đang sống.
    """
    train_csv = _build_synthetic_dataset(tmp_path / "images")
    ckpt_dir = tmp_path / "ckpt"

    cfg_epoch1 = _base_cfg(train_csv, ckpt_dir, epochs=1)
    train(cfg_epoch1, images_root=tmp_path / "images", export_config=REAL_EXPORT_CONFIG)
    assert _patched_env.tags["resumed_from_checkpoint"] == "False"

    cfg_epoch3 = _base_cfg(train_csv, ckpt_dir, epochs=3)  # CÙNG mọi thứ khác -> cùng config_hash
    train(cfg_epoch3, images_root=tmp_path / "images", export_config=REAL_EXPORT_CONFIG)

    assert _patched_env.tags["resumed_from_checkpoint"] == "True"
    # _patched_env dùng chung cho cả 2 lần gọi train() (mô phỏng 1 experiment
    # xuyên suốt) nên metrics CỘNG DỒN: lần 1 (epoch 0, 2 batch) = 2 step, lần 2
    # (epoch 1->2, KHÔNG chạy lại epoch 0, 2 epoch x 2 batch) = 4 step -> tổng 6.
    # Nếu lần 2 LỠ chạy lại từ epoch 0 thay vì resume, tổng sẽ là 2 + 6 = 8.
    assert len(_patched_env.metrics) == 6
    saved = sorted(p.name for p in ckpt_dir.iterdir())
    assert "ckpt_epoch0003_step00000000" in saved


def test_train_does_not_resume_when_config_hash_differs(tmp_path, _patched_env):
    """Đổi 1 hyperparameter thật (margin) -> config_hash khác -> KHÔNG được
    resume nhầm vào checkpoint của config cũ (CheckpointManager đã test riêng
    ở test_checkpoint.py — đây là test tích hợp qua đúng đường train() thật)."""
    train_csv = _build_synthetic_dataset(tmp_path / "images")
    ckpt_dir = tmp_path / "ckpt"

    cfg_a = _base_cfg(train_csv, ckpt_dir, epochs=1)
    train(cfg_a, images_root=tmp_path / "images", export_config=REAL_EXPORT_CONFIG)

    cfg_b = _base_cfg(train_csv, ckpt_dir, epochs=1)
    cfg_b["head"]["arcface_margin"] = 0.45  # hyperparameter thật khác -> hash khác
    train(cfg_b, images_root=tmp_path / "images", export_config=REAL_EXPORT_CONFIG)

    assert _patched_env.tags["resumed_from_checkpoint"] == "False"


def test_mlflow_run_ends_with_failed_status_on_exception(tmp_path, _patched_env, monkeypatch):
    """Lỗi bất kỳ trong training loop phải kết thúc mlflow run với status FAILED
    (không để run treo ở trạng thái RUNNING mãi) — xem try/except trong train()."""
    train_csv = _build_synthetic_dataset(tmp_path / "images")
    cfg = _base_cfg(train_csv, tmp_path / "ckpt", epochs=1)

    def _boom(*args, **kwargs):
        raise RuntimeError("giả lập lỗi giữa chừng")

    monkeypatch.setattr("src.training.train.load_batch", _boom)

    with pytest.raises(RuntimeError, match="giả lập lỗi giữa chừng"):
        train(cfg, images_root=tmp_path / "images", export_config=REAL_EXPORT_CONFIG)

    assert _patched_env.ended_with_status == ["FAILED"]

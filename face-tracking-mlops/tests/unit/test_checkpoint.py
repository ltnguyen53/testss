"""Unit test cho src/training/checkpoint.py.

Dùng fake save_fn/load_fn (pickle) thay torch.save/torch.load — test toàn bộ
logic file-management (atomic write, rotation, resume decision theo config_hash),
KHÔNG cần torch cài trong CI.
"""

import pickle
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.training.checkpoint import (
    CheckpointManager,
    CheckpointMeta,
    compute_config_hash,
    get_git_commit_hash,
)


def _pickle_save(obj, path: Path) -> None:
    path.write_bytes(pickle.dumps(obj))


def _pickle_load(path: Path):
    return pickle.loads(path.read_bytes())


def _make_manager(tmp_path: Path, keep_last_n: int = 2) -> CheckpointManager:
    return CheckpointManager(
        tmp_path / "checkpoints",
        keep_last_n=keep_last_n,
        save_fn=_pickle_save,
        load_fn=_pickle_load,
    )


def _meta(config_hash: str, epoch: int, step: int) -> CheckpointMeta:
    return CheckpointMeta(
        timestamp=datetime.now(timezone.utc).isoformat(),
        config_hash=config_hash,
        git_commit_hash="abc123",
        epoch=epoch,
        step=step,
    )


def test_compute_config_hash_ignores_key_order():
    h1 = compute_config_hash({"a": 1, "b": {"c": 2}})
    h2 = compute_config_hash({"b": {"c": 2}, "a": 1})
    assert h1 == h2


def test_compute_config_hash_changes_with_value():
    h1 = compute_config_hash({"a": 1})
    h2 = compute_config_hash({"a": 2})
    assert h1 != h2


def test_compute_config_hash_ignores_operational_sections(tmp_path):
    """Regression test cho fix 2026-09-29: đổi checkpoint.dir/save_every_n_steps
    hay mlflow.experiment_name KHÔNG được đổi hash — các section này thuần vận
    hành/logging, không ảnh hưởng training semantics. Đổi 1 hyperparameter thật
    (vd optimizer.head_lr) THÌ PHẢI đổi hash."""
    base = {
        "seed": 42,
        "optimizer": {"head_lr": 1e-3},
        "checkpoint": {"dir": "/content/drive/MyDrive/ckpt_v1", "save_every_n_steps": 500},
        "mlflow": {"experiment_name": "run-v1"},
    }
    only_checkpoint_dir_changed = {
        **base,
        "checkpoint": {"dir": str(tmp_path / "somewhere_else"), "save_every_n_steps": 9999},
    }
    only_mlflow_changed = {**base, "mlflow": {"experiment_name": "totally-different-name"}}
    real_hyperparameter_changed = {**base, "optimizer": {"head_lr": 5e-4}}

    assert compute_config_hash(base) == compute_config_hash(only_checkpoint_dir_changed)
    assert compute_config_hash(base) == compute_config_hash(only_mlflow_changed)
    assert compute_config_hash(base) != compute_config_hash(real_hyperparameter_changed)


def test_compute_config_hash_ignores_training_epochs_only(tmp_path):
    """Regression test cho bug THẬT phát hiện bằng test tích hợp
    (tests/unit/test_train.py): training.epochs chỉ là tiêu chí dừng vòng lặp,
    KHÔNG ảnh hưởng gradient — đổi epochs (vd để train thêm sau khi resume)
    KHÔNG được đổi hash, nhưng optimizer_type/momentum trong CÙNG section
    training vẫn PHẢI đổi hash (loại trừ phải chính xác ở field, không phải cả
    section — nếu loại nhầm cả section, bug này sẽ không bị phát hiện)."""
    base = {"training": {"optimizer_type": "sgd", "momentum": 0.9, "epochs": 1, "amp": False}}
    only_epochs_changed = {**base, "training": {**base["training"], "epochs": 30}}
    momentum_changed = {**base, "training": {**base["training"], "momentum": 0.5}}

    assert compute_config_hash(base) == compute_config_hash(only_epochs_changed)
    assert compute_config_hash(base) != compute_config_hash(momentum_changed)


def test_get_git_commit_hash_falls_back_when_not_a_repo(tmp_path):
    assert get_git_commit_hash(tmp_path) == "unknown"


def test_save_load_roundtrip(tmp_path):
    mgr = _make_manager(tmp_path)
    state = {"model": {"w": [1, 2, 3]}, "optimizer": {"lr": 0.001}}
    meta = _meta("cfgA", epoch=0, step=100)

    saved_dir = mgr.save(state, meta)

    assert (saved_dir / "state.ckpt").exists()
    assert (saved_dir / "meta.json").exists()
    loaded_state, loaded_meta = mgr.load(saved_dir)
    assert loaded_state == state
    assert loaded_meta == meta


def test_save_leaves_no_tmp_dir_behind(tmp_path):
    mgr = _make_manager(tmp_path)
    mgr.save({"x": 1}, _meta("cfgA", epoch=0, step=1))

    leftover = [p for p in (tmp_path / "checkpoints").iterdir() if p.name.startswith(".tmp_")]
    assert leftover == []


def test_rotation_keeps_only_last_n(tmp_path):
    mgr = _make_manager(tmp_path, keep_last_n=2)
    for step in (100, 200, 300, 400):
        mgr.save({"step": step}, _meta("cfgA", epoch=0, step=step))

    remaining = mgr._list_checkpoints()  # list[(dir, meta)] — xem docstring
    assert len(remaining) == 2
    remaining_steps = sorted(int(d.name.split("step")[1]) for d, _meta in remaining)
    assert remaining_steps == [300, 400]


def test_rotation_is_scoped_per_config_hash_not_global(tmp_path):
    """Regression test cho bug đã sửa (2026-09-28): trước đây _rotate() xoay vòng
    TOÀN BỘ checkpoint trong thư mục theo tên (mọi config_hash gộp chung), nên 1
    run mới (config khác, epoch bắt đầu từ 0) có thể bị coi là "cũ hơn" các
    checkpoint epoch cao của 1 run khác (hash khác) đang có sẵn trên Drive —
    checkpoint VỪA LƯU của run mới bị rotate xoá ngay lập tức.

    Kịch bản tái hiện đúng như đã gặp: run A (keep_last_n=2) đã train tới epoch
    39-40 (hai checkpoint cuối). Sau đó đổi 1 hyperparameter (=> config_hash mới,
    cfgB) và bắt đầu train lại — checkpoint đầu tiên của cfgB (epoch=0) được lưu
    VÀO CÙNG checkpoint_dir (mô phỏng đúng việc dùng lại 1 thư mục Drive).
    """
    mgr = _make_manager(tmp_path, keep_last_n=2)
    for epoch, step in ((39, 3900), (40, 4000)):
        mgr.save({"run": "A"}, _meta("cfgA", epoch=epoch, step=step))

    # Trước khi B lưu: chưa có gì để resume cho cfgB.
    assert mgr.find_resumable("cfgB") is None

    mgr.save({"run": "B"}, _meta("cfgB", epoch=0, step=500))

    # Checkpoint đầu tiên của B phải CÒN — đây chính là hành vi bị lỗi trước khi sửa
    # (trước đây bị rotate xoá ngay trong lệnh save() ở trên).
    resumable_b = mgr.find_resumable("cfgB")
    assert resumable_b is not None, "checkpoint đầu tiên của cfgB bị rotate xoá nhầm"
    assert "epoch0000_step00000500" in resumable_b.name

    # 2 checkpoint của cfgA không bị đụng tới bởi rotate của cfgB.
    assert mgr.find_resumable("cfgA") is not None
    assert len(mgr._list_checkpoints(config_hash="cfgA")) == 2
    assert len(mgr._list_checkpoints(config_hash="cfgB")) == 1


def test_find_resumable_matches_only_same_config_hash(tmp_path):
    mgr = _make_manager(tmp_path, keep_last_n=10)
    mgr.save({"step": 100}, _meta("cfgA", epoch=0, step=100))
    mgr.save({"step": 200}, _meta("cfgB", epoch=0, step=200))

    found_a = mgr.find_resumable("cfgA")
    found_b = mgr.find_resumable("cfgB")
    found_none = mgr.find_resumable("cfgC_khong_ton_tai")

    assert found_a is not None and "step00000100" in found_a.name
    assert found_b is not None and "step00000200" in found_b.name
    assert found_none is None


def test_find_resumable_returns_none_when_no_checkpoints(tmp_path):
    mgr = _make_manager(tmp_path)
    assert mgr.find_resumable("anything") is None


def test_load_raises_on_missing_meta(tmp_path):
    mgr = _make_manager(tmp_path)
    broken_dir = tmp_path / "checkpoints" / "ckpt_broken"
    broken_dir.mkdir(parents=True)
    (broken_dir / "state.ckpt").write_bytes(b"whatever")
    # Không có meta.json -> load phải raise, không được âm thầm trả state rác
    with pytest.raises(ValueError):
        mgr.load(broken_dir)

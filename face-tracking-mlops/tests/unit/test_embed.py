"""Unit test cho src/evaluation/embed.py bằng torch THẬT (model.embed() là
nn.Module thật). Bỏ qua sạch nếu torch chưa cài — cùng convention
test_head.py/test_model.py/test_train.py."""

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402
from PIL import Image  # noqa: E402

from src.data.labels import ImageRecord, write_manifest_csv  # noqa: E402
from src.data.normalize import PreprocessingContract  # noqa: E402
from src.evaluation.embed import embed_manifest  # noqa: E402
from src.training.head import ArcFaceHead  # noqa: E402
from src.training.model import FaceModel  # noqa: E402

CONTRACT = PreprocessingContract(
    input_size=8, channel_order="RGB", layout="CHW", scale=255.0, mean=0.5, std=0.5
)


class _TinyBackbone(nn.Module):
    def __init__(self, embed_dim: int = 6) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(self.conv(x)).flatten(1))


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), color=color).save(path)


def test_embed_manifest_returns_l2_normalized_vectors_matching_manifest_order(tmp_path):
    model = FaceModel(_TinyBackbone(), ArcFaceHead(6, num_classes=3, margin=0.3, scale=32.0))
    records = [
        ImageRecord("id001/a.jpg", "id001", False, "rmfd_unmasked"),
        ImageRecord("id001/b.jpg", "id001", True, "rmfd_masked"),
        ImageRecord("id002/a.jpg", "id002", False, "rmfd_unmasked"),
    ]
    _write_image(tmp_path / "id001/a.jpg", (255, 0, 0))
    _write_image(tmp_path / "id001/b.jpg", (0, 255, 0))
    _write_image(tmp_path / "id002/a.jpg", (0, 0, 255))
    manifest = tmp_path / "manifest.csv"
    write_manifest_csv(manifest, records)

    out = embed_manifest(model, manifest, CONTRACT, tmp_path, torch.device("cpu"))

    assert len(out) == 3
    assert [r.identity_id for r in out] == ["id001", "id001", "id002"]
    assert [r.source for r in out] == ["rmfd_unmasked", "rmfd_masked", "rmfd_unmasked"]
    for rec in out:
        norm = math.sqrt(sum(x * x for x in rec.embedding))
        assert abs(norm - 1.0) < 1e-5


def test_embed_manifest_restores_training_mode_after_running(tmp_path):
    """embed_manifest tạm chuyển model sang eval() để tắt BN update từ batch
    eval — phải trả model VỀ ĐÚNG trạng thái training trước đó sau khi xong,
    không làm lộ side-effect ra ngoài caller."""
    model = FaceModel(_TinyBackbone(), ArcFaceHead(6, num_classes=2, margin=0.3, scale=32.0))
    model.train()
    records = [ImageRecord("id001/a.jpg", "id001", False, "rmfd_unmasked")]
    _write_image(tmp_path / "id001/a.jpg", (10, 20, 30))
    manifest = tmp_path / "manifest.csv"
    write_manifest_csv(manifest, records)

    embed_manifest(model, manifest, CONTRACT, tmp_path, torch.device("cpu"))

    assert model.training is True


def test_embed_manifest_batches_do_not_change_result_vs_single_batch(tmp_path):
    """batch_size nhỏ hơn tổng số ảnh (nhiều batch) phải cho kết quả giống hệt
    batch_size đủ lớn (1 batch) — không có lỗi cắt batch làm sai embedding."""
    model = FaceModel(_TinyBackbone(), ArcFaceHead(6, num_classes=4, margin=0.3, scale=32.0))
    model.eval()
    records = []
    for i in range(5):
        _write_image(tmp_path / f"id{i:03d}/a.jpg", (i * 10, i * 20, i * 5))
        records.append(ImageRecord(f"id{i:03d}/a.jpg", f"id{i:03d}", False, "rmfd_unmasked"))
    manifest = tmp_path / "manifest.csv"
    write_manifest_csv(manifest, records)

    out_small_batch = embed_manifest(
        model, manifest, CONTRACT, tmp_path, torch.device("cpu"), batch_size=2
    )
    out_one_batch = embed_manifest(
        model, manifest, CONTRACT, tmp_path, torch.device("cpu"), batch_size=100
    )

    for a, b in zip(out_small_batch, out_one_batch, strict=True):
        assert a.embedding == pytest.approx(b.embedding, abs=1e-6)

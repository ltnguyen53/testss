"""Unit test cho src/training/dataset.py. Cần torch (stack tensor) + PIL (đã có
trong mọi môi trường qua requirements/train.in) — bỏ qua sạch nếu torch chưa
cài, cùng convention test_head.py/test_model.py."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402

from src.data.labels import ImageRecord  # noqa: E402
from src.data.normalize import PreprocessingContract  # noqa: E402
from src.training.dataset import load_batch  # noqa: E402

CONTRACT = PreprocessingContract(
    input_size=8, channel_order="RGB", layout="CHW", scale=255.0, mean=0.5, std=0.5
)


def _write_solid_color_image(path: Path, color: tuple[int, int, int], size: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color=color).save(path)


def test_load_batch_returns_correctly_shaped_normalized_tensors(tmp_path):
    _write_solid_color_image(tmp_path / "raw/id001/a.jpg", (255, 0, 0))
    _write_solid_color_image(tmp_path / "raw/id001/b.jpg", (0, 255, 0))
    _write_solid_color_image(tmp_path / "raw/id002/a.jpg", (0, 0, 255))
    records = [
        ImageRecord("raw/id001/a.jpg", "id001", False, "rmfd_unmasked"),
        ImageRecord("raw/id001/b.jpg", "id001", True, "rmfd_masked"),
        ImageRecord("raw/id002/a.jpg", "id002", False, "rmfd_unmasked"),
    ]
    identity_to_class = {"id001": 0, "id002": 1}

    images, labels = load_batch(records, identity_to_class, CONTRACT, images_root=tmp_path)

    assert images.shape == (3, 3, 8, 8)  # (B, C, H, W) đúng contract input_size=8
    assert images.dtype == torch.float32
    assert torch.equal(labels, torch.tensor([0, 0, 1]))
    # normalize_rgb_uint8: (x/255 - 0.5)/0.5 -> range lý thuyết [-1, 1]
    assert images.min() >= -1.0 - 1e-4
    assert images.max() <= 1.0 + 1e-4


def test_load_batch_grayscale_and_rgba_images_convert_to_rgb(tmp_path):
    """Ảnh thật ngoài đời không phải lúc nào cũng RGB thuần (RMFD có thể lẫn
    ảnh grayscale/RGBA) — convert("RGB") phải xử lý được, không crash."""
    (tmp_path / "raw/id003").mkdir(parents=True)
    Image.new("L", (20, 20), color=128).save(tmp_path / "raw/id003/gray.jpg")
    Image.new("RGBA", (20, 20), color=(10, 20, 30, 255)).save(tmp_path / "raw/id003/rgba.png")
    records = [
        ImageRecord("raw/id003/gray.jpg", "id003", False, "rmfd_unmasked"),
        ImageRecord("raw/id003/rgba.png", "id003", False, "rmfd_unmasked"),
    ]

    images, labels = load_batch(records, {"id003": 0}, CONTRACT, images_root=tmp_path)

    assert images.shape == (2, 3, 8, 8)


def test_load_batch_raises_keyerror_for_identity_not_in_split():
    """identity_to_class chỉ chứa identity của 1 split (SPEC 3.2: head size =
    train-split only) — lẫn identity của split khác vào phải raise ngay, không
    âm thầm gán nhãn sai."""
    records = [ImageRecord("x.jpg", "unknown_identity", False, "rmfd_unmasked")]
    with pytest.raises(KeyError):
        load_batch(records, {"id001": 0}, CONTRACT)


def test_load_batch_preserves_record_order_not_grouped_by_identity(tmp_path):
    _write_solid_color_image(tmp_path / "id_a/1.jpg", (1, 1, 1))
    _write_solid_color_image(tmp_path / "id_b/1.jpg", (2, 2, 2))
    records = [
        ImageRecord("id_a/1.jpg", "id_a", False, "rmfd_unmasked"),
        ImageRecord("id_b/1.jpg", "id_b", False, "rmfd_unmasked"),
        ImageRecord("id_a/1.jpg", "id_a", False, "rmfd_unmasked"),
    ]

    _, labels = load_batch(records, {"id_a": 0, "id_b": 1}, CONTRACT, images_root=tmp_path)

    assert torch.equal(labels, torch.tensor([0, 1, 0]))  # đúng thứ tự input, không bị gom nhóm

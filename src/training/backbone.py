"""Dựng + load backbone iresnet của arcface_torch (SPEC v2.1 mục 2.1/3.1).

Không copy code backbone vào repo: dùng đúng module `backbones.get_model` của
`deepinsight/insightface/recognition/arcface_torch` (sparse clone vào third_party/, xem
README) — cùng cách gọi với `inference.py`/`torch2onnx.py` chính thức:
`get_model(name, dropout=..., fp16=False, num_features=...)`.

Phần dựng module (build_backbone) test được bằng package `backbones` giả; phần load
.pth thật (load_pretrained_backbone) CHƯA verify — sandbox không có torch/weight/repo
thật. Chưa biết: `import backbones` có kéo theo dependency ngoài (vd `timm` cho ViT) hay
không — nếu báo thiếu module, thêm vào requirements/train.in.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from src.training.weights import load_weights_source, verify_weights


def add_arcface_torch_to_path(arcface_torch_dir: Path) -> None:
    d = Path(arcface_torch_dir).resolve()
    if not (d / "backbones").is_dir():
        raise FileNotFoundError(
            f"Không thấy {d}/backbones — sparse clone insightface vào third_party/ (xem README)"
        )
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))


def build_backbone(
    arch: str, arcface_torch_dir: Path, *, num_features: int = 512, dropout: float = 0.0
) -> Any:
    add_arcface_torch_to_path(arcface_torch_dir)
    from backbones import get_model  # noqa: PLC0415 — import trễ: chỉ có sau khi thêm path

    return get_model(arch, dropout=dropout, fp16=False, num_features=num_features)


def load_pretrained_backbone(cfg: dict[str, Any]) -> Any:
    """CHƯA VERIFY chạy thật. Kiểm SHA256 trước, rồi load_state_dict(strict=True) —
    strict để sai kiến trúc/sai file fail ngay thay vì âm thầm bỏ qua key lệch."""
    import torch  # noqa: PLC0415

    b = cfg["backbone"]
    source = load_weights_source(Path(b["weights_source"]))
    verify_weights(Path(b["pretrained_weights"]), source, expected_arch=b["arch"])
    model = build_backbone(
        b["arch"],
        Path(b["arcface_torch_dir"]),
        num_features=b["num_features"],
        dropout=b["dropout"],
    )
    state = torch.load(b["pretrained_weights"], map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model

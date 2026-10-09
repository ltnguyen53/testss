"""Chuẩn hoá ảnh cho ArcFace — hợp đồng train <-> browser (SPEC v2.1 mục 6, Module 10.2).

Một hàm duy nhất cho mọi nơi chạy Python (train, eval, export parity check, golden vector
cho frontend). Số liệu đọc từ configs/export.yaml, không hardcode. Công thức khớp
`inference.py` chính thức của arcface_torch: img.div_(255).sub_(0.5).div_(0.5), sau BGR->RGB
và transpose HWC->CHW.

Phạm vi: CHỈ chuẩn hoá. Detect/crop/align/resize nằm ngoài (alignment chưa chốt — xem yaml).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class PreprocessingContract:
    input_size: int
    channel_order: str
    layout: str
    scale: float
    mean: float
    std: float


def load_contract(export_yaml: Path) -> PreprocessingContract:
    cfg = yaml.safe_load(Path(export_yaml).read_text(encoding="utf-8"))["preprocessing"]
    contract = PreprocessingContract(
        input_size=int(cfg["input_size"]),
        channel_order=str(cfg["channel_order"]),
        layout=str(cfg["layout"]),
        scale=float(cfg["scale"]),
        mean=float(cfg["mean"]),
        std=float(cfg["std"]),
    )
    if contract.channel_order != "RGB" or contract.layout != "CHW":
        raise ValueError("normalize_rgb_uint8 chỉ hỗ trợ RGB + CHW (đúng hợp đồng arcface_torch)")
    if contract.std <= 0 or contract.scale <= 0:
        raise ValueError("scale và std phải > 0")
    return contract


def bgr_to_rgb(img_hwc: np.ndarray) -> np.ndarray:
    """cv2.imread trả BGR; model cần RGB.

    Quên bước này = lỗi im lặng (accuracy tụt, không crash).
    """
    return np.ascontiguousarray(img_hwc[..., ::-1])


def normalize_rgb_uint8(img_hwc: np.ndarray, contract: PreprocessingContract) -> np.ndarray:
    """Ảnh RGB uint8 HxWx3 (đã crop/align, đúng input_size) -> float32 3xHxW."""
    size = contract.input_size
    if img_hwc.dtype != np.uint8 or img_hwc.shape != (size, size, 3):
        raise ValueError(
            f"cần ảnh uint8 shape ({size}, {size}, 3), nhận dtype={img_hwc.dtype} "
            f"shape={img_hwc.shape} — resize/align làm ở bước trước, không làm ngầm ở đây"
        )
    x = img_hwc.astype(np.float32) / contract.scale
    x = (x - contract.mean) / contract.std
    return np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32)

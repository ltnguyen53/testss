"""Quản lý nguồn + toàn vẹn của weight pretrained (SPEC v2.1 mục 2.1 "Pretrained weights").

Weight tải tay (Baidu/OneDrive) nên không có gì đảm bảo file trên Colab hôm nay là file
đã dùng hôm qua. SOURCE.yaml ghi nguồn + SHA256; mọi lần load đều kiểm SHA256 (fail loudly).
Không phụ thuộc torch.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class WeightsSource:
    model_name: str
    architecture: str
    training_dataset: str
    download_date: str
    sha256: str
    upstream_repo: str
    upstream_commit: str
    license_note: str


def load_weights_source(path: Path) -> WeightsSource:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    names = [f.name for f in fields(WeightsSource)]
    missing = [n for n in names if raw.get(n) in (None, "")]
    if missing:
        raise ValueError(f"{path}: còn field chưa điền: {missing}")
    # YAML có thể parse ngày thành datetime.date -> ép về str để dataclass nhất quán
    return WeightsSource(**{n: str(raw[n]) for n in names})


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def verify_weights(weights_path: Path, source: WeightsSource, expected_arch: str) -> None:
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Không thấy {weights_path} — tải tay backbone.pth rồi "
            "`dvc pull`/`dvc add` (xem README)"
        )
    if source.architecture != expected_arch:
        raise ValueError(
            f"SOURCE.yaml ghi architecture={source.architecture} nhưng config dùng {expected_arch}"
        )
    actual = sha256_of_file(weights_path)
    if actual.lower() != source.sha256.lower():
        raise ValueError(
            f"SHA256 của {weights_path} ({actual}) không khớp SOURCE.yaml ({source.sha256}) — "
            "sai file hoặc file hỏng/bị thay"
        )

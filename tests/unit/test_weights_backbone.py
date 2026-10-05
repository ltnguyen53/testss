"""Unit test cho src/training/weights.py và phần dựng module của backbone.py.

Không có torch/weight thật: test SHA256 + SOURCE.yaml bằng file giả, test build_backbone
bằng package `backbones` giả (đúng chữ ký get_model(name, dropout, fp16, num_features)).
"""

import hashlib
import sys
from pathlib import Path

import pytest

from src.training.backbone import add_arcface_torch_to_path, build_backbone
from src.training.weights import load_weights_source, sha256_of_file, verify_weights

SOURCE_PATH = Path(__file__).resolve().parents[2] / "models" / "pretrained" / "SOURCE.yaml"

FULL = {
    "model_name": "ms1mv3_arcface_r50_fp16",
    "architecture": "r50",
    "training_dataset": "MS1MV3",
    "download_date": "2026-09-28",
    "upstream_repo": "https://github.com/deepinsight/insightface",
    "upstream_commit": "abc1234",
    "license_note": "non-commercial research only",
}


def _write_source(path, sha256, **override):
    data = {**FULL, "sha256": sha256, **override}
    lines = [f"{k}: {v}" for k, v in data.items() if v is not None]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_shipped_source_template_is_incomplete_on_purpose():
    # Template chưa điền -> load phải raise (nhắc người dùng điền), không âm thầm chạy.
    with pytest.raises(ValueError) as exc:
        load_weights_source(SOURCE_PATH)
    assert "sha256" in str(exc.value)


def test_sha256_of_file_matches_hashlib(tmp_path):
    f = tmp_path / "w.pth"
    f.write_bytes(b"abc" * 100_000)
    assert sha256_of_file(f) == hashlib.sha256(b"abc" * 100_000).hexdigest()


def test_verify_weights_ok_and_mismatch(tmp_path):
    f = tmp_path / "backbone.pth"
    f.write_bytes(b"weights")
    good = hashlib.sha256(b"weights").hexdigest()
    src_path = tmp_path / "SOURCE.yaml"

    _write_source(src_path, good)
    verify_weights(f, load_weights_source(src_path), expected_arch="r50")  # không raise

    _write_source(src_path, "0" * 64)
    with pytest.raises(ValueError):
        verify_weights(f, load_weights_source(src_path), expected_arch="r50")


def test_verify_weights_rejects_arch_mismatch_and_missing_file(tmp_path):
    f = tmp_path / "backbone.pth"
    f.write_bytes(b"weights")
    src_path = tmp_path / "SOURCE.yaml"
    _write_source(src_path, hashlib.sha256(b"weights").hexdigest())
    source = load_weights_source(src_path)

    with pytest.raises(ValueError):
        verify_weights(f, source, expected_arch="r100")
    with pytest.raises(FileNotFoundError):
        verify_weights(tmp_path / "khong_ton_tai.pth", source, expected_arch="r50")


def test_load_weights_source_parses_yaml_date_as_str(tmp_path):
    src_path = tmp_path / "SOURCE.yaml"
    _write_source(src_path, "a" * 64)  # 2026-09-28 không có nháy -> YAML parse thành date
    assert load_weights_source(src_path).download_date == "2026-09-28"


def test_add_arcface_torch_to_path_requires_backbones_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        add_arcface_torch_to_path(tmp_path)


def test_build_backbone_calls_get_model_with_official_signature(tmp_path):
    pkg = tmp_path / "backbones"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "def get_model(name, **kwargs):\n    return ('fake_backbone', name, kwargs)\n",
        encoding="utf-8",
    )
    sys.modules.pop("backbones", None)
    try:
        result = build_backbone("r50", tmp_path, num_features=512, dropout=0.0)
    finally:
        sys.modules.pop("backbones", None)
        if str(tmp_path.resolve()) in sys.path:
            sys.path.remove(str(tmp_path.resolve()))

    assert result == ("fake_backbone", "r50", {"dropout": 0.0, "fp16": False, "num_features": 512})

"""Unit test cho src/evaluation/report.py. Thuần Python."""

import json

from src.evaluation.report import build_report, write_report
from src.evaluation.threshold import ThresholdResult


def _val_result(threshold: float = 0.42) -> ThresholdResult:
    return ThresholdResult(
        threshold=threshold,
        target_far=0.01,
        achieved_far=0.008,
        achieved_tar=0.91,
        num_genuine=100,
        num_impostor=5000,
    )


def test_build_report_includes_val_and_test_per_group():
    report = build_report(
        config_hash="abc123",
        locked_threshold=0.42,
        target_far=0.01,
        val_results={"overall": _val_result(), "masked_masked": _val_result()},
        test_results={"overall": (0.88, 0.009), "masked_masked": (0.75, 0.012)},
    )

    assert report["config_hash"] == "abc123"
    assert report["locked_threshold"] == 0.42
    assert report["groups"]["overall"]["test"] == {"tar": 0.88, "far": 0.009}
    assert report["groups"]["masked_masked"]["val"]["threshold"] == 0.42


def test_build_report_fills_nan_when_group_missing_from_test_results():
    """Nhóm có trong val_results nhưng KHÔNG có trong test_results (vd test
    split không có đủ pair cho nhóm đó) -> NaN, không raise KeyError."""
    report = build_report(
        config_hash="abc123",
        locked_threshold=0.42,
        target_far=0.01,
        val_results={"masked_masked": _val_result()},
        test_results={},
    )

    import math

    assert math.isnan(report["groups"]["masked_masked"]["test"]["tar"])


def test_write_report_serializes_nan_as_null_not_literal_nan(tmp_path):
    """json.dumps mặc định ghi NaN thành literal `NaN` (không phải JSON chuẩn)
    — write_report phải đổi thành `null` để file đọc được bằng parser JSON
    strict bất kỳ (chưa kiểm chứng `dvc metrics diff` có tolerant hay không,
    nên không dựa vào đó)."""
    report = build_report(
        config_hash="abc123",
        locked_threshold=0.42,
        target_far=0.01,
        val_results={"masked_masked": _val_result()},
        test_results={},  # -> NaN cho group này
    )
    out_path = tmp_path / "eval_metrics.json"

    write_report(report, out_path)

    raw_text = out_path.read_text()
    assert "NaN" not in raw_text
    loaded = json.loads(raw_text)
    assert loaded["groups"]["masked_masked"]["test"]["tar"] is None


def test_write_report_creates_parent_dirs_and_valid_json(tmp_path):
    report = build_report(
        config_hash="abc123",
        locked_threshold=0.42,
        target_far=0.01,
        val_results={"overall": _val_result()},
        test_results={"overall": (0.9, 0.008)},
    )
    out_path = tmp_path / "nested" / "dir" / "eval_metrics.json"

    write_report(report, out_path)

    assert out_path.exists()
    loaded = json.loads(out_path.read_text())
    assert loaded["config_hash"] == "abc123"

"""Sinh reports/eval_metrics.json — SPEC v2.1 mục 2.2 (`dvc metrics diff` dùng
file này để so sánh giữa các lần fine-tune), mục 4.1 (report tách theo nhóm
occlusion), mục 5 (metric đi kèm model version trong MLflow).

Thuần Python (không torch) — nhận `ThresholdResult`/`(tar, far)` đã tính sẵn
từ threshold.py, không tự tính toán gì thêm ở đây, chỉ đóng gói + ghi file.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.evaluation.threshold import ThresholdResult


def build_report(
    *,
    config_hash: str,
    locked_threshold: float,
    target_far: float,
    val_results: dict[str, ThresholdResult],
    test_results: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    """`val_results`: tên nhóm ("overall", "unmasked_unmasked", "masked_masked",
    "masked_unmasked") -> ThresholdResult đo trên VAL bằng `locked_threshold`
    (nhóm "overall" là nơi threshold THẬT SỰ được sweep ra — SPEC 4.2: 1 ngưỡng
    toàn cục, không phải mỗi nhóm 1 ngưỡng riêng; các nhóm con chỉ dùng để BÁO
    CÁO, re-validate bằng CHÍNH threshold của "overall").
    `test_results`: cùng tên nhóm -> (tar, far) đo trên TEST bằng
    `locked_threshold` (SPEC 4.2: re-validate, KHÔNG tinh chỉnh lại).
    """
    groups: dict[str, Any] = {}
    for group_name, val_result in val_results.items():
        test_tar, test_far = test_results.get(group_name, (float("nan"), float("nan")))
        groups[group_name] = {
            "val": asdict(val_result),
            "test": {"tar": test_tar, "far": test_far},
        }
    return {
        "config_hash": config_hash,
        "locked_threshold": locked_threshold,
        "target_far": target_far,
        "groups": groups,
    }


def _nan_to_null(value: Any) -> Any:
    """json.dumps mặc định (`allow_nan=True`) ghi NaN thành literal `NaN` —
    KHÔNG phải JSON chuẩn (parser strict sẽ lỗi, và chưa kiểm chứng `dvc
    metrics diff` có chấp nhận hay không). Đổi NaN -> null (JSON chuẩn hiểu
    được ở mọi nơi) trước khi dump, thay vì tin `allow_nan=True` là đủ an
    toàn."""
    if isinstance(value, dict):
        return {k: _nan_to_null(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_nan_to_null(v) for v in value]
    if isinstance(value, float) and value != value:  # NaN != NaN, cách check không cần import math
        return None
    return value


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = _nan_to_null(report)
    path.write_text(json.dumps(sanitized, indent=2, sort_keys=True), encoding="utf-8")

"""Quyết định đăng ký model vào MLflow Model Registry — SPEC v2.1 mục 5.

Lý do file này thuộc Phase 3, KHÔNG phải Phase 2: SPEC 5 nói "run tốt hơn bản
đang deploy (so metric trên eval set cố định) -> mlflow.register_model...".
Phép so sánh "tốt hơn" cần SỐ ĐO THẬT (TAR@FAR) — thứ chỉ tồn tại sau khi
threshold.py chạy xong, không phải lúc train.py vừa kết thúc (train.py chỉ có
training loss, KHÔNG phải verification metric). Vì vậy quyết định đăng ký nằm
ở Phase 3 (sau eval), không nằm trong train.py.

`should_register`: logic thuần, test được không cần MLflow/torch thật.
`register_if_better`: lớp vỏ mỏng gọi mlflow thật (lazy import, cùng nguyên
tắc train.py) — KHÔNG tự promote lên Production (SPEC 5: "Review thủ công
(semi-auto): kiểm tra metric + demo thử -> đạt thì chuyển Production" — luôn
cần người xem lại trước khi promote, không tự động).
"""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


def should_register(new_metric: float, current_production_metric: float | None) -> bool:
    """`new_metric`/`current_production_metric`: TAR tại target FAR đã khoá,
    đo trên TEST (SPEC 4.2 — không dùng số đo trên val để quyết định, vì val
    đã được dùng để chọn threshold, dùng tiếp để quyết định registry là
    double-dipping trên cùng 1 tập).

    `current_production_metric=None` nghĩa là CHƯA có model nào ở Production
    (lần đăng ký đầu tiên) -> luôn đăng ký. NaN (nhóm "overall" không tính
    được vì thiếu pair — không nên xảy ra với nhóm "overall" nhưng vẫn phòng)
    -> KHÔNG đăng ký (không so sánh được an toàn với NaN).
    """
    if new_metric != new_metric:  # NaN check không cần import math
        logger.warning("new_metric là NaN — không đăng ký (không so sánh được an toàn)")
        return False
    if current_production_metric is None:
        return True
    return new_metric > current_production_metric


class RegistryClient(Protocol):
    def get_production_metric(self, registered_model_name: str) -> float | None: ...
    def register_model(
        self, registered_model_name: str, run_id: str, config_hash: str, metric: float
    ) -> str: ...  # trả về version string mới


def register_if_better(
    client: RegistryClient,
    *,
    registered_model_name: str,
    run_id: str,
    config_hash: str,
    new_test_metric: float,
) -> str | None:
    """Trả version mới nếu đăng ký, None nếu không đăng ký (metric không tốt
    hơn). KHÔNG set stage — version mới luôn vào stage mặc định (None/"None"
    của MLflow, tương đương "chưa review"); người dùng tự chuyển sang Staging
    rồi Production sau khi xem lại (SPEC 5: review thủ công bắt buộc)."""
    current = client.get_production_metric(registered_model_name)
    if not should_register(new_test_metric, current):
        logger.info(
            "Không đăng ký: metric mới (%.4f) không tốt hơn Production hiện tại (%s)",
            new_test_metric,
            current,
        )
        return None
    version = client.register_model(registered_model_name, run_id, config_hash, new_test_metric)
    logger.info(
        "Đã đăng ký version %s (metric=%.4f, Production trước đó=%s)",
        version,
        new_test_metric,
        current,
    )
    return version


class _RealMlflowRegistryClient:
    """Adapter mỏng gọi mlflow thật — lazy import (cùng nguyên tắc train.py)."""

    def get_production_metric(self, registered_model_name: str) -> float | None:
        import mlflow  # noqa: PLC0415
        from mlflow.tracking import MlflowClient  # noqa: PLC0415

        client = MlflowClient()
        try:
            versions = client.get_latest_versions(registered_model_name, stages=["Production"])
        except Exception:  # noqa: BLE001 — model chưa từng đăng ký lần nào -> coi như chưa có Production
            return None
        if not versions:
            return None
        run = mlflow.get_run(versions[0].run_id)
        return run.data.metrics.get("test_overall_tar")

    def register_model(
        self, registered_model_name: str, run_id: str, config_hash: str, metric: float
    ) -> str:
        import mlflow  # noqa: PLC0415

        result = mlflow.register_model(f"runs:/{run_id}/model", registered_model_name)
        return result.version


def build_real_registry_client() -> RegistryClient:
    return _RealMlflowRegistryClient()

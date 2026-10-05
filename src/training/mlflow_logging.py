"""MLflow logging — SPEC v2.1 mục 5 (Tracking + Model Registry).

Tách khỏi mlflow thật qua injection (cùng nguyên tắc với checkpoint.py tách
torch qua save_fn/load_fn): `TrainRunLogger` nhận 1 `client` bất kỳ khớp
`MlflowClient` Protocol — test được toàn bộ logic (cái gì log, dưới tên/step
nào) bằng fake client ghi lại lời gọi, KHÔNG cần MLflow server thật chạy
trong CI. Chỗ gọi mlflow thật (mlflow.start_run(), mlflow.register_model())
nằm ở train.py (nơi thật sự import mlflow), không nằm trong module này.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class MlflowClient(Protocol):
    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, float], step: int) -> None: ...
    def set_tag(self, key: str, value: Any) -> None: ...


def flatten_config_for_logging(cfg: dict, _prefix: str = "") -> dict[str, Any]:
    """MLflow log_params cần flat dict (không nested) —
    {"optimizer": {"head_lr": 1e-3}} -> {"optimizer.head_lr": 1e-3}.

    KHÔNG loại section nào (khác compute_config_hash trong checkpoint.py, nơi
    loại `checkpoint`/`mlflow` khỏi HASH vì lý do resume) — ở ĐÂY log hết mọi
    section, kể cả `checkpoint`/`mlflow`, vẫn hữu ích để biết run đó lưu
    checkpoint ở đâu/log đi đâu; 2 mối quan tâm (hash resume vs. logging) độc
    lập nhau, không dùng chung 1 danh sách loại trừ.
    """
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        key = f"{_prefix}.{k}" if _prefix else k
        if isinstance(v, dict):
            out.update(flatten_config_for_logging(v, key))
        else:
            out[key] = v
    return out


@dataclass
class TrainRunLogger:
    client: MlflowClient

    def log_config(self, cfg: dict) -> None:
        self.client.log_params(flatten_config_for_logging(cfg))

    def log_freeze_report(self, report: Any) -> None:
        """`report`: FreezeReport (model_setup.py) — nhận Any để module này
        không phải import model_setup.py (giữ mlflow_logging.py chỉ biết về
        logging, không biết gì về model)."""
        total = report.total_params
        self.client.log_params(
            {
                "freeze.total_params": total,
                "freeze.trainable_params": report.trainable_params,
                "freeze.bn_affine_params": report.bn_affine_params,
                "freeze.trainable_fraction": report.trainable_params / total if total else 0.0,
            }
        )

    def log_step_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.client.log_metrics(metrics, step=step)

    def log_resume(self, *, resumed: bool, config_hash: str, checkpoint_dir: str | None) -> None:
        self.client.set_tag("resumed_from_checkpoint", str(resumed))
        self.client.set_tag("config_hash", config_hash)
        if checkpoint_dir is not None:
            self.client.set_tag("resumed_checkpoint_dir", checkpoint_dir)

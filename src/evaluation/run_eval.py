"""Entrypoint evaluation — SPEC v2.1 mục 4 (Evaluation & Decision Policy), 5.

notebooks/train_colab.ipynb gọi lệnh này SAU KHI train.py xong — chạy ĐỘC LẬP
với train.py, không gộp chung (xem docstring registry.py: quyết định đăng ký
model cần số đo verification thật, thứ train.py không có):

    python -m src.evaluation.run_eval \\
        --config configs/finetune_occlusion.yaml \\
        --inference-config configs/inference.yaml \\
        --checkpoint-dir /content/drive/MyDrive/face-tracking-mlops/checkpoints

Luồng:
  1. Load checkpoint MỚI NHẤT khớp config_hash hiện tại (CheckpointManager) —
     KHÔNG tự train, chỉ load state train.py đã lưu.
  2. embed_manifest() trên val.csv và test.csv (configs/inference.yaml:
     data.test_csv — CỐ Ý không nằm trong finetune_occlusion.yaml, xem comment
     trong file đó) riêng biệt.
  3. build_pairs() cho val và test -> 3 nhóm occlusion mỗi tập.
  4. Pool cả 3 nhóm của VAL thành "overall" -> sweep_tar_at_far() khoá 1
     threshold TOÀN CỤC (SPEC 4.2 — 1 ngưỡng duy nhất, không phải mỗi nhóm 1
     ngưỡng riêng).
  5. evaluate_at_threshold() bằng threshold đã khoá — cho CẢ "overall" lẫn 3
     nhóm riêng, trên CẢ val (sanity, không dùng để báo cáo chính) lẫn test
     (số liệu báo cáo chính — SPEC 4.2: re-validate, KHÔNG tinh chỉnh lại).
  6. Ghi reports/eval_metrics.json + cập nhật configs/inference.yaml
     (threshold.value, threshold.swept_on_config_hash).
  7. Log MLflow (metric "test_overall_tar" — registry.py đọc lại ĐÚNG tên này
     để so với Production hiện tại, xem registry.py) rồi register_if_better()
     — KHÔNG tự promote Production (SPEC 5: review thủ công bắt buộc).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
import yaml

from src.data.labels import read_manifest_csv
from src.data.normalize import load_contract
from src.evaluation.embed import embed_manifest
from src.evaluation.pairs import Pair, build_pairs
from src.evaluation.registry import build_real_registry_client, register_if_better
from src.evaluation.report import build_report, write_report
from src.evaluation.threshold import ThresholdResult, evaluate_at_threshold, sweep_tar_at_far
from src.training.backbone import load_pretrained_backbone
from src.training.checkpoint import CheckpointManager, compute_config_hash
from src.training.config import load_finetune_config
from src.training.head import ArcFaceHead
from src.training.model import FaceModel

logger = logging.getLogger(__name__)

INFERENCE_REQUIRED_KEYS = ("threshold", "eval", "data")


def load_inference_config(path: Path) -> dict[str, Any]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in INFERENCE_REQUIRED_KEYS if k not in doc]
    if missing:
        raise ValueError(f"{path}: thiếu section bắt buộc {missing}")
    if "test_csv" not in doc["data"]:
        raise ValueError(f"{path}: thiếu data.test_csv")
    if "max_impostor_pairs" not in doc["eval"] or "pair_sampling_seed" not in doc["eval"]:
        raise ValueError(f"{path}: thiếu eval.max_impostor_pairs hoặc eval.pair_sampling_seed")
    if "target_far" not in doc["threshold"]:
        raise ValueError(f"{path}: thiếu threshold.target_far")
    return doc


def _load_model_from_checkpoint(
    cfg: dict[str, Any], checkpoint_dir: Path, device: torch.device
) -> tuple[FaceModel, str]:
    config_hash = compute_config_hash(cfg)
    num_classes = len({r.identity_id for r in read_manifest_csv(Path(cfg["data"]["train_csv"]))})
    backbone = load_pretrained_backbone(cfg)
    head = ArcFaceHead(
        embedding_dim=cfg["backbone"]["num_features"],
        num_classes=num_classes,
        margin=cfg["head"]["arcface_margin"],
        scale=cfg["head"]["arcface_scale"],
    )
    model = FaceModel(backbone, head).to(device)

    manager = CheckpointManager(checkpoint_dir, keep_last_n=cfg["checkpoint"]["keep_last_n"])
    ckpt_dir = manager.find_resumable(config_hash)
    if ckpt_dir is None:
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint khớp config_hash={config_hash} trong "
            f"{checkpoint_dir} — chạy train.py trước khi eval."
        )
    state, meta = manager.load(ckpt_dir)
    model.load_state_dict(state["model"])
    model.eval()
    logger.info("Đã load checkpoint %s (epoch=%d)", ckpt_dir, meta.epoch)
    return model, config_hash


def _pool_groups(groups: dict[str, list[Pair]]) -> list[Pair]:
    pooled: list[Pair] = []
    for group_pairs in groups.values():
        pooled.extend(group_pairs)
    return pooled


def run_eval(
    cfg: dict[str, Any],
    inference_cfg: dict[str, Any],
    *,
    images_root: Path,
    checkpoint_dir: Path,
    export_config: Path,
    report_path: Path,
    inference_config_path: Path,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contract = load_contract(export_config)
    model, config_hash = _load_model_from_checkpoint(cfg, checkpoint_dir, device)

    eval_cfg = inference_cfg["eval"]
    target_far = inference_cfg["threshold"]["target_far"]

    val_records = embed_manifest(model, Path(cfg["data"]["val_csv"]), contract, images_root, device)
    test_records = embed_manifest(
        model, Path(inference_cfg["data"]["test_csv"]), contract, images_root, device
    )

    pair_kwargs = {
        "max_impostor_pairs": eval_cfg["max_impostor_pairs"],
        "seed": eval_cfg["pair_sampling_seed"],
    }
    val_groups = build_pairs(val_records, **pair_kwargs)
    test_groups = build_pairs(test_records, **pair_kwargs)

    # Threshold TOÀN CỤC: sweep trên pool cả 3 nhóm của VAL (SPEC 4.2) — không
    # phải mỗi nhóm 1 threshold riêng, vì production chỉ có 1 ngưỡng quyết định
    # match/no-match, phải phản ánh đúng hỗn hợp điều kiện occlusion thực tế.
    threshold_result = sweep_tar_at_far(_pool_groups(val_groups), target_far=target_far)
    locked_threshold = threshold_result.threshold

    val_results: dict[str, ThresholdResult] = {"overall": threshold_result}
    test_results: dict[str, tuple[float, float]] = {
        "overall": evaluate_at_threshold(_pool_groups(test_groups), locked_threshold)
    }
    for group_name, pairs in val_groups.items():
        val_tar, val_far = evaluate_at_threshold(pairs, locked_threshold)
        val_results[group_name] = ThresholdResult(
            threshold=locked_threshold,
            target_far=target_far,
            achieved_far=val_far,
            achieved_tar=val_tar,
            num_genuine=sum(1 for p in pairs if p.is_genuine),
            num_impostor=sum(1 for p in pairs if not p.is_genuine),
        )
        test_results[group_name] = evaluate_at_threshold(test_groups[group_name], locked_threshold)

    report = build_report(
        config_hash=config_hash,
        locked_threshold=locked_threshold,
        target_far=target_far,
        val_results=val_results,
        test_results=test_results,
    )
    write_report(report, report_path)
    _update_inference_config(inference_config_path, locked_threshold, config_hash)
    _log_and_maybe_register(cfg, config_hash, locked_threshold, test_results)
    return report


def _log_and_maybe_register(
    cfg: dict[str, Any],
    config_hash: str,
    locked_threshold: float,
    test_results: dict[str, tuple[float, float]],
) -> None:
    import mlflow  # noqa: PLC0415 — lazy import, cùng nguyên tắc train.py

    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    with mlflow.start_run():  # mlflow tự end_run(FAILED) nếu exception, FINISHED nếu không
        mlflow.log_param("config_hash", config_hash)
        mlflow.log_metric("locked_threshold", locked_threshold)
        for group_name, (tar, far) in test_results.items():
            if tar == tar:  # loại NaN (NaN != NaN) — mlflow.log_metric không nhận NaN
                mlflow.log_metric(f"test_{group_name}_tar", tar)
            if far == far:
                mlflow.log_metric(f"test_{group_name}_far", far)
        run_id = mlflow.active_run().info.run_id

    client = build_real_registry_client()
    register_if_better(
        client,
        registered_model_name=cfg["mlflow"]["registered_model_name"],
        run_id=run_id,
        config_hash=config_hash,
        new_test_metric=test_results["overall"][0],
    )


def _update_inference_config(path: Path, threshold: float, config_hash: str) -> None:
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    doc.setdefault("threshold", {})
    doc["threshold"]["value"] = threshold
    doc["threshold"]["swept_on_config_hash"] = config_hash
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/finetune_occlusion.yaml"))
    parser.add_argument("--inference-config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--export-config", type=Path, default=Path("configs/export.yaml"))
    parser.add_argument("--images-root", type=Path, default=Path("."))
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, default=Path("reports/eval_metrics.json"))
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cfg = load_finetune_config(args.config)
    inference_cfg = load_inference_config(args.inference_config)
    run_eval(
        cfg,
        inference_cfg,
        images_root=args.images_root,
        checkpoint_dir=args.checkpoint_dir,
        export_config=args.export_config,
        report_path=args.report_path,
        inference_config_path=args.inference_config,
    )


if __name__ == "__main__":
    main()

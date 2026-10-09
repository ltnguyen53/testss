"""Entrypoint training — SPEC v2.1 mục 3, 10.

notebooks/train_colab.ipynb gọi DUY NHẤT lệnh này (mục 10: notebook chỉ chứa
lệnh chạy, mọi logic nằm ở đây, test được qua CI):

    python -m src.training.train \\
        --config configs/finetune_occlusion.yaml \\
        --resume-dir /content/drive/MyDrive/face-tracking-mlops/checkpoints

Luồng 1 lần chạy:
  1. load + validate config (fail loudly ngay nếu config sai — config.py).
  2. build backbone (weight thật, SHA256 check — weights.py) + ArcFaceHead.
  3. Bọc FaceModel(backbone, head) -> apply_freeze_policy -> build_param_groups.
  4. PKBatchSampler CHỈ đọc train.csv — val.csv KHÔNG dùng ở Phase 2 (chưa có
     eval module, xem SPEC Phase 3). train.py KHÔNG tự chọn "model tốt nhất":
     chưa có metric nào đáng tin để chọn lúc này (threshold/eval cần protocol
     embedding-only riêng — SPEC 4.2), chỉ tối ưu loss + checkpoint định kỳ.
  5. Nếu KHÔNG resume: 1 lượt forward không-gradient qua backbone pretrained để
     khởi tạo head từ trung bình embedding theo lớp (README rủi ro mục 9).
  6. CheckpointManager.find_resumable(config_hash) — có thì load state + resume
     RNG + optimizer + epoch/step, không thì train từ đầu epoch 0.
  7. Vòng lặp epoch/step, log MLflow, checkpoint định kỳ theo
     checkpoint.save_every_n_steps.

Ngữ nghĩa `CheckpointMeta.step` ở module này: số BATCH ĐÃ CHẠY TRONG EPOCH HIỆN
TẠI (reset về 0 mỗi epoch mới), KHÔNG phải step toàn cục — PKBatchSampler sinh
batch deterministic theo (seed, epoch) nên resume giữa epoch chỉ cần bỏ qua
đúng số batch đã chạy của epoch đó (`epoch_batches(epoch)[step:]`), không cần
step toàn cục.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.data.normalize import load_contract
from src.training.backbone import load_pretrained_backbone
from src.training.checkpoint import (
    CheckpointManager,
    CheckpointMeta,
    compute_config_hash,
    get_git_commit_hash,
)
from src.training.config import (
    freeze_policy_from_config,
    load_finetune_config,
    param_group_kwargs_from_config,
    sampler_kwargs_from_config,
    training_kwargs_from_config,
)
from src.training.dataset import load_batch
from src.training.head import ArcFaceHead
from src.training.mlflow_logging import TrainRunLogger
from src.training.model import FaceModel
from src.training.model_setup import apply_freeze_policy, build_param_groups
from src.training.rng_state import capture_all_rng, restore_all_rng
from src.training.sampler import PKBatchSampler

logger = logging.getLogger(__name__)


def build_optimizer(
    param_groups: list[dict[str, Any]], training_cfg: dict[str, Any]
) -> torch.optim.Optimizer:
    if training_cfg["optimizer_type"] == "sgd":
        return torch.optim.SGD(param_groups, momentum=training_cfg["momentum"])
    # validate_finetune_config (config.py) đã chặn giá trị lạ từ lúc load config —
    # nhánh này là phòng thủ kép, không nên tới được đây.
    raise ValueError(f"optimizer_type không hỗ trợ: {training_cfg['optimizer_type']}")


@torch.no_grad()
def compute_class_mean_embeddings(
    model: FaceModel,
    sampler: PKBatchSampler,
    contract: Any,
    images_root: Path,
    device: torch.device,
    embedding_dim: int,
) -> torch.Tensor:
    """Trung bình embedding theo lớp, dùng làm prototype khởi tạo head (SPEC
    3.2, README rủi ro mục 9) — thay vì random init.

    Tái dùng `sampler.epoch_batches(0)` thay vì tự viết vòng lặp load-toàn-bộ
    riêng: 1 epoch của PKBatchSampler vốn đã duyệt qua MỌI identity (docstring
    sampler.py: "1 epoch = 1 lượt qua toàn bộ identity"), nên đây là cách rẻ
    nhất để có vài ảnh đại diện/identity mà không cần thêm 1 đường load data
    thứ 2. Gọi TRƯỚC vòng lặp train (dù trước/sau apply_freeze_policy không
    quan trọng — freeze chỉ đổi requires_grad, KHÔNG đổi giá trị weight, nên
    forward pass ở đây luôn dùng đúng backbone pretrained nguyên bản bất kể đã
    apply_freeze_policy hay chưa) — chạy ở eval() mode để tắt BN update từ
    batch tạm thời này, không phải batch train thật.
    """
    was_training = model.training
    model.eval()
    try:
        num_classes = sampler.num_classes
        sums = torch.zeros(num_classes, embedding_dim, device=device)
        counts = torch.zeros(num_classes, device=device)
        identity_to_class = sampler.identity_to_class
        for batch_records in sampler.epoch_batches(0):
            images, labels = load_batch(batch_records, identity_to_class, contract, images_root)
            images, labels = images.to(device), labels.to(device)
            emb = model.embed(images)
            sums.index_add_(0, labels, emb)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
        counts = counts.clamp(min=1.0).unsqueeze(1)
        return sums / counts
    finally:
        model.train(was_training)


def _resume_or_start_fresh(
    checkpoint_manager: CheckpointManager,
    config_hash: str,
    model: FaceModel,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, int, bool]:
    """Trả (start_epoch, start_step_in_epoch, resumed). Không tìm thấy checkpoint
    khớp config_hash -> (0, 0, False), KHÔNG BAO GIỜ resume nhầm config khác
    (CheckpointManager.find_resumable đã đảm bảo điều này — xem checkpoint.py)."""
    ckpt_dir = checkpoint_manager.find_resumable(config_hash)
    if ckpt_dir is None:
        return 0, 0, False

    state, meta = checkpoint_manager.load(ckpt_dir)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    restore_all_rng(state["rng"])
    logger.info("Resume từ %s (epoch=%d, step_in_epoch=%d)", ckpt_dir, meta.epoch, meta.step)
    return meta.epoch, meta.step, True


def _save_checkpoint(
    checkpoint_manager: CheckpointManager,
    config_hash: str,
    model: FaceModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step_in_epoch: int,
) -> Path:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": capture_all_rng(),
    }
    meta = CheckpointMeta(
        timestamp=datetime.now(timezone.utc).isoformat(),
        config_hash=config_hash,
        git_commit_hash=get_git_commit_hash(),
        epoch=epoch,
        step=step_in_epoch,
    )
    return checkpoint_manager.save(state, meta)


def train(
    cfg: dict[str, Any],
    *,
    images_root: Path,
    resume_dir: Path | None = None,
    export_config: Path = Path("configs/export.yaml"),
) -> None:
    # BUG THẬT ĐÃ PHÁT HIỆN + SỬA (2026-09-29, bằng test tích hợp thật —
    # tests/unit/test_train.py:test_mlflow_run_ends_with_failed_status_on_exception):
    # bản đầu tiên mở mlflow run SAU khi đã load backbone/tính class-mean
    # embedding — nếu 1 trong các bước ĐÓ lỗi (backbone load sai weight, ảnh
    # hỏng...), lỗi bay ra TRƯỚC khi mlflow.start_run() từng được gọi, nên
    # KHÔNG CÓ RUN NÀO được ghi lại cho lần thất bại đó — mất hoàn toàn visibility
    # vào lỗi setup. Mở run NGAY TỪ ĐẦU, bọc try/except/finally quanh TOÀN BỘ
    # phần còn lại của hàm để mọi lỗi (kể cả lỗi setup) đều có 1 run FAILED
    # tương ứng trên MLflow.
    import mlflow  # noqa: PLC0415 — lazy import, cùng nguyên tắc backbone.py/weights.py

    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    mlflow.start_run()
    try:
        _train_inner(cfg, images_root, resume_dir, export_config)
    except BaseException:
        mlflow.end_run(status="FAILED")
        raise
    else:
        mlflow.end_run(status="FINISHED")


def _train_inner(
    cfg: dict[str, Any],
    images_root: Path,
    resume_dir: Path | None,
    export_config: Path,
) -> None:
    config_hash = compute_config_hash(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contract = load_contract(export_config)
    embedding_dim = cfg["backbone"]["num_features"]

    backbone = load_pretrained_backbone(cfg)
    sampler = PKBatchSampler(Path(cfg["data"]["train_csv"]), **sampler_kwargs_from_config(cfg))
    head = ArcFaceHead(
        embedding_dim=embedding_dim,
        num_classes=sampler.num_classes,
        margin=cfg["head"]["arcface_margin"],
        scale=cfg["head"]["arcface_scale"],
    )
    # .to(device) SAU khi bọc FaceModel — chỉ 1 lần, không dời device 3 lần riêng lẻ
    model = FaceModel(backbone, head).to(device)

    policy = freeze_policy_from_config(cfg)
    freeze_report = apply_freeze_policy(model, policy)
    param_groups = build_param_groups(model, policy, **param_group_kwargs_from_config(cfg))
    optimizer = build_optimizer(param_groups, training_kwargs_from_config(cfg))
    loss_fn = torch.nn.CrossEntropyLoss()

    checkpoint_dir = resume_dir or Path(cfg["checkpoint"]["dir"])
    checkpoint_manager = CheckpointManager(
        checkpoint_dir, keep_last_n=cfg["checkpoint"]["keep_last_n"]
    )
    start_epoch, start_step, resumed = _resume_or_start_fresh(
        checkpoint_manager, config_hash, model, optimizer
    )

    if not resumed:
        means = compute_class_mean_embeddings(
            model, sampler, contract, images_root, device, embedding_dim
        )
        head.init_from_class_means(means)
        logger.info("Khởi tạo head từ trung bình embedding theo lớp (%d lớp)", sampler.num_classes)

    run_logger = TrainRunLogger(_RealMlflowClient())
    run_logger.log_config(cfg)
    run_logger.log_freeze_report(freeze_report)
    run_logger.log_resume(
        resumed=resumed,
        config_hash=config_hash,
        checkpoint_dir=str(checkpoint_dir) if resumed else None,
    )
    _run_training_loop(
        cfg,
        model,
        optimizer,
        loss_fn,
        sampler,
        contract,
        images_root,
        device,
        checkpoint_manager,
        config_hash,
        run_logger,
        start_epoch,
        start_step,
    )


def _run_training_loop(
    cfg: dict[str, Any],
    model: FaceModel,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    sampler: PKBatchSampler,
    contract: Any,
    images_root: Path,
    device: torch.device,
    checkpoint_manager: CheckpointManager,
    config_hash: str,
    run_logger: TrainRunLogger,
    start_epoch: int,
    start_step: int,
) -> None:
    training_cfg = training_kwargs_from_config(cfg)
    save_every = cfg["checkpoint"]["save_every_n_steps"]
    global_step = 0

    for epoch in range(start_epoch, training_cfg["epochs"]):
        batches = sampler.epoch_batches(epoch)
        skip = start_step if epoch == start_epoch else 0
        for step_in_epoch, batch_records in enumerate(batches[skip:], start=skip):
            images, labels = load_batch(
                batch_records, sampler.identity_to_class, contract, images_root
            )
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images, labels)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            global_step += 1
            run_logger.log_step_metrics({"loss": loss.item()}, step=global_step)

            if global_step % save_every == 0:
                _save_checkpoint(
                    checkpoint_manager, config_hash, model, optimizer, epoch, step_in_epoch + 1
                )

        # Hết epoch: luôn checkpoint (step_in_epoch=0 cho epoch KẾ TIẾP) — không
        # phụ thuộc save_every_n_steps có chia hết đúng lúc epoch kết thúc hay không.
        _save_checkpoint(checkpoint_manager, config_hash, model, optimizer, epoch + 1, 0)

    logger.info("Hoàn tất %d epoch (config_hash=%s)", training_cfg["epochs"], config_hash)


class _RealMlflowClient:
    """Adapter mỏng khớp mlflow_logging.MlflowClient Protocol — gọi mlflow thật.
    Tách riêng để có thể patch/thay thế trong test mà không đụng logic
    TrainRunLogger (đã test độc lập bằng FakeMlflowClient, xem
    tests/unit/test_mlflow_logging.py)."""

    def log_params(self, params: dict[str, Any]) -> None:
        import mlflow  # noqa: PLC0415

        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        import mlflow  # noqa: PLC0415

        mlflow.log_metrics(metrics, step=step)

    def set_tag(self, key: str, value: Any) -> None:
        import mlflow  # noqa: PLC0415

        mlflow.set_tag(key, value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/finetune_occlusion.yaml"))
    parser.add_argument("--export-config", type=Path, default=Path("configs/export.yaml"))
    parser.add_argument("--images-root", type=Path, default=Path("."))
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=None,
        help="Override checkpoint.dir trong config — dùng khi mount Drive khác giữa các session.",
    )
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cfg = load_finetune_config(args.config)
    train(
        cfg,
        images_root=args.images_root,
        resume_dir=args.resume_dir,
        export_config=args.export_config,
    )


if __name__ == "__main__":
    main()

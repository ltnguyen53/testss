"""Load + validate configs/finetune_occlusion.yaml — fail loudly ngay lúc load.

Mục đích: lỗi config (thiếu key, LR ngược chiều discriminative, K < số slot masked...)
phải lộ ở giây đầu tiên, không phải sau khi Colab đã tốn vài phút setup GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.training.model_setup import FreezePolicy

REQUIRED_KEYS: dict[str, list[str]] = {
    "data": ["train_csv", "val_csv"],
    "backbone": [
        "arch",
        "num_features",
        "dropout",
        "arcface_torch_dir",
        "pretrained_weights",
        "weights_source",
    ],
    "freeze": [
        "trainable_blocks",
        "train_embedding_fc",
        "head_module",
        "train_bn_affine_everywhere",
        "bn_affine_exclude_modules",
    ],
    "head": ["arcface_margin", "arcface_scale"],
    "optimizer": [
        "head_lr",
        "backbone_unfrozen_lr",
        "bn_affine_lr",
        "weight_decay",
        "bn_affine_weight_decay",
    ],
    "sampler": ["identities_per_batch", "images_per_identity", "masked_per_identity"],
    "training": ["optimizer_type", "momentum", "epochs", "amp"],
    "checkpoint": ["save_every_n_steps", "keep_last_n", "dir"],
    "mlflow": ["experiment_name", "registered_model_name"],
}

SUPPORTED_OPTIMIZER_TYPES = ("sgd",)  # Adam cần dò lại toàn bộ optimizer.*_lr, xem yaml


def validate_finetune_config(cfg: dict[str, Any]) -> None:
    if "seed" not in cfg:
        raise ValueError("config thiếu key 'seed'")
    for section, keys in REQUIRED_KEYS.items():
        if section not in cfg or not isinstance(cfg[section], dict):
            raise ValueError(f"config thiếu section '{section}'")
        missing = [k for k in keys if k not in cfg[section]]
        if missing:
            raise ValueError(f"section '{section}' thiếu key: {missing}")

    opt = cfg["optimizer"]
    if not opt["head_lr"] > opt["backbone_unfrozen_lr"] > 0:
        raise ValueError(
            "SPEC 3.1: head_lr phải > backbone_unfrozen_lr > 0 (discriminative LR) — "
            f"nhận head_lr={opt['head_lr']}, backbone_unfrozen_lr={opt['backbone_unfrozen_lr']}"
        )
    if not 0 < cfg["head"]["arcface_margin"] < 1:
        raise ValueError("arcface_margin phải nằm trong (0, 1) — đơn vị radian, paper gốc 0.5")
    if cfg["head"]["arcface_scale"] <= 0:
        raise ValueError("arcface_scale phải > 0")

    s = cfg["sampler"]
    if not 0 <= s["masked_per_identity"] <= s["images_per_identity"]:
        raise ValueError("sampler.masked_per_identity phải trong [0, images_per_identity]")
    blocks = cfg["freeze"]["trainable_blocks"]
    if not blocks:
        raise ValueError("freeze.trainable_blocks rỗng — không unfreeze block nào")
    if "fc" in blocks:
        raise ValueError(
            "Không khai 'fc' trong freeze.trainable_blocks — dùng freeze.train_embedding_fc: true "
            "(fc mặc định bị freeze vì dễ overfit, SPEC v2.1 mục 3.1)"
        )

    t = cfg["training"]
    if t["optimizer_type"] not in SUPPORTED_OPTIMIZER_TYPES:
        raise ValueError(
            f"training.optimizer_type={t['optimizer_type']!r} chưa hỗ trợ — chỉ "
            f"{SUPPORTED_OPTIMIZER_TYPES} (đổi optimizer đòi dò lại optimizer.*_lr, "
            "xem comment trong finetune_occlusion.yaml)"
        )
    if t["epochs"] <= 0:
        raise ValueError("training.epochs phải > 0")
    if not 0 <= t["momentum"] < 1:
        raise ValueError("training.momentum phải nằm trong [0, 1)")


def load_finetune_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_finetune_config(cfg)
    return cfg


def freeze_policy_from_config(cfg: dict[str, Any]) -> FreezePolicy:
    """Đường dẫn module trong model bọc ngoài `FaceModel(backbone=..., head=...)`: khối của
    backbone có tiền tố "backbone."; head là `head_module` (không tiền tố)."""
    f = cfg["freeze"]
    unfreeze = [f"backbone.{b}" for b in f["trainable_blocks"]]
    if f["train_embedding_fc"]:
        unfreeze.append("backbone.fc")
    return FreezePolicy(
        unfreeze_modules=tuple(unfreeze),
        head_module=f["head_module"],
        train_bn_affine_everywhere=bool(f["train_bn_affine_everywhere"]),
        bn_affine_exclude_modules=tuple(f"backbone.{m}" for m in f["bn_affine_exclude_modules"]),
    )


def param_group_kwargs_from_config(cfg: dict[str, Any]) -> dict[str, float]:
    o = cfg["optimizer"]
    return {
        "head_lr": o["head_lr"],
        "backbone_unfrozen_lr": o["backbone_unfrozen_lr"],
        "bn_affine_lr": o["bn_affine_lr"],
        "weight_decay": o["weight_decay"],
        "bn_affine_weight_decay": o["bn_affine_weight_decay"],
    }


def training_kwargs_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    t = cfg["training"]
    return {
        "optimizer_type": t["optimizer_type"],
        "momentum": t["momentum"],
        "epochs": t["epochs"],
        "amp": t["amp"],
    }


def sampler_kwargs_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    s = cfg["sampler"]
    return {
        "identities_per_batch": s["identities_per_batch"],
        "images_per_identity": s["images_per_identity"],
        "masked_per_identity": s["masked_per_identity"],
        "seed": cfg["seed"],
    }

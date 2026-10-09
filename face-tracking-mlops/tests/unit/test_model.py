"""Unit test cho src/training/model.py bằng torch THẬT. Bỏ qua sạch nếu torch
chưa cài (xem test_head.py — cùng lý do)."""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from src.training.head import ArcFaceHead  # noqa: E402
from src.training.model import FaceModel  # noqa: E402
from src.training.model_setup import (  # noqa: E402
    FreezePolicy,
    apply_freeze_policy,
    build_param_groups,
)


class _TinyBackbone(nn.Module):
    """Backbone giả NHƯNG là nn.Module THẬT — mô phỏng đúng đặc điểm gây bug đã
    sửa ở model_setup.py: có 1 module cuối (`features`, BatchNorm1d) mà
    kiến trúc TỰ cố định weight=1.0/requires_grad=False (giống hệt iresnet
    thật của arcface_torch — đã verify, xem model_setup.py docstring điểm 4),
    còn bias thì không bị cố định. Có `layer4` (unfreeze target) và
    `layer_frozen` (phải giữ frozen) để phân biệt rõ 2 nhánh.
    """

    def __init__(self, embedding_dim: int = 8) -> None:
        super().__init__()
        self.layer_frozen = nn.Conv2d(3, 4, 3)
        self.layer4 = nn.Conv2d(4, 4, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, embedding_dim)
        self.features = nn.BatchNorm1d(embedding_dim)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False  # đúng như iresnet thật

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_frozen(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return self.features(x)


def _make_model(embedding_dim: int = 8, num_classes: int = 5) -> FaceModel:
    backbone = _TinyBackbone(embedding_dim=embedding_dim)
    head = ArcFaceHead(embedding_dim=embedding_dim, num_classes=num_classes, margin=0.3, scale=32.0)
    return FaceModel(backbone, head)


def test_embed_bypasses_head_entirely():
    model = _make_model(embedding_dim=8, num_classes=5)
    x = torch.randn(3, 3, 16, 16)

    emb = model.embed(x)

    assert emb.shape == (3, 8)  # đúng embedding_dim, KHÔNG phải (3, num_classes)


def test_forward_requires_labels_goes_through_head():
    model = _make_model(embedding_dim=8, num_classes=5)
    x = torch.randn(3, 3, 16, 16)
    labels = torch.tensor([0, 2, 4])

    logits = model(x, labels)

    assert logits.shape == (3, 5)


def test_forward_without_labels_raises_typeerror():
    """labels KHÔNG có default — quên truyền phải fail loudly ngay tại lời gọi,
    không được âm thầm chạy thiếu margin (xem docstring model.py)."""
    model = _make_model()
    x = torch.randn(2, 3, 16, 16)

    with pytest.raises(TypeError):
        model(x)  # type: ignore[call-arg]


def test_freeze_policy_integration_respects_features_weight_on_real_module():
    """Test tích hợp: apply_freeze_policy chạy trên FaceModel THẬT (không phải
    FakeModule của test_model_setup.py) — đóng nốt khoảng trống "CHƯA verify
    trên nn.Module thật" mà model_setup.py từng tự flag."""
    model = _make_model(embedding_dim=8, num_classes=5)
    policy = FreezePolicy(
        unfreeze_modules=("backbone.layer4",),
        head_module="head",
        train_bn_affine_everywhere=True,
    )

    report = apply_freeze_policy(model, policy)

    assert model.backbone.layer_frozen.weight.requires_grad is False
    assert model.backbone.layer4.weight.requires_grad is True
    assert model.backbone.features.weight.requires_grad is False  # bị cố định gốc, không bật lại
    assert model.backbone.features.bias.requires_grad is True  # bias vẫn được unfreeze
    assert all(p.requires_grad for p in model.head.parameters())
    assert report.trainable_params > 0

    groups = build_param_groups(
        model, policy, head_lr=1e-3, backbone_unfrozen_lr=1e-5, bn_affine_lr=1e-5, weight_decay=0.0
    )
    grouped_ids = {id(p) for g in groups for p in g["params"]}
    assert id(model.backbone.features.weight) not in grouped_ids
    assert id(model.backbone.features.bias) in grouped_ids


def test_one_real_train_step_reduces_loss_on_tiny_overfit_batch():
    """Sanity end-to-end: forward + backward + optimizer.step() chạy được KHÔNG
    lỗi trên FaceModel thật, và loss giảm sau vài step trên 1 batch cố định cực
    nhỏ (overfit test kinh điển) — không chứng minh model học tốt trên data
    thật, chỉ chứng minh dây chuyền backbone+head+optimizer không có lỗi cấu
    trúc (shape sai, gradient không chảy, param group rỗng...).
    """
    torch.manual_seed(0)
    model = _make_model(embedding_dim=8, num_classes=4)
    policy = FreezePolicy(unfreeze_modules=("backbone.layer4",), head_module="head")
    apply_freeze_policy(model, policy)
    groups = build_param_groups(
        model, policy, head_lr=1e-2, backbone_unfrozen_lr=1e-3, bn_affine_lr=1e-3, weight_decay=0.0
    )
    optimizer = torch.optim.SGD(groups)
    loss_fn = torch.nn.CrossEntropyLoss()

    x = torch.randn(6, 3, 16, 16)
    labels = torch.tensor([0, 1, 2, 3, 0, 1])

    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        logits = model(x, labels)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"loss không giảm: {losses[0]:.4f} -> {losses[-1]:.4f}"

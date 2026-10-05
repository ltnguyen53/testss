"""Unit test cho src/training/head.py bằng torch THẬT (không duck-typing/fake —
khác model_setup.py, head.py định nghĩa nn.Module thật nên PHẢI có torch để
import được). Bỏ qua sạch (không lỗi CI) nếu torch chưa cài — xem README, torch
chỉ cài trên Colab/dev machine, không cài trong lint-test CI job."""

import math

import pytest

torch = pytest.importorskip("torch")

from src.training.head import ArcFaceHead  # noqa: E402 — sau importorskip, đúng convention pytest


def test_rejects_invalid_margin_and_scale():
    with pytest.raises(ValueError, match="margin"):
        ArcFaceHead(embedding_dim=8, num_classes=5, margin=1.5, scale=32.0)
    with pytest.raises(ValueError, match="margin"):
        ArcFaceHead(embedding_dim=8, num_classes=5, margin=0.0, scale=32.0)
    with pytest.raises(ValueError, match="scale"):
        ArcFaceHead(embedding_dim=8, num_classes=5, margin=0.3, scale=0.0)


def test_forward_with_labels_returns_correct_shape():
    head = ArcFaceHead(embedding_dim=8, num_classes=5, margin=0.3, scale=32.0)
    x = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 2, 4])

    logits = head(x, labels)

    assert logits.shape == (4, 5)


def test_forward_without_labels_returns_plain_scaled_cosine():
    """Nhánh labels=None chỉ dùng debug (docstring) — không áp margin, chỉ
    cosine*scale thuần, giá trị nằm trong [-scale, scale]."""
    head = ArcFaceHead(embedding_dim=8, num_classes=5, margin=0.3, scale=32.0)
    x = torch.randn(4, 8)

    logits_no_label = head(x, labels=None)

    assert logits_no_label.shape == (4, 5)
    assert torch.all(logits_no_label.abs() <= 32.0 + 1e-4)


def test_margin_actually_changes_target_logit_vs_plain_cosine():
    """Bài test QUAN TRỌNG NHẤT: chứng minh margin THẬT SỰ được áp — logit của
    đúng lớp target (có margin) phải khác logit cosine thuần (không margin) tại
    CHÍNH vị trí đó, còn các lớp không phải target thì giống hệt cosine thuần
    (margin chỉ cộng vào đúng 1 lớp/sample, đúng theo công thức ArcFace)."""
    head = ArcFaceHead(embedding_dim=16, num_classes=6, margin=0.4, scale=1.0)  # scale=1 dễ so sánh
    x = torch.randn(3, 16)
    labels = torch.tensor([0, 3, 5])

    plain_cosine = head(x, labels=None)  # scale=1.0 -> chính là cosine thuần
    margin_logits = head(x, labels)

    for i, target_class in enumerate(labels.tolist()):
        assert not math.isclose(
            margin_logits[i, target_class].item(),
            plain_cosine[i, target_class].item(),
            abs_tol=1e-5,
        ), "logit của lớp target phải khác cosine thuần khi có margin"
        for c in range(6):
            if c != target_class:
                assert math.isclose(
                    margin_logits[i, c].item(), plain_cosine[i, c].item(), abs_tol=1e-5
                ), "logit của lớp KHÔNG phải target phải giữ nguyên = cosine thuần"


def test_gradient_flows_to_weight_and_input():
    head = ArcFaceHead(embedding_dim=8, num_classes=5, margin=0.3, scale=32.0)
    x = torch.randn(4, 8, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 4])

    logits = head(x, labels)
    logits.sum().backward()

    assert head.weight.grad is not None
    assert not torch.all(head.weight.grad == 0)
    assert x.grad is not None
    assert not torch.all(x.grad == 0)


def test_init_from_class_means_sets_normalized_prototypes():
    head = ArcFaceHead(embedding_dim=4, num_classes=3, margin=0.3, scale=32.0)
    means = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],  # lớp không có dữ liệu -> giữ nguyên init cũ
        ]
    )
    old_weight_row_2 = head.weight.data[2].clone()

    head.init_from_class_means(means)

    assert torch.allclose(head.weight.data[0], torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(head.weight.data[1], torch.tensor([0.0, 1.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(head.weight.data[2], old_weight_row_2)  # hàng toàn 0 không bị đụng tới


def test_init_from_class_means_rejects_wrong_shape():
    head = ArcFaceHead(embedding_dim=4, num_classes=3, margin=0.3, scale=32.0)
    with pytest.raises(ValueError, match="shape"):
        head.init_from_class_means(torch.randn(2, 4))

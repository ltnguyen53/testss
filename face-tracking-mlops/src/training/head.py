"""ArcFace head (additive angular margin) — SPEC v2.1 mục 3.2.

Train: forward(embeddings, labels) CẦN labels — margin chỉ cộng vào đúng lớp
target, đây chính là lý do ArcFace là "closed-set classification-style" lúc
train (SPEC 3.2). Eval: KHÔNG dùng qua head — lấy embedding trực tiếp từ
FaceModel.embed() (xem model.py) rồi so cosine similarity, không quan tâm
num_classes/head nữa (SPEC 3.2: "Eval bỏ hẳn classification head, chỉ lấy
embedding"). Head chỉ tồn tại để có gradient signal lúc train, KHÔNG bao giờ đi
vào ONNX export graph (xem model.py).

margin/scale là hyperparameter DÒ ĐƯỢC (SPEC 3.2 — không hardcode theo giá trị
paper gốc m=0.5, s=64: train_split chỉ ~370 class với rất ít ảnh/class, margin
lớn dễ gây bất ổn gradient trên tập nhỏ). Đọc từ
configs/finetune_occlusion.yaml (head.arcface_margin, head.arcface_scale),
validate ở config.py (0 < margin < 1, scale > 0) — validate lại 1 lần nữa ở
đây vì class này có thể được dùng độc lập ngoài luồng load config.

Công thức additive angular margin (cùng dạng công thức paper gốc + hầu hết
reimplementation phổ biến — KHÔNG phải bản PartialFC phân tán của
arcface_torch, vì project này train single-GPU với vài trăm class, không cần
partial FC):
    cos(theta)   = normalize(x) . normalize(W)^T
    phi          = cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
    # Giữ monotonic giảm gần biên theta -> pi (easy_margin=False kiểu paper
    # gốc): nếu cos(theta) <= cos(pi - m), dùng cos(theta) - sin(m)*m thay vì
    # phi, tránh phi tăng ngược khi theta vượt (pi - m).
    target_logit = where(cos(theta) > cos(pi - m), phi, cos(theta) - sin(m)*m)
    logits       = scale * (one_hot*target_logit + (1-one_hot)*cos(theta))

Đã verify bằng torch thật (2026-09-29, torch 2.14) — xem
tests/unit/test_head.py: shape, gradient flow, margin thực sự thay đổi target
logit so với cosine thuần, và init_from_class_means cho prototype đúng hướng
trung bình lớp.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, margin: float, scale: float) -> None:
        super().__init__()
        if not 0 < margin < 1:
            raise ValueError(f"margin phải nằm trong (0, 1) — đơn vị radian, nhận {margin}")
        if scale <= 0:
            raise ValueError(f"scale phải > 0, nhận {scale}")
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(
            self.weight
        )  # bị ghi đè nếu gọi init_from_class_means (khuyến nghị)

        self._cos_m = math.cos(margin)
        self._sin_m = math.sin(margin)
        self._threshold = math.cos(math.pi - margin)
        self._mm = math.sin(math.pi - margin) * margin

    @torch.no_grad()
    def init_from_class_means(self, class_mean_embeddings: torch.Tensor) -> None:
        """Khởi tạo prototype = trung bình embedding theo lớp thay vì random —
        quyết định đã chốt ở review trước (README rủi ro mục 9): prototype gần
        đúng ngay từ đầu giúp margin loss ổn định hơn trên tập nhỏ ít ảnh/lớp
        (so với random init, vốn cần nhiều step hơn để "tìm" đúng hướng lớp).

        class_mean_embeddings: (num_classes, embedding_dim), CHƯA cần normalize
        trước — hàm này tự normalize. Hàng nào toàn 0 (lớp không có embedding
        nào lúc gom trung bình — không nên xảy ra sau split nhưng vẫn phòng)
        giữ nguyên init random cũ, không chia 0 ra NaN.
        """
        expected = (self.num_classes, self.embedding_dim)
        if tuple(class_mean_embeddings.shape) != expected:
            raise ValueError(
                f"class_mean_embeddings phải shape {expected}, "
                f"nhận {tuple(class_mean_embeddings.shape)}"
            )
        norms = class_mean_embeddings.norm(dim=1, keepdim=True)
        has_data = norms.squeeze(1) > 1e-8
        normalized = class_mean_embeddings[has_data] / norms[has_data]
        self.weight.data[has_data] = normalized.to(self.weight.dtype)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))  # (B, num_classes)
        if labels is None:
            # KHÔNG dùng ở eval path chuẩn (SPEC 3.2 dùng FaceModel.embed() thay
            # — bỏ hẳn head). Nhánh này chỉ để debug/sanity-check thủ công,
            # KHÔNG áp margin (không có target class để biết cộng margin vào đâu).
            return cosine * self.scale

        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=1e-12))
        phi = cosine * self._cos_m - sine * self._sin_m
        phi = torch.where(cosine > self._threshold, phi, cosine - self._mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return logits * self.scale

"""FaceModel — wrapper backbone + head (SPEC v2.1 mục 3.2, 6).

.embed(x): CHỈ backbone, KHÔNG qua head — dùng cho EVAL/EXPORT (SPEC 3.2:
"Eval bỏ hẳn classification head, chỉ lấy embedding"). Đây là đường DUY NHẤT
được dùng lúc export ONNX (mục 6) — ONNX graph không bao giờ chứa head, vì
head chỉ cần thiết cho gradient signal lúc train, browser không bao giờ cần
so cosine với "prototype lớp", chỉ so 2 embedding với nhau.

.forward(x, labels): backbone + head — CHỈ dùng lúc TRAIN (ArcFace cần labels
để biết cộng margin vào đúng lớp nào). KHÔNG dùng đường này cho eval — dùng
.embed() thay.

Đặt tên thuộc tính `backbone`/`head` CỐ Ý khớp với dot-path mà
config.py:freeze_policy_from_config() đã giả định sẵn ("backbone.<block>" cho
mọi thứ trong backbone, "<head_module>" — mặc định "head" — cho ArcFaceHead)
— đổi tên 1 trong 2 thuộc tính này thì phải sửa cả freeze_policy_from_config.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FaceModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Đường DUY NHẤT dùng cho eval/export — bỏ hẳn head (SPEC 3.2)."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """CHỈ dùng lúc train — `labels` bắt buộc (không có default None), cố ý
        khác với ArcFaceHead.forward (cho phép labels=None để debug head độc
        lập). Bắt buộc ở đây để tránh đúng bẫy: quên truyền labels sẽ khiến
        head bỏ qua margin mà KHÔNG BÁO LỖI (xem ArcFaceHead.forward) — loss
        vẫn chạy, vẫn giảm, nhưng không còn là ArcFace loss thật. FaceModel đại
        diện cho đường TRAIN duy nhất (SPEC 3.2), nên fail loudly ngay ở chữ ký
        hàm thay vì để lỗi trôi qua tới lúc debug loss curve mới phát hiện."""
        embeddings = self.backbone(x)
        return self.head(embeddings, labels)

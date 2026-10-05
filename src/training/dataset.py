"""Load + preprocess 1 batch ImageRecord -> tensor — SPEC v2.1 mục 2.2, 3.3, 6.

KHÔNG dùng torch.utils.data.Dataset/DataLoader: PKBatchSampler (sampler.py) đã
tự quyết định THÀNH PHẦN từng batch (P identity x K ảnh, mix real+synthetic
theo logic riêng — xem docstring sampler.py) qua `epoch_batches(epoch) ->
list[list[ImageRecord]]`. Đây là logic mà Sampler/BatchSampler chuẩn của torch
(chỉ yield index vào 1 Dataset cố định) không biểu đạt trực tiếp được, nên ta
tự viết hàm load ảnh đơn giản cho 1 batch thay vì ép vào Dataset/DataLoader.

Dùng PIL (KHÔNG dùng cv2): requirements/train.in CỐ Ý không có opencv-python
(opencv chỉ thuộc MaskTheFace, cài riêng trong third_party/, xem comment trong
train.in) — nên `bgr_to_rgb()` của normalize.py KHÔNG được gọi ở đây: PIL trả
RGB trực tiếp qua `.convert("RGB")`, không phải BGR như cv2.imread.
`bgr_to_rgb()` dành cho caller khác lỡ dùng cv2 (vd đọc frame video sau này),
không phải batch loader này.

Alignment CHƯA CHỐT (configs/export.yaml: preprocessing.alignment: null — SPEC
v2.1 mục 14 "alignment là rủi ro độ chính xác lớn nhất"). Hàm này hiện chỉ
RESIZE đơn giản về input_size x input_size, KHÔNG align 5-điểm landmark — đủ để
Phase 2 (train.py) chạy end-to-end, KHÔNG phải giải pháp cuối cùng cho
accuracy. Phải quay lại khi có RetinaFace align pipeline (Phase 4).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.labels import ImageRecord
from src.data.normalize import PreprocessingContract, normalize_rgb_uint8


def _load_and_resize(path: Path, contract: PreprocessingContract) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB").resize((contract.input_size, contract.input_size), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)


def load_batch(
    records: list[ImageRecord],
    identity_to_class: dict[str, int],
    contract: PreprocessingContract,
    images_root: Path = Path("."),
) -> tuple[torch.Tensor, torch.Tensor]:
    """(images, labels) sẵn sàng cho FaceModel.forward.

    `identity_to_class`: từ PKBatchSampler.identity_to_class — head size = số
    identity TRONG TẬP TRAIN (SPEC 3.2), KHÔNG phải cả 525 identity của RMFD.
    Identity không có trong dict này (vd lỡ truyền nhầm batch của split khác)
    raise KeyError NGAY (kiểm tra label trước, load ảnh sau — fail-fast, không
    tốn công đọc/resize ảnh rồi mới phát hiện lỗi cấu hình).
    """
    images_root = Path(images_root)
    # Xây labels TRƯỚC (rẻ, fail-fast): nếu batch lẫn identity không thuộc split
    # hiện tại (SPEC 3.2: head size = train-split only), raise KeyError NGAY,
    # không tốn công load/resize ảnh trước rồi mới phát hiện lỗi.
    batch_labels = torch.tensor(
        [identity_to_class[r.identity_id] for r in records], dtype=torch.long
    )
    images = [_load_and_resize(images_root / r.image_path, contract) for r in records]
    normalized = [normalize_rgb_uint8(img, contract) for img in images]
    batch_images = torch.from_numpy(np.stack(normalized))
    return batch_images, batch_labels

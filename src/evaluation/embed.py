"""Trích embedding cho eval — SPEC v2.1 mục 3.2, 4.1.

Dùng FaceModel.embed() (KHÔNG qua head) — đúng nguyên tắc "Eval bỏ hẳn
classification head, chỉ lấy embedding" (SPEC 3.2). KHÔNG cần
identity_to_class thật (đó là khái niệm của TRAIN — head size = train-split
identities, xem config.py/head.py) — eval không đụng gì tới head/class index,
chỉ cần ảnh -> embedding.

Trả về `EmbeddingRecord` với `embedding: tuple[float, ...]` (KHÔNG phải
torch.Tensor) — ranh giới torch dừng lại ở module này. `pairs.py`/`threshold.py`
phía sau thuần Python, test được không cần torch cài (cùng triết lý
model_setup.py/checkpoint.py/sampler.py/config.py trong src/training/).
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F

from src.data.labels import read_manifest_csv
from src.data.normalize import PreprocessingContract
from src.training.dataset import load_batch
from src.training.model import FaceModel


class EmbeddingRecord(NamedTuple):
    identity_id: str
    source: str
    embedding: tuple[float, ...]  # ĐÃ L2-normalize — so 2 embedding = dot product thuần


@torch.no_grad()
def embed_manifest(
    model: FaceModel,
    manifest_csv: Path,
    contract: PreprocessingContract,
    images_root: Path,
    device: torch.device,
    batch_size: int = 32,
) -> list[EmbeddingRecord]:
    """Chạy model.embed() theo batch cho MỌI ảnh trong manifest, trả danh sách
    PHẲNG (không gom theo identity — pairs.py tự gom theo nhu cầu riêng của nó).
    Normalize L2 NGAY TẠI ĐÂY — toàn bộ phần sau của Phase 3 (pairs, threshold)
    chỉ cần cosine similarity = dot product của 2 vector đã normalize, không
    phải lo normalize lại nhiều lần ở nhiều nơi.
    """
    was_training = model.training
    model.eval()
    try:
        records = read_manifest_csv(manifest_csv)
        # load_batch cần 1 dict identity_id -> class index để build tensor
        # labels — eval KHÔNG dùng labels này vào đâu cả (chỉ cần tránh
        # KeyError), nên map theo thứ tự xuất hiện là đủ, không cần khớp gì
        # với head.py/config train thật.
        identity_to_class = {
            identity: i for i, identity in enumerate(sorted({r.identity_id for r in records}))
        }
        out: list[EmbeddingRecord] = []
        for i in range(0, len(records), batch_size):
            batch_records = records[i : i + batch_size]
            images, _unused_labels = load_batch(
                batch_records, identity_to_class, contract, images_root
            )
            images = images.to(device)
            emb = F.normalize(model.embed(images), dim=1)
            for record, vec in zip(batch_records, emb, strict=True):
                out.append(
                    EmbeddingRecord(record.identity_id, record.source, tuple(vec.cpu().tolist()))
                )
        return out
    finally:
        model.train(was_training)

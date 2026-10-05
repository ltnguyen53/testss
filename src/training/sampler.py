"""PK sampler — SPEC v2 mục 3.3.

Lý do ĐÚNG của PK ở đây (không phải mining hard-negative như triplet): ArcFace tính
loss so với TOÀN BỘ prototype matrix của head, không phụ thuộc identity nào khác có
mặt trong batch. Lý do thật: mỗi identity trong batch phải có K ảnh gồm CẢ masked VÀ
unmasked cùng lúc, để gradient của đúng class đó học từ cả 2 biến thể occlusion.
=> K (và tỷ lệ masked/unmasked trong K) mới là thứ quan trọng; P chỉ cần đủ lớn.

Vấn đề dữ liệu thật: RMFD lệch nặng (~9-10 ảnh mask thật vs ~170 unmasked / identity,
và trung bình che giấu vài identity gần như không có ảnh mask thật). Cách xử lý:
  1. Slot masked: ảnh mask THẬT trước, thiếu thì bù masktheface_synthetic CỦA CHÍNH
     identity đó (sinh từ ảnh unmasked của identity đó — không phá person-disjoint).
  2. Không bao giờ lấy ảnh của identity khác để fill — mọi ảnh trả về đều có
     identity_id trùng identity đang được sample.
  3. Trước khi sample, tính count THẬT từng identity (`coverage`) và cảnh báo
     identity nào phải dựa vào synthetic / không đủ — không giả định phân bố đều.

Không phụ thuộc torch: đọc trực tiếp manifest CSV do src/data/labels.py sinh ra.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

from src.data.labels import ImageRecord, read_manifest_csv

logger = logging.getLogger(__name__)

SYNTHETIC_SOURCE = "masktheface_synthetic"


def _real_masked(rows: list[ImageRecord]) -> list[ImageRecord]:
    return [r for r in rows if r.is_masked and r.source != SYNTHETIC_SOURCE]


def _synthetic_masked(rows: list[ImageRecord]) -> list[ImageRecord]:
    return [r for r in rows if r.is_masked and r.source == SYNTHETIC_SOURCE]


def _unmasked(rows: list[ImageRecord]) -> list[ImageRecord]:
    return [r for r in rows if not r.is_masked]


@dataclass(frozen=True)
class IdentityCoverage:
    identity_id: str
    real_masked: int
    synthetic_masked: int
    unmasked: int


def compute_coverage(grouped: dict[str, list[ImageRecord]]) -> list[IdentityCoverage]:
    """Count thật từng identity — sort theo real_masked tăng dần (identity yếu nhất trước)."""
    out = [
        IdentityCoverage(
            identity_id=identity_id,
            real_masked=len(_real_masked(rows)),
            synthetic_masked=len(_synthetic_masked(rows)),
            unmasked=len(_unmasked(rows)),
        )
        for identity_id, rows in grouped.items()
    ]
    return sorted(out, key=lambda c: (c.real_masked, c.identity_id))


def pick_for_identity(
    rows: list[ImageRecord],
    k: int,
    masked_per_identity: int,
    rng: random.Random,
) -> list[ImageRecord]:
    """Chọn đúng K ảnh cho 1 identity: `masked_per_identity` ảnh masked + phần còn lại unmasked.

    Thứ tự ưu tiên slot masked: mask thật -> synthetic (cùng identity). Nếu vẫn thiếu
    (identity không có cả synthetic) thì slot đó chuyển sang unmasked (warning).
    Nếu tổng ảnh của identity < K thì lấy lặp có hoàn lại (warning) — dấu hiệu nên
    tăng split.min_images_per_identity.
    """
    if not rows:
        raise ValueError("identity không có ảnh nào")
    identity_id = rows[0].identity_id

    real = _real_masked(rows)
    synth = _synthetic_masked(rows)
    unmasked = _unmasked(rows)
    rng.shuffle(real)
    rng.shuffle(synth)
    rng.shuffle(unmasked)

    n_masked_target = min(masked_per_identity, k)
    masked_pick = (real + synth)[:n_masked_target]  # real trước, synthetic bù
    if len(masked_pick) < n_masked_target:
        # debug, không phải warning: hàm này chạy mỗi batch mỗi epoch; cảnh báo tổng hợp
        # 1 lần ở PKBatchSampler._log_coverage (tránh spam log + che cảnh báo quan trọng).
        logger.debug(
            "identity %s: chỉ có %d ảnh masked (thật+synthetic) < %d slot masked",
            identity_id,
            len(masked_pick),
            n_masked_target,
        )

    n_unmasked_target = k - len(masked_pick)
    unmasked_pick = unmasked[:n_unmasked_target]

    picked = masked_pick + unmasked_pick
    if len(picked) < k:
        # Thiếu cả unmasked: thử dùng nốt masked còn dư trước khi lặp lại ảnh.
        leftover = [r for r in (real + synth) if r not in picked]
        picked += leftover[: k - len(picked)]
    if len(picked) < k:
        logger.debug(
            "identity %s: tổng chỉ %d ảnh < K=%d, lấy lặp có hoàn lại",
            identity_id,
            len(picked),
            k,
        )
        picked += rng.choices(picked, k=k - len(picked))

    assert all(r.identity_id == identity_id for r in picked)  # bất biến: không lẫn identity
    return picked


class PKBatchSampler:
    """Batch = P identity x K ảnh/identity. 1 epoch = 1 lượt qua toàn bộ identity (shuffle).

    Deterministic theo (seed, epoch): resume giữa epoch chỉ cần sinh lại
    `epoch_batches(epoch)` và bỏ qua số batch đã chạy.
    """

    def __init__(
        self,
        manifest_csv: Path,
        identities_per_batch: int,
        images_per_identity: int,
        masked_per_identity: int,
        seed: int = 42,
        drop_last: bool = True,
    ) -> None:
        if not 0 <= masked_per_identity <= images_per_identity:
            raise ValueError("masked_per_identity phải nằm trong [0, images_per_identity]")
        rows = read_manifest_csv(Path(manifest_csv))
        self._grouped: dict[str, list[ImageRecord]] = {}
        for r in rows:
            self._grouped.setdefault(r.identity_id, []).append(r)
        self._identity_ids = sorted(self._grouped)
        if identities_per_batch > len(self._identity_ids):
            raise ValueError(
                f"identities_per_batch={identities_per_batch} > số identity "
                f"trong manifest ({len(self._identity_ids)})"
            )
        self.identities_per_batch = identities_per_batch
        self.images_per_identity = images_per_identity
        self.masked_per_identity = masked_per_identity
        self.seed = seed
        self.drop_last = drop_last

        self.coverage = compute_coverage(self._grouped)
        self._log_coverage()

    @property
    def identity_to_class(self) -> dict[str, int]:
        """Head size = số identity TRONG TẬP TRAIN (SPEC 3.2), không phải cả 525."""
        return {identity_id: i for i, identity_id in enumerate(self._identity_ids)}

    @property
    def num_classes(self) -> int:
        return len(self._identity_ids)

    def identities_needing_fallback(self) -> list[str]:
        return [c.identity_id for c in self.coverage if c.real_masked < self.masked_per_identity]

    def identities_unable_to_fill_masked(self) -> list[str]:
        return [
            c.identity_id
            for c in self.coverage
            if c.real_masked + c.synthetic_masked < self.masked_per_identity
        ]

    def _log_coverage(self) -> None:
        total = len(self.coverage)
        fallback = self.identities_needing_fallback()
        cannot = self.identities_unable_to_fill_masked()
        logger.info(
            "PK coverage: %d identity | cần bù synthetic: %d | không đủ masked kể cả synthetic: %d",
            total,
            len(fallback),
            len(cannot),
        )
        if cannot:
            logger.warning(
                "identity không đủ %d ảnh masked (thật+synthetic): %s",
                self.masked_per_identity,
                cannot[:20],
            )
        too_few = [
            c.identity_id
            for c in self.coverage
            if c.real_masked + c.synthetic_masked + c.unmasked < self.images_per_identity
        ]
        if too_few:
            logger.warning(
                "identity có tổng ảnh < K=%d (sẽ bị lấy lặp): %s — cân nhắc tăng "
                "split.min_images_per_identity trong configs/data.yaml",
                self.images_per_identity,
                too_few[:20],
            )

    def __len__(self) -> int:
        n = len(self._identity_ids)
        if self.drop_last:
            return n // self.identities_per_batch
        return -(-n // self.identities_per_batch)

    def epoch_batches(self, epoch: int) -> list[list[ImageRecord]]:
        rng = random.Random(self.seed * 100_000 + epoch)
        ids = list(self._identity_ids)
        rng.shuffle(ids)
        batches: list[list[ImageRecord]] = []
        for start in range(0, len(ids), self.identities_per_batch):
            batch_ids = ids[start : start + self.identities_per_batch]
            if self.drop_last and len(batch_ids) < self.identities_per_batch:
                break
            batch: list[ImageRecord] = []
            for identity_id in batch_ids:
                batch.extend(
                    pick_for_identity(
                        self._grouped[identity_id],
                        k=self.images_per_identity,
                        masked_per_identity=self.masked_per_identity,
                        rng=rng,
                    )
                )
            batches.append(batch)
        return batches

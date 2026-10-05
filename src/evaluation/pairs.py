"""Dựng cặp genuine/impostor để đo verification — SPEC v2.1 mục 4.1.

3 nhóm cặp TÁCH RIÊNG (occlusion mismatch giữa ảnh enrollment và ảnh test là
kịch bản thực tế nhất — enrollment thường chụp mặt sạch, verification thường
gặp mặt bị che, xem SPEC 4.1):
  - unmasked_unmasked: cả 2 ảnh đều rmfd_unmasked
  - masked_masked: cả 2 ảnh đều rmfd_masked THẬT — KHÔNG tính
    masktheface_synthetic (SPEC "B6", đã ghi nợ từ review spec trước: synthetic
    cùng phân phối với lúc train, tính vào sẽ làm chỉ số masked lạc quan giả
    tạo — đây chính là chỗ đóng nợ "B6")
  - masked_unmasked: 1 ảnh rmfd_masked (thật) + 1 ảnh rmfd_unmasked

Mỗi nhóm: genuine = 2 ảnh CÙNG identity (khác ảnh), impostor = 2 ảnh KHÁC
identity. Impostor pool đầy đủ là O(n^2) — quá lớn để tính hết trên vài nghìn
ảnh, nên sample ngẫu nhiên có seed cố định (tái lập được kết quả sweep).

Thuần Python (không torch) — EmbeddingRecord.embedding là tuple[float, ...]
(xem embed.py), test được không cần torch cài, cùng triết lý
model_setup.py/checkpoint.py/sampler.py/config.py trong src/training/.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

from src.data.labels import REAL_MASKED_SOURCE, REAL_UNMASKED_SOURCE
from src.evaluation.embed import EmbeddingRecord

PAIR_GROUPS: dict[str, tuple[str, str]] = {
    "unmasked_unmasked": (REAL_UNMASKED_SOURCE, REAL_UNMASKED_SOURCE),
    "masked_masked": (REAL_MASKED_SOURCE, REAL_MASKED_SOURCE),
    "masked_unmasked": (REAL_MASKED_SOURCE, REAL_UNMASKED_SOURCE),
}


@dataclass(frozen=True)
class Pair:
    similarity: float
    is_genuine: bool


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    # Đã L2-normalize ở embed_manifest -> dot product THUẦN = cosine similarity.
    return sum(x * y for x, y in zip(a, b, strict=True))


def build_pairs(
    records: list[EmbeddingRecord],
    *,
    max_impostor_pairs: int,
    seed: int,
) -> dict[str, list[Pair]]:
    """1 lần gọi duy nhất cho cả 3 nhóm — cùng 1 `random.Random(seed)` dùng
    xuyên suốt (không tạo mới mỗi nhóm) để kết quả sweep tái lập được toàn bộ
    bằng 1 seed duy nhất, không phải nhớ seed riêng cho từng nhóm."""
    rng = random.Random(seed)
    return {
        group_name: _build_pairs_for_group(records, source_a, source_b, max_impostor_pairs, rng)
        for group_name, (source_a, source_b) in PAIR_GROUPS.items()
    }


def _build_pairs_for_group(
    records: list[EmbeddingRecord],
    source_a: str,
    source_b: str,
    max_impostor_pairs: int,
    rng: random.Random,
) -> list[Pair]:
    pool_a = [r for r in records if r.source == source_a]
    pool_b = [r for r in records if r.source == source_b]

    by_identity_a: dict[str, list[EmbeddingRecord]] = {}
    for r in pool_a:
        by_identity_a.setdefault(r.identity_id, []).append(r)
    by_identity_b: dict[str, list[EmbeddingRecord]] = {}
    for r in pool_b:
        by_identity_b.setdefault(r.identity_id, []).append(r)

    pairs: list[Pair] = []

    # Genuine: mọi cặp (a, b) CÙNG identity, KHÁC ảnh.
    # BUG ĐÃ SỬA (phát hiện bằng test thật): khi source_a == source_b (nhóm
    # unmasked_unmasked/masked_masked), a_list và b_list là CÙNG 1 danh sách —
    # itertools.product đếm mỗi cặp KHÔNG có thứ tự 2 LẦN, vd (u1,u2) VÀ
    # (u2,u1), dù đó cùng 1 cặp ảnh, cùng 1 giá trị similarity. Dùng
    # itertools.combinations (không lặp, không có thứ tự) cho trường hợp này —
    # tự động không bao giờ ghép 1 ảnh với chính nó, không cần check `a is b`
    # nữa. Khi source_a != source_b (masked_unmasked), 2 pool tách biệt hoàn
    # toàn (1 ảnh chỉ có 1 source) nên product không có rủi ro trùng lặp này.
    for identity, a_list in by_identity_a.items():
        if source_a == source_b:
            candidate_pairs = list(itertools.combinations(a_list, 2))
        else:
            candidate_pairs = list(itertools.product(a_list, by_identity_b.get(identity, [])))
        for a, b in candidate_pairs:
            pairs.append(Pair(_cosine(a.embedding, b.embedding), is_genuine=True))

    # Impostor: sample ngẫu nhiên (O(n^2) đầy đủ quá lớn) — cap max_impostor_pairs.
    identities_a = list(by_identity_a.keys())
    identities_b = list(by_identity_b.keys())
    count = 0
    attempts = 0
    max_attempts = max(max_impostor_pairs * 20, 100)  # chặn vòng lặp vô hạn nếu quá ít identity
    # BUG ĐÃ SỬA (phát hiện bằng test thật, không phải suy đoán): điều kiện cũ
    # `len({*identities_a, *identities_b}) >= 2` sai khi 1 trong 2 pool RỖNG —
    # vd identities_a=[] (group masked_unmasked nhưng không có ảnh masked nào),
    # identities_b=["id1","id2"] -> union có 2 phần tử nên check cũ cho qua,
    # nhưng rng.choice(identities_a) trên list rỗng crash ngay lập tức. Chỉ cần
    # cả 2 pool non-empty là đủ an toàn — trường hợp "chỉ có 1 identity trùng ở
    # cả 2 bên" tự nhiên bị lọc bởi "if id_a == id_b: continue" bên dưới trong
    # giới hạn max_attempts, không cần tính trước chính xác có tạo được pair
    # khác identity hay không.
    can_form_impostor = bool(identities_a) and bool(identities_b)
    while count < max_impostor_pairs and attempts < max_attempts and can_form_impostor:
        attempts += 1
        id_a = rng.choice(identities_a)
        id_b = rng.choice(identities_b)
        if id_a == id_b:
            continue
        a = rng.choice(by_identity_a[id_a])
        b = rng.choice(by_identity_b[id_b])
        pairs.append(Pair(_cosine(a.embedding, b.embedding), is_genuine=False))
        count += 1

    return pairs

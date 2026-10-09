"""TAR@FAR sweep + khoá ngưỡng — SPEC v2.1 mục 4.2.

Quy trình BẮT BUỘC (không chọn threshold tuỳ ý — xem SPEC 4.2):
  1. `sweep_tar_at_far` trên pairs của tập VAL (identity-disjoint) -> khoá 1
     threshold tại FAR mục tiêu (mặc định 1%).
  2. `evaluate_at_threshold` dùng ĐÚNG threshold đó re-validate (KHÔNG tinh
     chỉnh lại) trên pairs của tập TEST — tránh leakage do tối ưu threshold
     trên chính tập dùng để báo cáo kết quả cuối.

Thuần Python (không torch) — cùng triết lý pairs.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.evaluation.pairs import Pair


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    target_far: float
    achieved_far: float
    achieved_tar: float
    num_genuine: int
    num_impostor: int


def sweep_tar_at_far(pairs: list[Pair], target_far: float) -> ThresholdResult:
    """Tìm threshold sao cho FAR (tỷ lệ impostor pair bị chấp nhận NHẦM, tức
    similarity >= threshold) KHÔNG VƯỢT target_far, rồi báo cáo TAR (tỷ lệ
    genuine pair similarity >= threshold) tại đúng threshold đó.

    Thuật toán: sort similarity của impostor giảm dần, threshold = giá trị tại
    vị trí floor(target_far * num_impostor) từ trên xuống — cách chuẩn để chọn
    threshold cho 1 FAR mục tiêu cụ thể trên tập hữu hạn (không nội suy liên
    tục giữa 2 điểm), đảm bảo achieved_far <= target_far (không bao giờ vượt,
    chỉ có thể THẤP hơn do làm tròn xuống).
    """
    genuine = [p.similarity for p in pairs if p.is_genuine]
    impostor = [p.similarity for p in pairs if not p.is_genuine]
    if not genuine:
        raise ValueError("Không có genuine pair nào — không tính được TAR")
    if not impostor:
        raise ValueError("Không có impostor pair nào — không tính được FAR")
    if not 0 < target_far < 1:
        raise ValueError(f"target_far phải nằm trong (0, 1), nhận {target_far}")

    impostor_sorted = sorted(impostor, reverse=True)
    num_accept = int(target_far * len(impostor_sorted))  # floor — xem docstring
    if num_accept == 0:
        # target_far quá nhỏ so với num_impostor để chấp nhận dù chỉ 1 impostor
        # -> threshold CHẶT hơn impostor cao nhất, FAR thật = 0.
        threshold = impostor_sorted[0] + 1e-6
    else:
        threshold = impostor_sorted[num_accept - 1]

    achieved_far = sum(1 for s in impostor if s >= threshold) / len(impostor)
    achieved_tar = sum(1 for s in genuine if s >= threshold) / len(genuine)

    return ThresholdResult(
        threshold=threshold,
        target_far=target_far,
        achieved_far=achieved_far,
        achieved_tar=achieved_tar,
        num_genuine=len(genuine),
        num_impostor=len(impostor),
    )


def evaluate_at_threshold(pairs: list[Pair], threshold: float) -> tuple[float, float]:
    """Re-validate threshold ĐÃ KHOÁ (từ val) lên 1 tập KHÁC (test) — KHÔNG
    tinh chỉnh lại threshold ở đây, chỉ đo. Trả (tar, far); NaN nếu nhóm đó
    không có đủ pair genuine/impostor (vd masked_masked rỗng vì test-split
    không có identity nào còn ảnh masked thật — xem SPEC "B5")."""
    genuine = [p.similarity for p in pairs if p.is_genuine]
    impostor = [p.similarity for p in pairs if not p.is_genuine]
    tar = sum(1 for s in genuine if s >= threshold) / len(genuine) if genuine else float("nan")
    far = sum(1 for s in impostor if s >= threshold) / len(impostor) if impostor else float("nan")
    return tar, far

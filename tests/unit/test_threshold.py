"""Unit test cho src/evaluation/threshold.py. Thuần Python — KHÔNG cần torch,
chạy được trong CI lint-test job."""

import pytest

from src.evaluation.pairs import Pair
from src.evaluation.threshold import evaluate_at_threshold, sweep_tar_at_far


def _pairs(genuine_similarities: list[float], impostor_similarities: list[float]) -> list[Pair]:
    return [Pair(s, is_genuine=True) for s in genuine_similarities] + [
        Pair(s, is_genuine=False) for s in impostor_similarities
    ]


def test_sweep_achieves_at_most_the_target_far():
    """Thuộc tính BẮT BUỘC theo thiết kế (SPEC 4.2): achieved_far KHÔNG BAO GIỜ
    vượt target_far (được phép thấp hơn do làm tròn xuống, không được cao hơn)."""
    impostor = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]  # 10 impostor
    genuine = [0.95, 0.85, 0.75]
    pairs = _pairs(genuine, impostor)

    result = sweep_tar_at_far(
        pairs, target_far=0.1
    )  # 10% của 10 impostor = 1 impostor được chấp nhận

    assert result.achieved_far <= 0.1 + 1e-9
    assert result.threshold == 0.9  # impostor cao nhất -> threshold = giá trị đó


def test_sweep_threshold_at_zero_accept_count_is_stricter_than_max_impostor():
    """target_far quá nhỏ so với num_impostor (floor ra 0 impostor được chấp
    nhận) -> threshold phải CHẶT hơn impostor similarity cao nhất, FAR thật = 0."""
    impostor = [0.9, 0.5, 0.1]  # 3 impostor, target_far=0.1 -> floor(0.3)=0
    genuine = [0.95]
    pairs = _pairs(genuine, impostor)

    result = sweep_tar_at_far(pairs, target_far=0.1)

    assert result.threshold > 0.9
    assert result.achieved_far == 0.0


def test_sweep_tar_reflects_genuine_separation_from_impostor():
    """Genuine tách biệt rõ khỏi impostor (không chồng lấn) -> TAR phải = 1.0
    tại threshold chọn được (mọi genuine similarity đều > threshold)."""
    impostor = [0.3, 0.2, 0.1, 0.05, 0.0]
    genuine = [0.99, 0.98, 0.97, 0.96, 0.95]
    pairs = _pairs(genuine, impostor)

    result = sweep_tar_at_far(pairs, target_far=0.2)  # chấp nhận 1/5 impostor

    assert result.achieved_tar == 1.0


def test_sweep_raises_without_genuine_pairs():
    pairs = _pairs(genuine_similarities=[], impostor_similarities=[0.5, 0.3])
    with pytest.raises(ValueError, match="genuine"):
        sweep_tar_at_far(pairs, target_far=0.1)


def test_sweep_raises_without_impostor_pairs():
    pairs = _pairs(genuine_similarities=[0.9], impostor_similarities=[])
    with pytest.raises(ValueError, match="impostor"):
        sweep_tar_at_far(pairs, target_far=0.1)


@pytest.mark.parametrize("bad_far", [0.0, 1.0, -0.1, 1.5])
def test_sweep_rejects_target_far_outside_open_interval(bad_far):
    pairs = _pairs(genuine_similarities=[0.9], impostor_similarities=[0.1])
    with pytest.raises(ValueError, match="target_far"):
        sweep_tar_at_far(pairs, target_far=bad_far)


def test_evaluate_at_threshold_does_not_retune_just_measures():
    """Re-validate trên tập KHÁC (test split) bằng threshold ĐÃ KHOÁ — khác hẳn
    sweep_tar_at_far (không tự chọn threshold mới)."""
    val_pairs = _pairs(genuine_similarities=[0.9, 0.8], impostor_similarities=[0.5, 0.1])
    locked = sweep_tar_at_far(val_pairs, target_far=0.5).threshold

    test_pairs = _pairs(genuine_similarities=[0.95, 0.4], impostor_similarities=[0.6, 0.2])
    tar, far = evaluate_at_threshold(test_pairs, locked)

    # Đo thủ công bằng đúng threshold đã khoá để đối chiếu, không suy đoán
    expected_tar = sum(1 for s in [0.95, 0.4] if s >= locked) / 2
    expected_far = sum(1 for s in [0.6, 0.2] if s >= locked) / 2
    assert tar == expected_tar
    assert far == expected_far


def test_evaluate_at_threshold_returns_nan_when_group_has_no_pairs():
    """Nhóm rỗng (vd masked_masked của 1 split không có identity nào còn ảnh
    masked thật — SPEC 'B5') không được crash, trả NaN để caller tự xử lý."""
    import math

    tar, far = evaluate_at_threshold([], threshold=0.5)

    assert math.isnan(tar)
    assert math.isnan(far)

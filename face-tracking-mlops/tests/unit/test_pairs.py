"""Unit test cho src/evaluation/pairs.py. Thuần Python — KHÔNG cần torch, chạy
được trong CI lint-test job (không cài torch)."""

from src.evaluation.embed import EmbeddingRecord
from src.evaluation.pairs import build_pairs

# Vector đơn vị 2D ở 2 góc biết trước -> cosine similarity tính tay được:
# SAME: dot = 1.0 (trùng hướng) | ORTHO: dot = 0.0 (vuông góc)
SAME_A = (1.0, 0.0)
SAME_B = (1.0, 0.0)
ORTHO = (0.0, 1.0)


def _rec(identity: str, source: str, vec: tuple[float, float]) -> EmbeddingRecord:
    return EmbeddingRecord(identity_id=identity, source=source, embedding=vec)


def test_masked_masked_group_excludes_synthetic_entirely():
    """Đúng chỗ đóng nợ SPEC 'B6': masked_masked CHỈ gồm rmfd_masked thật,
    không bao giờ lẫn masktheface_synthetic — kể cả khi synthetic chiếm đa số
    dữ liệu (RMFD lệch nặng, xem src/data/labels.py)."""
    records = [
        _rec("id1", "rmfd_masked", SAME_A),
        _rec("id1", "masktheface_synthetic", SAME_A),
        _rec("id1", "masktheface_synthetic", SAME_A),
        _rec("id2", "rmfd_masked", SAME_B),
        _rec("id2", "masktheface_synthetic", SAME_B),
    ]

    groups = build_pairs(records, max_impostor_pairs=1, seed=0)

    # Chỉ 2 ảnh rmfd_masked thật (id1, id2) -> không đủ để tạo genuine pair
    # (mỗi identity chỉ có 1 ảnh thật), chỉ có impostor pair giữa id1/id2.
    assert all(not p.is_genuine for p in groups["masked_masked"])
    assert len(groups["masked_masked"]) == 1  # max_impostor_pairs=1


def test_unmasked_unmasked_group_never_includes_masked_source():
    records = [
        _rec("id1", "rmfd_unmasked", SAME_A),
        _rec("id1", "rmfd_unmasked", SAME_A),
        _rec("id1", "rmfd_masked", ORTHO),  # không được lọt vào unmasked_unmasked
    ]

    groups = build_pairs(records, max_impostor_pairs=100, seed=0)

    assert len(groups["unmasked_unmasked"]) == 1  # 1 genuine pair (2 ảnh unmasked cùng id1)
    assert groups["unmasked_unmasked"][0].similarity == 1.0  # SAME_A . SAME_A = 1.0


def test_masked_unmasked_group_pairs_across_the_two_sources():
    records = [
        _rec("id1", "rmfd_masked", SAME_A),
        _rec("id1", "rmfd_unmasked", SAME_B),  # cùng hướng -> cosine = 1.0
    ]

    groups = build_pairs(records, max_impostor_pairs=100, seed=0)

    assert len(groups["masked_unmasked"]) == 1
    assert groups["masked_unmasked"][0].is_genuine is True
    assert abs(groups["masked_unmasked"][0].similarity - 1.0) < 1e-9


def test_genuine_pairs_never_pair_an_image_with_itself():
    """1 identity chỉ có DUY NHẤT 1 ảnh unmasked -> không có genuine pair nào
    (không được tự ghép ảnh đó với chính nó)."""
    records = [_rec("id1", "rmfd_unmasked", SAME_A)]

    groups = build_pairs(records, max_impostor_pairs=100, seed=0)

    assert groups["unmasked_unmasked"] == []


def test_impostor_pairs_only_between_different_identities():
    records = [
        _rec("id1", "rmfd_unmasked", SAME_A),
        _rec("id1", "rmfd_unmasked", SAME_A),
        _rec("id2", "rmfd_unmasked", ORTHO),
    ]

    groups = build_pairs(records, max_impostor_pairs=100, seed=0)
    impostors = [p for p in groups["unmasked_unmasked"] if not p.is_genuine]

    assert all(abs(p.similarity - 0.0) < 1e-9 for p in impostors)  # SAME_A . ORTHO = 0.0
    assert len(impostors) > 0


def test_max_impostor_pairs_caps_the_count():
    records = [_rec(f"id{i}", "rmfd_unmasked", (1.0, 0.0)) for i in range(20)]

    groups = build_pairs(records, max_impostor_pairs=5, seed=0)
    impostors = [p for p in groups["unmasked_unmasked"] if not p.is_genuine]

    assert len(impostors) == 5


def test_same_seed_gives_identical_impostor_sample():
    records = [_rec(f"id{i}", "rmfd_unmasked", (float(i), 1.0)) for i in range(10)]

    g1 = build_pairs(records, max_impostor_pairs=5, seed=7)
    g2 = build_pairs(records, max_impostor_pairs=5, seed=7)

    assert g1["unmasked_unmasked"] == g2["unmasked_unmasked"]


def test_no_impostor_possible_when_single_identity_returns_empty_not_crash():
    records = [
        _rec("id1", "rmfd_unmasked", SAME_A),
        _rec("id1", "rmfd_unmasked", SAME_A),
    ]

    groups = build_pairs(records, max_impostor_pairs=100, seed=0)
    impostors = [p for p in groups["unmasked_unmasked"] if not p.is_genuine]

    assert impostors == []  # không crash, không vòng lặp vô hạn

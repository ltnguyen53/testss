"""Unit test cho src/training/sampler.py — SPEC v2 mục 13 ("Sampler fallback").

Fixture 5 identity mô phỏng đúng vấn đề dữ liệu thật của RMFD (mục 3.3): phân bố
KHÔNG đều — có identity đủ ảnh mask thật, có identity chỉ 1 ảnh mask thật, có
identity không có ảnh mask thật nào (kể cả synthetic).
"""

import random

import pytest

from src.data.labels import ImageRecord, write_manifest_csv
from src.training.sampler import PKBatchSampler, pick_for_identity

K = 4
MASKED = 2


def _rows(identity_id, real_masked=0, synthetic=0, unmasked=0):
    rows = []
    for i in range(real_masked):
        rows.append(
            ImageRecord(f"raw/{identity_id}/real_m{i}.jpg", identity_id, True, "rmfd_masked")
        )
    for i in range(synthetic):
        rows.append(
            ImageRecord(
                f"proc/{identity_id}/syn{i}.jpg", identity_id, True, "masktheface_synthetic"
            )
        )
    for i in range(unmasked):
        rows.append(
            ImageRecord(f"raw/{identity_id}/um{i}.jpg", identity_id, False, "rmfd_unmasked")
        )
    return rows


@pytest.fixture
def identities():
    return {
        "A": _rows("A", real_masked=1, synthetic=5, unmasked=10),  # cần bù synthetic
        "B": _rows("B", real_masked=6, synthetic=0, unmasked=20),  # đủ mask thật
        "C": _rows("C", real_masked=0, synthetic=0, unmasked=12),  # không có masked nào
        "D": _rows("D", real_masked=3, synthetic=0, unmasked=1),  # tổng đúng = K
        "E": _rows("E", real_masked=0, synthetic=0, unmasked=1),  # chỉ 1 ảnh
    }


@pytest.fixture
def manifest_csv(tmp_path, identities):
    path = tmp_path / "train.csv"
    write_manifest_csv(path, [r for rows in identities.values() for r in rows])
    return path


def _sampler(manifest_csv, **kw):
    args = dict(identities_per_batch=2, images_per_identity=K, masked_per_identity=MASKED, seed=7)
    args.update(kw)
    return PKBatchSampler(manifest_csv, **args)


def test_fallback_uses_synthetic_of_same_identity(identities):
    rng = random.Random(0)
    picked = pick_for_identity(identities["A"], k=K, masked_per_identity=MASKED, rng=rng)

    assert len(picked) == K
    assert all(r.identity_id == "A" for r in picked)
    assert sum(r.source == "rmfd_masked" for r in picked) == 1  # dùng hết ảnh mask thật
    assert sum(r.source == "masktheface_synthetic" for r in picked) == 1  # bù đúng phần thiếu
    assert sum(not r.is_masked for r in picked) == 2
    assert {r.image_path for r in picked} <= {r.image_path for r in identities["A"]}


def test_mix_of_masked_and_unmasked_when_enough_real_masked(identities):
    picked = pick_for_identity(
        identities["B"], k=K, masked_per_identity=MASKED, rng=random.Random(1)
    )

    assert sum(r.is_masked for r in picked) == MASKED
    assert sum(not r.is_masked for r in picked) == K - MASKED
    assert all(r.source != "masktheface_synthetic" for r in picked)  # không cần synthetic


def test_identity_without_any_masked_falls_back_to_unmasked_without_crash(identities):
    picked = pick_for_identity(
        identities["C"], k=K, masked_per_identity=MASKED, rng=random.Random(2)
    )

    assert len(picked) == K
    assert all(not r.is_masked for r in picked)
    assert len({r.image_path for r in picked}) == K  # đủ ảnh unmasked nên không lặp


def test_no_duplicate_images_when_identity_has_exactly_k(identities):
    picked = pick_for_identity(
        identities["D"], k=K, masked_per_identity=MASKED, rng=random.Random(3)
    )

    assert len({r.image_path for r in picked}) == K  # dùng nốt masked dư thay vì lặp ảnh


def test_identity_with_single_image_repeats_with_replacement(identities):
    picked = pick_for_identity(
        identities["E"], k=K, masked_per_identity=MASKED, rng=random.Random(4)
    )

    assert len(picked) == K
    assert len({r.image_path for r in picked}) == 1


def test_batches_never_mix_identities_across_epochs(manifest_csv):
    sampler = _sampler(manifest_csv)
    for epoch in range(20):
        for batch in sampler.epoch_batches(epoch):
            assert len(batch) == 2 * K
            for block_start in range(0, len(batch), K):
                block = batch[block_start : block_start + K]
                identity_ids = {r.identity_id for r in block}
                assert len(identity_ids) == 1
                (iid,) = identity_ids
                # ảnh phải nằm đúng thư mục của identity đó (raw/<iid>/ hoặc proc/<iid>/)
                assert all(f"/{iid}/" in r.image_path for r in block)


def test_batch_count_with_and_without_drop_last(manifest_csv):
    dropped = _sampler(manifest_csv, drop_last=True)
    kept = _sampler(manifest_csv, drop_last=False)

    assert len(dropped) == 2 and len(dropped.epoch_batches(0)) == 2
    assert len(kept) == 3 and len(kept.epoch_batches(0)) == 3
    assert len(kept.epoch_batches(0)[-1]) == 1 * K  # batch cuối chỉ có 1 identity


def test_epoch_batches_deterministic_by_seed_and_epoch(manifest_csv):
    s1 = _sampler(manifest_csv)
    s2 = _sampler(manifest_csv)

    assert s1.epoch_batches(3) == s2.epoch_batches(3)
    assert any(s1.epoch_batches(0) != s1.epoch_batches(e) for e in range(1, 6))


def test_coverage_report_flags_identities_needing_fallback(manifest_csv):
    sampler = _sampler(manifest_csv)

    # real_masked < MASKED(=2): C(0), E(0), A(1) — sort theo real_masked tăng dần
    assert sampler.identities_needing_fallback() == ["C", "E", "A"]
    # không đủ kể cả synthetic: C và E
    assert sampler.identities_unable_to_fill_masked() == ["C", "E"]
    assert sampler.coverage[0].real_masked == 0  # identity yếu nhất đứng đầu


def test_head_size_equals_number_of_identities_in_manifest(manifest_csv):
    sampler = _sampler(manifest_csv)

    assert sampler.num_classes == 5
    assert sampler.identity_to_class == {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def test_invalid_config_raises(manifest_csv):
    with pytest.raises(ValueError):
        _sampler(manifest_csv, identities_per_batch=99)
    with pytest.raises(ValueError):
        _sampler(manifest_csv, masked_per_identity=K + 1)

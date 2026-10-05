"""Unit test cho src/data/labels.py — Module 4.2 (data test), mức intern bắt buộc.

Không dùng dataset thật (không có mạng/RMFD trong CI) — dựng fixture nhỏ bằng
file rỗng, đủ để test LOGIC split/manifest, không test đọc nội dung ảnh thật.
"""

from pathlib import Path

import pytest

from src.data.labels import (
    ImageRecord,
    build_manifest,
    group_by_identity,
    largest_remainder_alloc,
    mask_ratio,
    real_count,
    real_masked_bin,
    records_for_identities,
    run,
    split_identities,
)


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00")


@pytest.fixture
def fixture_cfg(tmp_path: Path) -> dict:
    # id001-004: mask_ratio cao (masked nhiều hơn unmasked)
    for i in range(1, 5):
        iid = f"id{i:03d}"
        _touch(tmp_path / "rmfd_masked" / iid / "a.jpg")
        _touch(tmp_path / "rmfd_masked" / iid / "b.jpg")
        _touch(tmp_path / "rmfd_unmasked" / iid / "a.jpg")
    # id005-008: mask_ratio thấp
    for i in range(5, 9):
        iid = f"id{i:03d}"
        _touch(tmp_path / "rmfd_unmasked" / iid / "a.jpg")
        _touch(tmp_path / "rmfd_unmasked" / iid / "b.jpg")
        _touch(tmp_path / "rmfd_masked" / iid / "a.jpg")
    # id009-010: 50/50, có ảnh synthetic
    for i in range(9, 11):
        iid = f"id{i:03d}"
        _touch(tmp_path / "rmfd_masked" / iid / "a.jpg")
        _touch(tmp_path / "rmfd_unmasked" / iid / "a.jpg")
        _touch(tmp_path / "masktheface_rmfd" / iid / "a_surgical.jpg")
    # id011: 1 ảnh -> phải bị drop (min_images_per_identity=2)
    _touch(tmp_path / "rmfd_masked" / "id011" / "a.jpg")
    # ảnh rời không có subfolder identity -> phải bị bỏ qua hoàn toàn
    _touch(tmp_path / "rmfd_masked" / "loose_image.jpg")

    # enrollment — không được lẫn vào metric_records
    _touch(tmp_path / "enrollment" / "self" / "me1.jpg")
    _touch(tmp_path / "enrollment" / "self" / "me2.jpg")
    _touch(tmp_path / "masktheface_enrollment" / "self" / "me1_surgical.jpg")

    return {
        "seed": 42,
        "paths": {
            "rmfd_masked_dir": str(tmp_path / "rmfd_masked"),
            "rmfd_unmasked_dir": str(tmp_path / "rmfd_unmasked"),
            "enrollment_dir": str(tmp_path / "enrollment"),
            "masktheface_rmfd_out_dir": str(tmp_path / "masktheface_rmfd"),
            "masktheface_enrollment_out_dir": str(tmp_path / "masktheface_enrollment"),
            "splits_dir": str(tmp_path / "out_splits"),
        },
        "split": {
            "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "real_masked_thresholds": [1, 2],
            "min_images_per_identity": 2,
        },
    }


def test_build_manifest_separates_enrollment_from_metrics(fixture_cfg):
    metric_records, enrollment_records = build_manifest(fixture_cfg)
    metric_ids = {r.identity_id for r in metric_records}

    assert "loose_image" not in metric_ids
    assert "self" not in metric_ids
    assert len(enrollment_records) == 3


def test_split_is_identity_disjoint_and_drops_low_count_identity(fixture_cfg):
    metric_records, _ = build_manifest(fixture_cfg)
    grouped = group_by_identity(metric_records)

    split_ids, dropped = split_identities(
        grouped,
        ratios=fixture_cfg["split"]["ratios"],
        seed=fixture_cfg["seed"],
        real_masked_thresholds=fixture_cfg["split"]["real_masked_thresholds"],
        min_images_per_identity=fixture_cfg["split"]["min_images_per_identity"],
    )

    assert dropped == ["id011"]

    all_assigned = split_ids["train"] + split_ids["val"] + split_ids["test"]
    assert sorted(all_assigned) == sorted(f"id{i:03d}" for i in range(1, 11))

    seen: set[str] = set()
    for name in ("train", "val", "test"):
        s = set(split_ids[name])
        assert not (s & seen), f"identity leak: {s & seen}"
        seen |= s


def test_synthetic_augment_follows_source_identity_into_same_split(fixture_cfg):
    metric_records, _ = build_manifest(fixture_cfg)
    grouped = group_by_identity(metric_records)
    split_ids, _ = split_identities(
        grouped,
        ratios=fixture_cfg["split"]["ratios"],
        seed=fixture_cfg["seed"],
        real_masked_thresholds=fixture_cfg["split"]["real_masked_thresholds"],
        min_images_per_identity=fixture_cfg["split"]["min_images_per_identity"],
    )
    for name in ("train", "val", "test"):
        recs = records_for_identities(grouped, split_ids[name])
        synth_ids = {r.identity_id for r in recs if r.source == "masktheface_synthetic"}
        assert synth_ids <= set(split_ids[name])


def test_same_seed_gives_identical_split(fixture_cfg):
    metric_records, _ = build_manifest(fixture_cfg)
    grouped = group_by_identity(metric_records)
    kwargs = dict(
        grouped=grouped,
        ratios=fixture_cfg["split"]["ratios"],
        seed=42,
        real_masked_thresholds=fixture_cfg["split"]["real_masked_thresholds"],
        min_images_per_identity=fixture_cfg["split"]["min_images_per_identity"],
    )
    split_a, _ = split_identities(**kwargs)
    split_b, _ = split_identities(**kwargs)
    assert split_a == split_b


def test_real_masked_bin_separates_identities_that_mask_ratio_would_conflate():
    """Regression test cho bug đã sửa (SPEC v2.1 "B4"): identity A (0 ảnh mask
    thật, nhiều synthetic) và identity B (5 ảnh mask thật, cùng lượng synthetic)
    có mask_ratio GẦN NHƯ BẰNG NHAU (vì synthetic áp đảo cả 2) — bin theo
    mask_ratio liên tục sẽ gộp cả 2 vào cùng 1 bin, làm stratify vô tác dụng.
    real_masked_bin() phải tách rõ 2 identity này vì đó chính là điều mask_ratio
    không làm được.
    """

    def _make(n_real_masked: int, n_unmasked: int, n_synthetic: int) -> list[ImageRecord]:
        recs = [ImageRecord(f"m/{i}.jpg", "x", True, "rmfd_masked") for i in range(n_real_masked)]
        recs += [ImageRecord(f"u/{i}.jpg", "x", False, "rmfd_unmasked") for i in range(n_unmasked)]
        recs += [
            ImageRecord(f"s/{i}.jpg", "x", True, "masktheface_synthetic")
            for i in range(n_synthetic)
        ]
        return recs

    identity_a = _make(n_real_masked=0, n_unmasked=10, n_synthetic=30)
    identity_b = _make(n_real_masked=5, n_unmasked=10, n_synthetic=30)

    # mask_ratio (cách tính CŨ dùng để stratify) gần như giống hệt nhau:
    assert abs(mask_ratio(identity_a) - mask_ratio(identity_b)) < 0.05

    # nhưng real_masked_bin (cách tính MỚI) phải tách rõ 2 identity này:
    bin_a = real_masked_bin(identity_a, thresholds=[1, 5])
    bin_b = real_masked_bin(identity_b, thresholds=[1, 5])
    assert bin_a != bin_b, "real_masked_bin phải phân biệt 0 vs 5 ảnh mask thật"
    assert bin_a == 0
    assert bin_b == 2


def test_min_images_per_identity_excludes_synthetic_from_the_count(fixture_cfg):
    """Regression test cho bug đã sửa (SPEC v2.1 "B5"): identity chỉ có 1 ảnh
    THẬT + nhiều bản synthetic sinh từ chính ảnh đó (tổng len(recs) >= 2) trước
    đây KHÔNG bị drop vì min_images_per_identity đếm cả synthetic. Giờ phải bị
    drop vì real_count (chỉ ảnh thật) mới là thứ được so với ngưỡng.
    """
    recs_one_real_plus_synthetic = [
        ImageRecord("m/a.jpg", "id_only_1_real", True, "rmfd_masked"),
        ImageRecord("s/a1.jpg", "id_only_1_real", True, "masktheface_synthetic"),
        ImageRecord("s/a2.jpg", "id_only_1_real", True, "masktheface_synthetic"),
        ImageRecord("s/a3.jpg", "id_only_1_real", True, "masktheface_synthetic"),
    ]
    assert len(recs_one_real_plus_synthetic) == 4  # tổng ảnh (kiểu đếm CŨ) >= min=2
    assert real_count(recs_one_real_plus_synthetic) == 1  # ảnh THẬT (kiểu đếm MỚI) < min=2

    grouped = {"id_only_1_real": recs_one_real_plus_synthetic}
    _split_ids, dropped = split_identities(
        grouped,
        ratios=fixture_cfg["split"]["ratios"],
        seed=fixture_cfg["seed"],
        real_masked_thresholds=fixture_cfg["split"]["real_masked_thresholds"],
        min_images_per_identity=fixture_cfg["split"]["min_images_per_identity"],
    )
    assert dropped == ["id_only_1_real"]


def test_run_writes_expected_csv_files(fixture_cfg):
    run(fixture_cfg)
    splits_dir = Path(fixture_cfg["paths"]["splits_dir"])

    train_csv = splits_dir / "train.csv"
    enrollment_csv = splits_dir / "enrollment_demo.csv"
    assert train_csv.exists()
    assert enrollment_csv.exists()

    header = train_csv.read_text(encoding="utf-8").splitlines()[0]
    assert header == "image_path,identity_id,is_masked,source"


@pytest.mark.parametrize(
    ("n", "ratios", "expected_total"),
    [
        (10, {"train": 0.7, "val": 0.15, "test": 0.15}, 10),
        (1, {"train": 0.7, "val": 0.15, "test": 0.15}, 1),
        (2, {"train": 0.5, "val": 0.5}, 2),
    ],
)
def test_largest_remainder_alloc_sums_to_n(n, ratios, expected_total):
    result = largest_remainder_alloc(n, ratios)
    assert sum(result.values()) == expected_total
    assert all(v >= 0 for v in result.values())

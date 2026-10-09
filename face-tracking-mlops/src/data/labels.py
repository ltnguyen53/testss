"""Phase 1 — build manifest hợp nhất từ raw + processed, split identity-disjoint.

Chạy: python -m src.data.labels --config configs/data.yaml
Là 1 DVC stage (`make_splits` trong dvc.yaml) — không tự gọi dvc/mlflow, chỉ đọc/ghi
theo path lấy từ config (nguyên tắc Module 0: code thuần không biết hạ tầng).

Quyết định đã chốt (không tự đổi khi sửa file này — sửa thì phải nói rõ lý do):
- split theo identity_id (identity_disjoint), KHÔNG theo ảnh — bài toán là open-set
  verification, chia theo ảnh gây identity leakage (xem giải thích đầy đủ trong lịch
  sử quyết định dự án, không lặp lại ở đây).
- stratify identity theo SỐ ẢNH MASKED THẬT (rmfd_masked, không tính synthetic) để
  3 split có identity "khan hiếm ảnh mask thật" phân bố đều — SỬA 2026-09-28 (SPEC
  v2.1 "B4"): trước đây stratify theo mask_ratio LIÊN TỤC (gồm cả synthetic), nhưng
  vì mỗi ảnh unmasked sinh ra nhiều bản synthetic (đúng bằng số mask_type), synthetic
  luôn áp đảo ảnh thật trong mọi identity theo gần như cùng 1 tỷ lệ -> toàn bộ 525
  identity rơi vào 1 bin duy nhất (đã tái hiện: 100% vào 1 bin dù tính hay không
  tính synthetic, chỉ khác bin nào) -> stratify thực chất KHÔNG có tác dụng, split
  ngẫu nhiên như không stratify. Xem real_masked_bin().
- min_images_per_identity ĐẾM ẢNH THẬT (rmfd_masked + rmfd_unmasked), KHÔNG TÍNH
  synthetic — SỬA 2026-09-28 (SPEC v2.1 "B5"): trước đây đếm cả synthetic, nên 1
  identity chỉ có 1 ảnh thật + nhiều bản synthetic sinh từ chính ảnh đó vẫn được
  giữ, dù không đủ ảnh thật độc lập để tạo cặp genuine/impostor đáng tin cho eval.
  Xem real_count().
- ảnh augment (masktheface_synthetic) tự động đi theo đúng identity_id gốc vào cùng
  1 split, vì gán split là theo identity_id chứ không theo ảnh — không cần xử lý
  thêm.
- enrollment (+ bản augment của enrollment) KHÔNG vào train/val/test — ghi riêng
  splits/enrollment_demo.csv, không tính vào bất kỳ metric báo cáo nào.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Cùng giá trị với sampler.py:SYNTHETIC_SOURCE — không import chéo giữa 2 module
# này để tránh circular import (sampler.py import từ labels.py), nên literal
# string được giữ độc lập ở cả 2 nơi. Đổi giá trị này thì phải đổi cả bên kia.
REAL_MASKED_SOURCE = "rmfd_masked"
REAL_UNMASKED_SOURCE = "rmfd_unmasked"
SYNTHETIC_SOURCE = "masktheface_synthetic"


@dataclass(frozen=True)
class ImageRecord:
    image_path: str  # relative path, posix (forward slash) — ổn định qua Windows/Linux
    identity_id: str
    is_masked: bool
    source: str
    # rmfd_masked | rmfd_unmasked | masktheface_synthetic
    # | enrollment | enrollment_masktheface


def scan_identity_root(root: Path, source: str, is_masked: bool) -> list[ImageRecord]:
    """Quét root/<identity_id>/*.{ext} -> list ImageRecord.

    Cấu trúc bắt buộc: root/<identity_id>/anh.jpg. Ảnh nằm thẳng trong root
    (không có subfolder identity) bị bỏ qua — không đoán identity_id thay.
    """
    records: list[ImageRecord] = []
    if not root.exists():
        logger.warning("Thư mục không tồn tại, bỏ qua: %s", root)
        return records
    for identity_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted(
            p
            for p in identity_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        for img in images:
            records.append(
                ImageRecord(
                    image_path=img.as_posix(),
                    identity_id=identity_dir.name,
                    is_masked=is_masked,
                    source=source,
                )
            )
    return records


def build_manifest(cfg: dict) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Trả về (metric_records, enrollment_records) — 2 nhóm tách biệt.

    metric_records đi vào train/val/test + tính metric báo cáo.
    enrollment_records CHỈ dùng demo/qualitative check, không tính metric.
    """
    paths = cfg["paths"]
    metric_records: list[ImageRecord] = []
    metric_records += scan_identity_root(
        Path(paths["rmfd_masked_dir"]), "rmfd_masked", is_masked=True
    )
    metric_records += scan_identity_root(
        Path(paths["rmfd_unmasked_dir"]), "rmfd_unmasked", is_masked=False
    )
    metric_records += scan_identity_root(
        Path(paths["masktheface_rmfd_out_dir"]), "masktheface_synthetic", is_masked=True
    )

    enrollment_records: list[ImageRecord] = []
    enrollment_records += scan_identity_root(
        Path(paths["enrollment_dir"]), "enrollment", is_masked=False
    )
    enrollment_records += scan_identity_root(
        Path(paths["masktheface_enrollment_out_dir"]),
        "enrollment_masktheface",
        is_masked=True,
    )
    return metric_records, enrollment_records


def group_by_identity(records: list[ImageRecord]) -> dict[str, list[ImageRecord]]:
    grouped: dict[str, list[ImageRecord]] = {}
    for r in records:
        grouped.setdefault(r.identity_id, []).append(r)
    return grouped


def mask_ratio(records: list[ImageRecord]) -> float:
    """Tỷ lệ ảnh masked/tổng, TÍNH CẢ synthetic — chỉ dùng để LOG/báo cáo mô tả
    tổng quan 1 split, KHÔNG dùng để stratify (xem real_masked_bin — lý do trong
    docstring module, mục "B4")."""
    if not records:
        return 0.0
    return sum(1 for r in records if r.is_masked) / len(records)


def real_count(records: list[ImageRecord]) -> int:
    """Số ảnh THẬT (rmfd_masked + rmfd_unmasked), KHÔNG tính synthetic/enrollment.

    Dùng cho min_images_per_identity (SPEC v2.1 "B5") — synthetic là ảnh SINH RA
    từ 1 ảnh thật khác trong cùng identity, không phải bằng chứng độc lập, nên
    không được tính vào ngưỡng "đủ dữ liệu" của 1 identity.
    """
    return sum(1 for r in records if r.source in (REAL_MASKED_SOURCE, REAL_UNMASKED_SOURCE))


def real_masked_count(records: list[ImageRecord]) -> int:
    """Số ảnh masked THẬT (rmfd_masked) — KHÔNG tính masktheface_synthetic."""
    return sum(1 for r in records if r.source == REAL_MASKED_SOURCE)


def real_masked_bin(records: list[ImageRecord], thresholds: list[int]) -> int:
    """Bin identity theo SỐ ẢNH MASKED THẬT (rời rạc), thay cho mask_ratio liên
    tục (SPEC v2.1 "B4" — xem docstring module để biết lý do đổi).

    `thresholds` tăng dần, vd [1, 5] -> 3 bin: {real_masked_count == 0},
    {1 <= count < 5}, {count >= 5}. Số bin = len(thresholds) + 1.
    """
    count = real_masked_count(records)
    bin_idx = 0
    for t in thresholds:
        if count >= t:
            bin_idx += 1
        else:
            break
    return bin_idx


def largest_remainder_alloc(n: int, ratios: dict[str, float]) -> dict[str, int]:
    """Chia n phần tử nguyên theo tỷ lệ `ratios`, tổng luôn = n (largest remainder method).

    `ratios` không bắt buộc tổng = 1 tuyệt đối (sai số float nhỏ chấp nhận được);
    tỷ trọng tương đối giữa các key mới là thứ quyết định kết quả.
    """
    names = list(ratios.keys())
    raw = {name: n * ratios[name] for name in names}
    floors = {name: int(raw[name]) for name in names}
    remainder = n - sum(floors.values())
    # Tie-break theo tên cố định — kết quả không phụ thuộc thứ tự dict tại runtime.
    order = sorted(names, key=lambda name: (-(raw[name] - floors[name]), name))
    for name in order[:remainder]:
        floors[name] += 1
    return floors


def split_identities(
    grouped: dict[str, list[ImageRecord]],
    ratios: dict[str, float],
    seed: int,
    real_masked_thresholds: list[int],
    min_images_per_identity: int,
) -> tuple[dict[str, list[str]], list[str]]:
    """Trả về (split_name -> list identity_id, list identity_id bị loại vì thiếu ảnh THẬT).

    `min_images_per_identity` áp lên real_count() (chỉ ảnh thật — SPEC "B5").
    `real_masked_thresholds`: xem real_masked_bin() (SPEC "B4").
    """
    kept: dict[str, list[ImageRecord]] = {}
    dropped: list[str] = []
    for identity_id, recs in grouped.items():
        if real_count(recs) < min_images_per_identity:
            dropped.append(identity_id)
        else:
            kept[identity_id] = recs

    bins: dict[int, list[str]] = {}
    for identity_id, recs in kept.items():
        b = real_masked_bin(recs, real_masked_thresholds)
        bins.setdefault(b, []).append(identity_id)

    rng = random.Random(seed)
    result: dict[str, list[str]] = {name: [] for name in ratios}
    for b in sorted(bins):
        ids = sorted(bins[b])  # sort trước shuffle -> deterministic bất kể thứ tự filesystem
        rng.shuffle(ids)
        counts = largest_remainder_alloc(len(ids), ratios)
        cursor = 0
        for name in ratios:  # thứ tự cắt lát cố định theo thứ tự khai báo trong config
            take = counts[name]
            result[name].extend(ids[cursor : cursor + take])
            cursor += take

    return result, sorted(dropped)


def records_for_identities(
    grouped: dict[str, list[ImageRecord]], identity_ids: list[str]
) -> list[ImageRecord]:
    out: list[ImageRecord] = []
    for identity_id in identity_ids:
        out.extend(grouped[identity_id])
    return out


def read_manifest_csv(path: Path) -> list[ImageRecord]:
    """Đọc lại manifest CSV do write_manifest_csv ghi ra — inverse function.

    Dùng ở Phase 2 (src/training/sampler.py) để load train.csv/val.csv. is_masked
    đọc lại từ text "True"/"False" (csv.DictWriter ghi qua str() mặc định).
    """
    records: list[ImageRecord] = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                ImageRecord(
                    image_path=row["image_path"],
                    identity_id=row["identity_id"],
                    is_masked=row["is_masked"] == "True",
                    source=row["source"],
                )
            )
    return records


def write_manifest_csv(path: Path, records: list[ImageRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "identity_id", "is_masked", "source"])
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


def log_identity_composition(grouped: dict[str, list[ImageRecord]]) -> None:
    """Bổ sung 2026-09-28 (SPEC v2.1 "B5") — báo cáo 1 lần, TRƯỚC khi split, số
    identity thiếu hẳn 1 trong 2 loại ảnh thật. Thiếu real_masked là bình thường
    (RMFD lệch nặng, xem docstring sampler.py), nhưng thiếu real_unmasked hoàn
    toàn (identity chỉ có ảnh trong rmfd_masked/ mà không có trong rmfd_unmasked/)
    là dấu hiệu lệch identity_id giữa 2 thư mục nguồn — cần xem lại dữ liệu thô,
    không phải phân phối tự nhiên của RMFD.
    """
    total = len(grouped)
    zero_real_masked = sum(1 for recs in grouped.values() if real_masked_count(recs) == 0)
    zero_real_unmasked = sum(
        1
        for recs in grouped.values()
        if sum(1 for r in recs if r.source == REAL_UNMASKED_SOURCE) == 0
    )
    logger.info(
        "Thành phần dữ liệu (%d identity): %d không có ảnh masked THẬT nào "
        "(bình thường với RMFD) | %d không có ảnh unmasked THẬT nào (BẤT THƯỜNG "
        "— kiểm tra lệch identity_id giữa rmfd_masked_dir và rmfd_unmasked_dir)",
        total,
        zero_real_masked,
        zero_real_unmasked,
    )


def run(cfg: dict) -> None:
    metric_records, enrollment_records = build_manifest(cfg)
    grouped = group_by_identity(metric_records)
    log_identity_composition(grouped)

    split_cfg = cfg["split"]
    split_ids, dropped = split_identities(
        grouped,
        ratios=split_cfg["ratios"],
        seed=cfg["seed"],
        real_masked_thresholds=split_cfg["real_masked_thresholds"],
        min_images_per_identity=split_cfg["min_images_per_identity"],
    )
    if dropped:
        logger.warning(
            "Loại %d identity vì < %d ảnh THẬT (không tính synthetic): %s",
            len(dropped),
            split_cfg["min_images_per_identity"],
            dropped,
        )

    splits_dir = Path(cfg["paths"]["splits_dir"])
    for name, ids in split_ids.items():
        recs = records_for_identities(grouped, ids)
        write_manifest_csv(splits_dir / f"{name}.csv", recs)
        logger.info(
            "%s: %d identity, %d ảnh (%d ảnh masked THẬT), mask_ratio(gồm synthetic)=%.3f",
            name,
            len(ids),
            len(recs),
            sum(1 for r in recs if r.source == REAL_MASKED_SOURCE),
            mask_ratio(recs),
        )

    write_manifest_csv(splits_dir / "enrollment_demo.csv", enrollment_records)
    logger.info("enrollment_demo: %d ảnh (không tính vào metric)", len(enrollment_records))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run(cfg)


if __name__ == "__main__":
    main()

"""Phase 1 — wrapper gọi MaskTheFace (external CLI tool, KHÔNG phải thư viện import được).

Chạy: python -m src.data.preprocessing --config configs/data.yaml
Là 1 DVC stage (`augment_masks` trong dvc.yaml).

Contract input/output (--path X -> output tại X_masked/) đã VERIFY trực tiếp từ
source mask_the_face.py (`args.write_path = args.path + "_masked"`), không phải
suy đoán. Chưa verify: tên file cụ thể trong thư mục *_masked/ — wrapper cố tình
quét toàn bộ ảnh thay vì đoán tên (collect_masked_outputs) nên không phụ thuộc
điều đó. Logic staging/collect/cleanup đã tự test bằng fake runner mô phỏng đúng
contract này (tests/test_preprocessing.py) — nhưng subprocess gọi MaskTheFace
THẬT chưa chạy thử lần nào (sandbox viết code này không có mạng/dlib).

Việc đầu tiên khi có máy thật (Colab, có mạng, đã `git submodule update --init`
third_party/MaskTheFace + `pip install -r third_party/MaskTheFace/requirements.txt`):
    python -m src.data.preprocessing --config configs/data.yaml --dry-run
chỉ in lệnh sẽ chạy, không thực thi — soát command trước khi chạy thật trên toàn
bộ RMFD.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def build_masktheface_command(
    *,
    python_exe: str,
    script_path: Path,
    input_dir: Path,
    mask_type: str,
    write_original_image: bool,
) -> list[str]:
    """Build argv gọi mask_the_face.py.

    Theo đúng CLI argument đã xác nhận từ tài liệu chính thức của MaskTheFace
    (aqeelanwar/MaskTheFace): --path, --mask_type, --verbose, --write_original_image
    (flag, không nhận giá trị). Hành vi output (--path X -> ghi vào X_masked) là
    phần DUY NHẤT của contract có xác nhận trực tiếp từ doc — mọi thứ khác
    (tên file cụ thể trong X_masked) không được assume, xem collect_masked_outputs.
    """
    cmd = [
        python_exe,
        str(script_path),
        "--path",
        str(input_dir),
        "--mask_type",
        mask_type,
        "--verbose",
    ]
    if write_original_image:
        cmd.append("--write_original_image")
    return cmd


def stage_identity_images(
    identity_dir: Path, staging_dir: Path, max_images: int | None = None
) -> int:
    """Symlink ảnh của 1 identity vào staging_dir (không copy byte ảnh).

    `max_images` (bổ sung 2026-09-28, SPEC v2.1 "B7"): giới hạn số ảnh/identity
    đưa vào MaskTheFace — mặc định None = không giới hạn (giữ nguyên hành vi cũ,
    không phá test hiện có). Sort trước khi cắt để deterministic.

    Trả về số ảnh đã stage.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    images = sorted(
        img
        for img in identity_dir.rglob("*")
        if img.is_file() and img.suffix.lower() in IMAGE_EXTENSIONS
    )
    if max_images is not None:
        images = images[:max_images]
    for img in images:
        link = staging_dir / img.name
        if link.exists() or link.is_symlink():
            continue  # trùng tên trong cùng 1 identity — bỏ qua an toàn, không ghi đè
        link.symlink_to(img.resolve())
        count += 1
    return count


def collect_masked_outputs(masked_dir: Path, dest_dir: Path, prefix: str) -> int:
    """Quét TOÀN BỘ ảnh xuất hiện trong masked_dir, copy vào dest_dir với tên có
    prefix (tránh đụng tên giữa các lần chạy khác mask_type). KHÔNG đoán tên file
    cụ thể MaskTheFace sinh ra — xem docstring module.

    Trả về số ảnh đã collect.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    if not masked_dir.exists():
        logger.warning(
            "Thư mục output MaskTheFace không tồn tại (script chạy fail?): %s",
            masked_dir,
        )
        return count
    for img in sorted(masked_dir.rglob("*")):
        if img.is_file() and img.suffix.lower() in IMAGE_EXTENSIONS:
            shutil.copyfile(img, dest_dir / f"{prefix}_{img.name}")
            count += 1
    return count


def process_identity(
    identity_dir: Path,
    dest_dir: Path,
    *,
    tmp_root: Path,
    python_exe: str,
    script_path: Path,
    mtf_cwd: Path,
    mask_types: list[str],
    write_original_image: bool,
    max_images_per_identity: int | None = None,
    runner=subprocess.run,
) -> int:
    """Chạy MaskTheFace cho 1 identity qua từng mask_type, gom output vào
    dest_dir/<identity_id>/. Trả về tổng số ảnh output.

    `mtf_cwd`: thư mục sẽ dùng làm cwd của subprocess (BUG ĐÃ SỬA, 2026-09-28) —
    mask_the_face.py đọc `masks/masks.cfg` và tải `dlib_models/...` theo đường dẫn
    TƯƠNG ĐỐI VỚI CWD của process, không phải tương đối với vị trí script. Trước
    đây wrapper không truyền cwd nên subprocess chạy với cwd = gốc repo của DỰ ÁN
    NÀY (không phải thư mục MaskTheFace) — mọi identity fail 100% (không tìm
    thấy masks.cfg), wrapper chỉ log lỗi rồi continue nên không ai để ý.
    Fix: truyền `cwd=mtf_cwd` (repo_dir của MaskTheFace) cho runner, ĐỒNG THỜI
    `script_path` và `input_dir` (staging_dir) phải là đường dẫn TUYỆT ĐỐI — nếu
    không, đổi cwd sẽ làm sai luôn 2 đường dẫn ta tự đưa vào lệnh.

    `runner` cho phép inject fake subprocess trong test (tests/test_preprocessing.py)
    — không phụ thuộc subprocess/MaskTheFace thật khi test logic orchestration.
    """
    identity_id = identity_dir.name
    script_path_abs = script_path.resolve()
    total = 0
    for mask_type in mask_types:
        staging_dir = tmp_root / f"{identity_id}_{mask_type}_input"
        n_staged = stage_identity_images(
            identity_dir, staging_dir, max_images=max_images_per_identity
        )
        if n_staged == 0:
            shutil.rmtree(staging_dir, ignore_errors=True)
            continue

        cmd = build_masktheface_command(
            python_exe=python_exe,
            script_path=script_path_abs,
            input_dir=staging_dir.resolve(),
            mask_type=mask_type,
            write_original_image=write_original_image,
        )
        result = runner(cmd, capture_output=True, text=True, cwd=str(mtf_cwd))
        if result.returncode != 0:
            logger.error(
                "MaskTheFace fail cho %s (mask_type=%s): %s",
                identity_id,
                mask_type,
                result.stderr,
            )
            shutil.rmtree(staging_dir, ignore_errors=True)
            continue

        masked_dir = staging_dir.parent / f"{staging_dir.name}_masked"
        total += collect_masked_outputs(masked_dir, dest_dir / identity_id, prefix=mask_type)
        shutil.rmtree(staging_dir, ignore_errors=True)
        shutil.rmtree(masked_dir, ignore_errors=True)
    return total


def check_masktheface_available(repo_dir: Path) -> None:
    """Preflight fail-fast (bổ sung 2026-09-28): trước đây không có bước này —
    nếu submodule chưa `git submodule update --init` hoặc `masks/masks.cfg`
    không tồn tại, mọi identity sẽ fail lặp lại (có thể hàng nghìn lần cho toàn
    bộ RMFD) trước khi ai nhận ra nguyên nhân chỉ là thiếu submodule. Raise ngay
    ở đây thay vì để log lỗi trôi qua từng identity một.
    """
    cfg_file = repo_dir / "masks" / "masks.cfg"
    if not repo_dir.is_dir() or not cfg_file.exists():
        raise FileNotFoundError(
            f"Không tìm thấy MaskTheFace tại {repo_dir} (thiếu {cfg_file}). "
            "Chạy: git submodule update --init third_party/MaskTheFace "
            "&& pip install -r third_party/MaskTheFace/requirements.txt"
        )


def run(cfg: dict, *, dry_run: bool = False, runner=subprocess.run) -> None:
    mtf_cfg = cfg["masktheface"]
    repo_dir = Path(mtf_cfg["repo_dir"]).resolve()
    script_path = repo_dir / mtf_cfg["script_name"]
    mask_types = mtf_cfg["mask_types"]
    write_original_image = mtf_cfg["write_original_image"]
    # Giới hạn quy mô (SPEC v2.1 mục "Vấn đề dữ liệu B7"): spec ghi "subset vài
    # nghìn ảnh" nhưng không cap thì code trước đây xử lý TOÀN BỘ ~90K ảnh
    # unmasked x 3 mask_type (~270K job con), mỗi job 1 lần khởi động Python +
    # nạp dlib model riêng — quá nặng cho 1 DVC stage all-or-nothing chạy trên
    # Colab free. None = không cap (dùng khi đã sẵn sàng chạy full).
    max_images_per_identity: int | None = mtf_cfg.get("max_images_per_identity")

    if not dry_run:
        check_masktheface_available(repo_dir)

    # 2 nguồn cần augment (mục 2.1 spec): rmfd_unmasked (bổ sung masked variant)
    # và enrollment (chỉ để demo, KHÔNG vào metric — xem src/data/labels.py)
    paths = cfg["paths"]
    jobs = [
        (Path(paths["rmfd_unmasked_dir"]), Path(paths["masktheface_rmfd_out_dir"])),
        (Path(paths["enrollment_dir"]), Path(paths["masktheface_enrollment_out_dir"])),
    ]

    tmp_root = Path(cfg["paths"].get("tmp_dir", "data/processed/_staging_masktheface"))
    tmp_root.mkdir(parents=True, exist_ok=True)

    n_identity_total = 0
    n_identity_zero_output = 0

    for src_root, dest_root in jobs:
        if not src_root.exists():
            logger.warning("Nguồn không tồn tại, bỏ qua: %s", src_root)
            continue
        for identity_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
            if dry_run:
                for mask_type in mask_types:
                    cmd = build_masktheface_command(
                        python_exe=sys.executable,
                        script_path=script_path,
                        input_dir=tmp_root / f"{identity_dir.name}_{mask_type}_input",
                        mask_type=mask_type,
                        write_original_image=write_original_image,
                    )
                    print(f"(cwd={repo_dir}) " + " ".join(cmd))
                continue
            n_identity_total += 1
            n = process_identity(
                identity_dir,
                dest_root,
                tmp_root=tmp_root,
                python_exe=sys.executable,
                script_path=script_path,
                mtf_cwd=repo_dir,
                mask_types=mask_types,
                write_original_image=write_original_image,
                max_images_per_identity=max_images_per_identity,
                runner=runner,
            )
            if n == 0:
                n_identity_zero_output += 1
            logger.info("%s: %d ảnh masked output", identity_dir.name, n)

    if not dry_run:
        shutil.rmtree(tmp_root, ignore_errors=True)
        if n_identity_total > 0:
            fail_rate = n_identity_zero_output / n_identity_total
            logger.info(
                "Tổng kết: %d/%d identity có 0 ảnh output (%.1f%%)",
                n_identity_zero_output,
                n_identity_total,
                fail_rate * 100,
            )
            min_success_rate = mtf_cfg.get("min_success_rate", 0.5)
            if fail_rate > (1 - min_success_rate):
                raise RuntimeError(
                    f"{fail_rate:.0%} identity augment thất bại (0 ảnh output) — "
                    f"vượt ngưỡng cho phép (min_success_rate={min_success_rate}). "
                    "Kiểm tra MaskTheFace/dlib model trước khi chạy lại toàn bộ "
                    "(xem check_masktheface_available)."
                )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ in lệnh sẽ chạy, không thực thi — soát command trước khi có MaskTheFace thật",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

"""Unit test cho src/data/preprocessing.py.

Không có mạng/dlib/MaskTheFace thật trong CI — test bằng fake `runner` mô phỏng
đúng contract I/O mà tài liệu chính thức MaskTheFace xác nhận (--path X ghi output
vào X_masked/). KHÔNG test MaskTheFace thật, chỉ test logic orchestration của
wrapper (staging/collect/cleanup/cwd) trong file này.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from src.data.preprocessing import (
    build_masktheface_command,
    check_masktheface_available,
    collect_masked_outputs,
    process_identity,
    stage_identity_images,
)


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00")


def _fake_runner_ok(cmd, capture_output=True, text=True, cwd=None):
    input_dir = Path(cmd[cmd.index("--path") + 1])
    masked_dir = input_dir.parent / f"{input_dir.name}_masked"
    masked_dir.mkdir(parents=True, exist_ok=True)
    for img in input_dir.iterdir():
        if img.is_file():
            shutil.copyfile(img, masked_dir / img.name)
    return subprocess.CompletedProcess(cmd, returncode=0, stdout="ok", stderr="")


def _fake_runner_fail(cmd, capture_output=True, text=True, cwd=None):
    return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="dlib model not found")


def test_build_masktheface_command_shape():
    cmd = build_masktheface_command(
        python_exe="python3",
        script_path=Path("third_party/MaskTheFace/mask_the_face.py"),
        input_dir=Path("/tmp/foo"),
        mask_type="surgical",
        write_original_image=False,
    )
    assert cmd == [
        "python3",
        "third_party/MaskTheFace/mask_the_face.py",
        "--path",
        "/tmp/foo",
        "--mask_type",
        "surgical",
        "--verbose",
    ]


def test_build_masktheface_command_write_original_image_flag():
    cmd = build_masktheface_command(
        python_exe="python3",
        script_path=Path("x.py"),
        input_dir=Path("/tmp/foo"),
        mask_type="N95",
        write_original_image=True,
    )
    assert cmd[-1] == "--write_original_image"


def test_stage_identity_images_symlinks_only_images(tmp_path):
    identity_dir = tmp_path / "raw" / "id001"
    _touch(identity_dir / "a.jpg")
    _touch(identity_dir / "b.png")
    _touch(identity_dir / "readme.txt")

    staging = tmp_path / "staging"
    n = stage_identity_images(identity_dir, staging)

    assert n == 2
    assert (staging / "a.jpg").is_symlink()
    assert not (staging / "readme.txt").exists()


def test_stage_identity_images_respects_max_images_cap(tmp_path):
    """Bổ sung 2026-09-28 (SPEC v2.1 B7) — max_images=None (mặc định) giữ đúng
    hành vi cũ (không cap, test ở trên), max_images=N chỉ lấy N ảnh đầu theo
    thứ tự sort (deterministic, không phụ thuộc thứ tự filesystem)."""
    identity_dir = tmp_path / "raw" / "id001"
    for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg"):
        _touch(identity_dir / name)

    staging = tmp_path / "staging"
    n = stage_identity_images(identity_dir, staging, max_images=2)

    assert n == 2
    assert sorted(p.name for p in staging.iterdir()) == ["a.jpg", "b.jpg"]


def test_collect_masked_outputs_scans_nested_without_guessing_names(tmp_path):
    masked_dir = tmp_path / "masked_out"
    _touch(masked_dir / "whatever_weird_name_123.jpg")
    _touch(masked_dir / "nested" / "sub" / "another.png")

    dest = tmp_path / "dest"
    n = collect_masked_outputs(masked_dir, dest, prefix="surgical")

    assert n == 2
    assert (dest / "surgical_whatever_weird_name_123.jpg").exists()
    assert (dest / "surgical_another.png").exists()


def test_process_identity_full_orchestration_with_ok_runner(tmp_path):
    identity_dir = tmp_path / "raw" / "id002"
    _touch(identity_dir / "x.jpg")
    _touch(identity_dir / "y.jpg")
    dest_root = tmp_path / "dest"
    tmp_root = tmp_path / "tmproot"
    mtf_cwd = tmp_path / "third_party" / "MaskTheFace"
    mtf_cwd.mkdir(parents=True)

    total = process_identity(
        identity_dir,
        dest_root,
        tmp_root=tmp_root,
        python_exe="python3",
        script_path=mtf_cwd / "mask_the_face.py",
        mtf_cwd=mtf_cwd,
        mask_types=["surgical", "N95"],
        write_original_image=False,
        runner=_fake_runner_ok,
    )

    assert total == 4  # 2 ảnh gốc x 2 mask_type
    out_files = sorted(p.name for p in (dest_root / "id002").iterdir())
    assert out_files == ["N95_x.jpg", "N95_y.jpg", "surgical_x.jpg", "surgical_y.jpg"]
    # staging + masked_dir tạm phải được dọn sạch sau khi xong
    assert not (tmp_root / "id002_surgical_input").exists()
    assert not (tmp_root / "id002_surgical_input_masked").exists()


def test_process_identity_passes_cwd_and_absolute_paths_to_runner(tmp_path):
    """BUG ĐÃ SỬA (2026-09-28): trước đây runner được gọi KHÔNG có cwd, nên
    mask_the_face.py tìm masks/masks.cfg tương đối với cwd gốc của TIẾN TRÌNH
    GỌI (repo dự án này), không phải thư mục MaskTheFace -> fail 100%. Test này
    khoá lại 2 điều kiện của fix: (1) runner nhận đúng cwd=mtf_cwd; (2)
    script_path/input_dir trong cmd là tuyệt đối (đổi cwd không được làm sai
    2 đường dẫn này)."""
    identity_dir = tmp_path / "raw" / "id010"
    _touch(identity_dir / "p.jpg")
    mtf_cwd = tmp_path / "third_party" / "MaskTheFace"
    mtf_cwd.mkdir(parents=True)

    captured: dict = {}

    def _spy_runner(cmd, capture_output=True, text=True, cwd=None):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    process_identity(
        identity_dir,
        tmp_path / "dest",
        tmp_root=tmp_path / "tmproot",
        python_exe="python3",
        script_path=mtf_cwd / "mask_the_face.py",
        mtf_cwd=mtf_cwd,
        mask_types=["surgical"],
        write_original_image=False,
        runner=_spy_runner,
    )

    assert captured["cwd"] == str(mtf_cwd)
    script_arg, input_arg = captured["cmd"][1], captured["cmd"][3]
    assert Path(script_arg).is_absolute()
    assert Path(input_arg).is_absolute()


def test_process_identity_does_not_crash_when_runner_fails(tmp_path):
    identity_dir = tmp_path / "raw" / "id003"
    _touch(identity_dir / "z.jpg")
    mtf_cwd = tmp_path / "third_party" / "MaskTheFace"
    mtf_cwd.mkdir(parents=True)

    total = process_identity(
        identity_dir,
        tmp_path / "dest",
        tmp_root=tmp_path / "tmproot",
        python_exe="python3",
        script_path=mtf_cwd / "x.py",
        mtf_cwd=mtf_cwd,
        mask_types=["surgical"],
        write_original_image=False,
        runner=_fake_runner_fail,
    )

    assert total == 0


def test_check_masktheface_available_raises_when_masks_cfg_missing(tmp_path):
    repo_dir = tmp_path / "third_party" / "MaskTheFace"
    repo_dir.mkdir(parents=True)  # tồn tại thư mục nhưng KHÔNG có masks/masks.cfg

    with pytest.raises(FileNotFoundError, match="masks.cfg"):
        check_masktheface_available(repo_dir)


def test_check_masktheface_available_passes_when_masks_cfg_present(tmp_path):
    repo_dir = tmp_path / "third_party" / "MaskTheFace"
    _touch(repo_dir / "masks" / "masks.cfg")

    check_masktheface_available(repo_dir)  # không raise

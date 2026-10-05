"""Unit test cho src/training/rng_state.py.

Test chạy đúng bất kể môi trường CÓ hay KHÔNG có torch cài (kiểm tra động, không
hard-code giả định) — phần torch chỉ được assert khi thực sự import được.
"""

import random

from src.training.rng_state import capture_all_rng, restore_all_rng

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def test_capture_includes_available_backends_only():
    snapshot = capture_all_rng()
    assert "python" in snapshot
    assert ("numpy" in snapshot) == HAS_NUMPY
    assert ("torch" in snapshot) == HAS_TORCH


def test_restore_resumes_exact_rng_position_not_just_reseed():
    """Đây là thuộc tính QUAN TRỌNG NHẤT: restore phải cho ra đúng chuỗi số ngẫu
    nhiên TIẾP THEO tại vị trí capture, không phải reset về đầu chuỗi (seed lại).

    Nhánh torch ĐÃ VERIFY chạy thật (2026-09-29, torch 2.14 — trước đó
    capture_torch_random/restore_torch_random chỉ được flag "CHƯA VERIFY" vì
    sandbox viết code không có torch cài, xem docstring module)."""
    random.seed(999)
    if HAS_NUMPY:
        np.random.seed(999)
    if HAS_TORCH:
        torch.manual_seed(999)

    snapshot = capture_all_rng()

    seq_a_python = [random.random() for _ in range(5)]
    seq_a_numpy = list(np.random.rand(5)) if HAS_NUMPY else []
    seq_a_torch = torch.rand(5).tolist() if HAS_TORCH else []

    restore_all_rng(snapshot)

    seq_b_python = [random.random() for _ in range(5)]
    seq_b_numpy = list(np.random.rand(5)) if HAS_NUMPY else []
    seq_b_torch = torch.rand(5).tolist() if HAS_TORCH else []

    assert seq_a_python == seq_b_python
    if HAS_NUMPY:
        assert seq_a_numpy == seq_b_numpy
    if HAS_TORCH:
        assert seq_a_torch == seq_b_torch


def test_restore_after_further_draws_still_recovers_original_position():
    """Capture -> draw thêm nhiều lần (mô phỏng train tiếp sau lúc checkpoint) ->
    restore về đúng lúc capture -> phải ra lại đúng seq_a, không lẫn ảnh hưởng của
    những lần draw ở giữa.
    """
    random.seed(42)
    if HAS_TORCH:
        torch.manual_seed(42)
    snapshot = capture_all_rng()
    seq_a = [random.random() for _ in range(3)]
    seq_a_torch = torch.rand(3).tolist() if HAS_TORCH else []

    # mô phỏng train tiếp — draw thêm, KHÔNG liên quan gì đến snapshot ở trên
    _ = [random.random() for _ in range(50)]
    if HAS_TORCH:
        _ = torch.rand(50)

    restore_all_rng(snapshot)
    seq_b = [random.random() for _ in range(3)]
    seq_b_torch = torch.rand(3).tolist() if HAS_TORCH else []

    assert seq_a == seq_b
    if HAS_TORCH:
        assert seq_a_torch == seq_b_torch

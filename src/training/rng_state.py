"""Capture/restore RNG state — mục 2.2 spec + Module 10.3 (reproducibility).

Chỉ set lại SEED ban đầu khi resume KHÔNG đủ — RNG đã tiến qua hàng nghìn lần
gọi kể từ lúc bắt đầu train, set lại seed sẽ lặp lại đúng chuỗi ngẫu nhiên TỪ ĐẦU,
không phải TỪ VỊ TRÍ NGẮT QUÃNG. Phải lưu/khôi phục state object thật của từng
RNG (python `random`, `numpy.random`, `torch`).

Phần torch degrade graceful khi không có torch cài (import trễ trong hàm, bắt
ImportError) — nhờ vậy capture_all_rng/restore_all_rng test được ĐẦY ĐỦ trong môi
trường không có torch (xem tests/unit/test_rng_state.py, chạy thật, không phải
mock). capture_torch_random/restore_torch_random ĐÃ VERIFY chạy thật (2026-09-29,
torch 2.14 — round-trip giữ đúng vị trí trong chuỗi random, xem test cùng file).
"""

from __future__ import annotations

import random
from typing import Any


def capture_python_random() -> Any:
    return random.getstate()


def restore_python_random(state: Any) -> None:
    random.setstate(state)


def capture_numpy_random() -> Any:
    import numpy as np

    return np.random.get_state()


def restore_numpy_random(state: Any) -> None:
    import numpy as np

    np.random.set_state(state)


def capture_torch_random() -> dict:
    """Đã verify chạy thật (2026-09-29, torch 2.14) — xem test_rng_state.py."""
    import torch

    state = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_torch_random(state: dict) -> None:
    """Đã verify chạy thật (2026-09-29, torch 2.14) — xem test_rng_state.py."""
    import torch

    torch.set_rng_state(state["cpu"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def capture_all_rng() -> dict:
    """Capture tất cả RNG có sẵn trong môi trường hiện tại. Không có torch/numpy
    thì bỏ qua phần đó thay vì raise — checkpoint vẫn ghi được, chỉ resume kém
    chính xác hơn (chấp nhận được, còn hơn crash toàn bộ training).
    """
    state: dict[str, Any] = {"python": capture_python_random()}
    try:
        state["numpy"] = capture_numpy_random()
    except ImportError:
        pass
    try:
        state["torch"] = capture_torch_random()
    except ImportError:
        pass
    return state


def restore_all_rng(state: dict) -> None:
    if "python" in state:
        restore_python_random(state["python"])
    if "numpy" in state:
        try:
            restore_numpy_random(state["numpy"])
        except ImportError:
            pass
    if "torch" in state:
        try:
            restore_torch_random(state["torch"])
        except ImportError:
            pass

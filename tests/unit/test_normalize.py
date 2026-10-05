"""Unit test cho src/data/normalize.py — khoá hợp đồng preprocessing train <-> browser."""

from pathlib import Path

import numpy as np
import pytest

from src.data.normalize import bgr_to_rgb, load_contract, normalize_rgb_uint8

EXPORT_YAML = Path(__file__).resolve().parents[2] / "configs" / "export.yaml"


@pytest.fixture
def contract():
    return load_contract(EXPORT_YAML)


def test_shipped_contract_matches_official_arcface_torch(contract):
    assert contract.input_size == 112
    assert (contract.channel_order, contract.layout) == ("RGB", "CHW")
    assert (contract.scale, contract.mean, contract.std) == (255.0, 0.5, 0.5)


def test_normalize_endpoints_and_midpoint(contract):
    black = np.zeros((112, 112, 3), dtype=np.uint8)
    white = np.full((112, 112, 3), 255, dtype=np.uint8)

    assert normalize_rgb_uint8(black, contract).min() == -1.0
    assert normalize_rgb_uint8(white, contract).max() == 1.0
    mid = np.full((112, 112, 3), 128, dtype=np.uint8)
    assert abs(float(normalize_rgb_uint8(mid, contract)[0, 0, 0]) - (128 / 255 - 0.5) / 0.5) < 1e-6


def test_output_is_float32_chw_and_channels_land_in_right_plane(contract):
    img = np.zeros((112, 112, 3), dtype=np.uint8)
    img[..., 0], img[..., 1], img[..., 2] = 255, 0, 51  # R, G, B khác nhau

    out = normalize_rgb_uint8(img, contract)

    assert out.dtype == np.float32 and out.shape == (3, 112, 112)
    assert out[0, 0, 0] == 1.0  # plane 0 = R
    assert out[1, 0, 0] == -1.0  # plane 1 = G
    assert abs(float(out[2, 0, 0]) - (51 / 255 - 0.5) / 0.5) < 1e-6  # plane 2 = B


def test_matches_reference_formula_of_official_inference_script(contract):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(112, 112, 3), dtype=np.uint8)

    # Công thức y hệt inference.py: transpose CHW -> float -> div(255) -> sub(0.5) -> div(0.5)
    reference = (np.transpose(img, (2, 0, 1)).astype(np.float32) / 255 - 0.5) / 0.5

    np.testing.assert_allclose(normalize_rgb_uint8(img, contract), reference, rtol=0, atol=1e-6)


def test_bgr_to_rgb_swaps_channels(contract):
    bgr = np.zeros((112, 112, 3), dtype=np.uint8)
    bgr[..., 0] = 10  # B
    bgr[..., 2] = 200  # R

    rgb = bgr_to_rgb(bgr)

    assert rgb[0, 0, 0] == 200 and rgb[0, 0, 2] == 10


def test_wrong_dtype_or_shape_raises_instead_of_resizing_silently(contract):
    with pytest.raises(ValueError):
        normalize_rgb_uint8(np.zeros((112, 112, 3), dtype=np.float32), contract)
    with pytest.raises(ValueError):
        normalize_rgb_uint8(np.zeros((160, 160, 3), dtype=np.uint8), contract)
    with pytest.raises(ValueError):
        normalize_rgb_uint8(np.zeros((112, 112), dtype=np.uint8), contract)

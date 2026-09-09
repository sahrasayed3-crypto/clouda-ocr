"""Distortion engines: registry, ported B ops, composite backend, QC gate.

Migrated from the standalone repository (tests/test_distort.py); imports
retargeted to ``clouda_data.factory``.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from clouda_data.factory.distort import atomic as atomic_engine
from clouda_data.factory.distort import scan_composite
from clouda_data.factory.distort.qc import assess

cv2 = pytest.importorskip("cv2")

PAGE = np.full((120, 90, 3), 250, np.uint8)
PAGE[40:80, 20:70] = 30  # an "ink" block


def _rng(seed):
    return np.random.default_rng(seed)


def test_registry_contains_20_ops():
    ops = atomic_engine.available()
    assert len(ops) == 20
    for required in (
        "vertical_streaks",
        "ink_density",
        "photocopy_generation",
        "bleedthrough",
        "motion_blur",
        "perspective",
    ):
        assert required in ops


def test_atomic_op_is_deterministic():
    a1 = atomic_engine.apply_distortion(
        "gaussian_noise", PAGE.copy(), {"sigma": 5.0}, _rng(7)
    )
    a2 = atomic_engine.apply_distortion(
        "gaussian_noise", PAGE.copy(), {"sigma": 5.0}, _rng(7)
    )
    assert np.array_equal(a1, a2)


def test_atomic_op_changes_pixels():
    out = atomic_engine.apply_distortion(
        "gaussian_noise", PAGE.copy(), {"sigma": 20.0}, _rng(3)
    )
    assert not np.array_equal(out, PAGE)


def test_unknown_op_raises():
    with pytest.raises(atomic_engine.DistortionError):
        atomic_engine.apply_distortion("nope", PAGE.copy(), {}, _rng(1))


def test_vertical_streaks_matches_b_math():
    """Ported op must reproduce the original B loop for the same seed."""
    damage = 0.4
    rng = np.random.default_rng(99)
    expected = PAGE.astype(np.float32)
    for _ in range(max(1, int(damage * 8))):
        xpos = int(rng.integers(0, expected.shape[1]))
        expected[:, max(0, xpos - 1) : xpos + 1] -= np.float32(
            rng.uniform(2, 10) * damage
        )
    expected = np.clip(expected, 0, 255).astype(np.uint8)
    got = atomic_engine.apply_distortion(
        "vertical_streaks",
        PAGE.copy(),
        {"count": max(1, int(damage * 8)), "strength": damage},
        np.random.default_rng(99),
    )
    assert np.array_equal(got, expected)


def test_ink_density_matches_b_math():
    rng = np.random.default_rng(5)
    factor = np.float32(rng.uniform(1 - 0.06 * 0.5, 1 + 0.03 * 0.5))
    expected = np.clip(PAGE.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    got = atomic_engine.apply_distortion(
        "ink_density",
        PAGE.copy(),
        {"low_gain": 1 - 0.06 * 0.5, "high_gain": 1 + 0.03 * 0.5},
        np.random.default_rng(5),
    )
    assert np.array_equal(got, expected)


def test_photocopy_generation_matches_b_math():
    from PIL import ImageEnhance, ImageFilter

    img = Image.fromarray(PAGE)
    expected = img
    for _ in range(2):
        expected = (
            ImageEnhance.Contrast(expected.convert("L"))
            .enhance(1.08 + 0.5 * 0.2)
            .filter(ImageFilter.UnsharpMask(1, 80, 2))
            .convert("RGB")
        )
    got = atomic_engine.apply_distortion(
        "photocopy_generation",
        PAGE.copy(),
        {"generations": 2, "contrast": 1.08 + 0.5 * 0.2, "unsharp": 80},
        _rng(1),
    )
    assert np.array_equal(got, np.asarray(expected))


def test_scan_composite_deterministic():
    params = {
        "damage": 0.34,
        "paper": "aged",
        "photocopy_generation": 0,
        "color": "rgb",
    }
    img = Image.fromarray(PAGE)
    a = np.asarray(scan_composite.degrade(img, params, 12345))
    b = np.asarray(scan_composite.degrade(img, params, 12345))
    assert np.array_equal(a, b)
    c = np.asarray(scan_composite.degrade(img, params, 12346))
    assert not np.array_equal(a, c)


def test_bleedthrough_with_verso_context():
    verso = np.full_like(PAGE, 200)
    verso[50:60, 30:60] = 10
    out = atomic_engine.apply_distortion(
        "bleedthrough",
        PAGE.copy(),
        {"alpha": 0.05, "blur_sigma": 2.0, "offset_frac": [0.0, 0.0]},
        _rng(1),
        {"verso": verso},
    )
    assert out.shape == PAGE.shape


def test_qc_gate_passes_light_damage_and_flags_flooding():
    clean = PAGE
    light = atomic_engine.apply_distortion(
        "gaussian_noise",
        clean.copy(),
        {"sigma": 2.0},
        _rng(2),
    )
    result = assess(
        light,
        clean,
        min_contrast_ratio=0.1,
        min_sharpness_ratio=0.001,
        min_ink_ratio=0.15,
        max_ink_ratio=4.0,
    )
    assert result.passed
    flooded = np.full_like(clean, 40)  # everything dark -> ink ratio explodes
    result2 = assess(
        flooded,
        clean,
        min_contrast_ratio=0.1,
        min_sharpness_ratio=0.001,
        min_ink_ratio=0.15,
        max_ink_ratio=4.0,
    )
    assert not result2.passed

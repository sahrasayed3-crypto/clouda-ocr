"""System B scan-family profiles as a first-class Clouda resource.

Field-for-field conversion of the vendored
``legacy/arabic_scan_factory/arabic_scan_factory/profiles.py`` PROFILES dict.
The standalone data factory loaded these through an ``importlib`` file-location
hack against its vendored legacy tree; in the integrated repository the
parameters live here directly (the frozen legacy tree is not part of the
canonical package). All 12 profile ids, dpi/damage/paper/photocopy/color/
jpeg-quality/family/recoverability values, and the ``variant_seed`` derivation
contract are preserved exactly, so historical scan-family runs remain
reproducible through ``clouda_data.factory.seed.legacy``.
"""

from __future__ import annotations

import hashlib

_NAMES = [
    "high_quality_flatbed",
    "clean_book_scan",
    "normal_office_scan",
    "old_book_light",
    "old_book_medium",
    "old_book_heavy",
    "old_photocopy",
    "recopied_photocopy",
    "archive_scan",
    "low_dpi_scan",
    "aged_but_readable",
    "hard_composite",
]
_DPI = [400, 300, 300, 300, 300, 300, 200, 200, 600, 150, 300, 200]
_LEVEL = [0.06, 0.10, 0.18, 0.20, 0.34, 0.48, 0.34, 0.52, 0.16, 0.38, 0.26, 0.62]
_FAMILIES = [
    "flatbed",
    "book_scan",
    "office_scanner",
    "old_book",
    "old_book",
    "old_book",
    "photocopy",
    "photocopy",
    "archive",
    "low_dpi_scanner",
    "aged_book",
    "composite",
]
_SEVERE = {"06_old_book_heavy", "08_recopied_photocopy", "12_hard_composite"}
_QUALITY = [94, 92, 88, 90, 84, 76, 75, 62, 91, 65, 82, 58]
_EFFECTS = [
    "illumination_falloff",
    "sensor_noise",
    "bleed_through_proxy",
    "ink_density",
    "vertical_streaks",
    "paper_tint",
    "foxing",
    "gutter_shadow",
    "edge_shadow",
    "scanner_blur",
    "skew",
    "photocopy_contrast",
    "resolution_resampling",
    "jpeg_encoding",
]

PROFILES: dict[str, dict] = {}
for _index, _name in enumerate(_NAMES):
    _key = f"{_index + 1:02d}_{_name}"
    PROFILES[_key] = {
        "dpi": _DPI[_index],
        "damage": _LEVEL[_index],
        "paper": "white" if _index < 3 else "aged",
        "photocopy_generation": (
            1 if _index == 6 else (2 if _index == 7 else (3 if _index == 11 else 0))
        ),
        "color": "rgb" if _index in (1, 3, 4, 5, 8, 10, 11) else "grayscale",
        "jpeg_quality": _QUALITY[_index],
        "family": _FAMILIES[_index],
        "recoverability": "SEVERE" if _key in _SEVERE else "RECOVERABLE",
        "effects": list(_EFFECTS),
    }


def variant_seed(
    source_sha256: str, variant_index: int, profile: str, base_seed: int = 0
) -> int:
    payload = f"{base_seed}:{source_sha256}:variant:{variant_index}:{profile}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def select_profiles(variants: int, profile: str | None = None) -> list[str]:
    """Return a recoverable-first deterministic profile sequence."""
    if variants < 1:
        raise ValueError("variants must be positive")
    if profile is not None:
        if profile not in PROFILES:
            raise ValueError(f"unknown profile: {profile}")
        return [profile] * variants
    names = list(PROFILES)
    return [names[index % len(names)] for index in range(variants)]

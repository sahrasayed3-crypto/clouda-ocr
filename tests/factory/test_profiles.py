"""Profile schema + lossless legacy adapters (migrated from the standalone repo)."""

from __future__ import annotations

import pytest

from clouda_data.factory.profiles import load_profile_book, load_unified_yaml
from clouda_data.factory.profiles.schema import ProfileError
from clouda_data.factory.profiles import scan_families

EXPECTED_A_PROFILES = {
    "old_book_light",
    "old_book_medium",
    "bad_scan_medium",
    "bad_scan_heavy",
    "phone_photo_light",
    "phone_photo_medium",
    "faded_archive",
    "compressed_scan",
}
EXPECTED_B_PROFILES = {
    "01_high_quality_flatbed",
    "02_clean_book_scan",
    "03_normal_office_scan",
    "04_old_book_light",
    "05_old_book_medium",
    "06_old_book_heavy",
    "07_old_photocopy",
    "08_recopied_photocopy",
    "09_archive_scan",
    "10_low_dpi_scan",
    "11_aged_but_readable",
    "12_hard_composite",
}


@pytest.fixture(scope="module")
def book():
    return load_profile_book()


def test_all_20_legacy_profiles_load(book):
    assert set(book.profiles) >= EXPECTED_A_PROFILES | EXPECTED_B_PROFILES
    assert len(book.profiles) == 20


def test_all_17_distortion_specs_load(book):
    assert len(book.distortions) == 17
    for required in (
        "gaussian_blur",
        "motion_blur",
        "bleedthrough",
        "perspective",
        "paper_degradation",
        "stains",
        "salt_pepper_noise",
        "skew",
    ):
        assert required in book.distortions


def test_scan_factory_fields_are_lossless(book):
    p = book.profile("05_old_book_medium")
    assert (
        p.dpi_target,
        p.damage,
        p.color,
        p.jpeg_quality,
        p.photocopy_generations,
        p.recoverability,
    ) == (300, 0.34, "rgb", 84, 0, "RECOVERABLE")
    severe = book.profile("06_old_book_heavy")
    assert severe.recoverability == "SEVERE"
    composite = book.profile("12_hard_composite")
    assert composite.composite is True and composite.photocopy_generations == 3


def test_benchmark_legacy_profiles_have_steps(book):
    p = book.profile("old_book_medium")
    assert p.steps, "A profile must expand to ordered atomic steps"


def test_unknown_profile_raises(book):
    with pytest.raises(ProfileError):
        book.profile("does_not_exist")


def test_unknown_distortion_raises(book):
    with pytest.raises(ProfileError):
        book.spec("does_not_exist")


def test_config_paths_resolve_inside_repository():
    """The integrated package must load its configs from the canonical tree."""
    from clouda_data.factory.profiles import _CONFIGS

    assert (_CONFIGS / "ocr_benchmark.yaml").is_file()
    assert (_CONFIGS / "render_layout.yaml").is_file()
    assert "configs" + chr(92) + "data_factory" in str(
        _CONFIGS
    ) or "configs/data_factory" in str(_CONFIGS)


def test_scan_families_parameters_match_legacy_semantics():
    params = scan_families.PROFILES["05_old_book_medium"]
    assert params == {
        "dpi": 300,
        "damage": 0.34,
        "paper": "aged",
        "photocopy_generation": 0,
        "color": "rgb",
        "jpeg_quality": 84,
        "family": "old_book",
        "recoverability": "RECOVERABLE",
        "effects": scan_families._EFFECTS,
    }


def test_load_unified_yaml_roundtrip(tmp_path):
    unified = tmp_path / "unified.yaml"
    unified.write_text(
        """
profiles:
  custom_profile:
    schema: unified_v1
    steps:
      - {distortion: gaussian_blur, severity: light}
    dpi_target: 200
    color: rgb
    jpeg_quality: 85
""",
        encoding="utf-8",
    )
    profiles = load_unified_yaml(unified)
    assert "custom_profile" in profiles
    assert profiles["custom_profile"].dpi_target == 200

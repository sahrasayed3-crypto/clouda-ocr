from __future__ import annotations

from .base import Distortion, DistortionSpec, MetadataOnlyDistortion
from .operators import RealImageDistortion


class DistortionRegistry:
    def __init__(self) -> None:
        self._classes: dict[str, type[Distortion]] = {}

    def register(self, name: str, cls: type[Distortion]) -> None:
        if name in self._classes:
            raise ValueError(f"Distortion already registered: {name}")
        self._classes[name] = cls

    def create(self, spec: DistortionSpec) -> Distortion:
        cls = self._classes.get(spec.name)
        if cls is None:
            raise KeyError(f"Unknown distortion: {spec.name}")
        return cls(spec)

    def names(self) -> list[str]:
        return sorted(self._classes)


DEFAULT_DISTORTIONS = [
    "small_rotation",
    "dpi_reduction",
    "downscale_upscale",
    "blur",
    "gaussian_blur",
    "defocus_blur",
    "motion_blur",
    "anisotropic_blur",
    "edge_blur",
    "local_blur",
    "text_region_blur",
    "gaussian_noise",
    "poisson_noise",
    "salt_pepper_noise",
    "speckle_noise",
    "scanner_noise",
    "row_noise",
    "column_noise",
    "banding",
    "streaks",
    "jpeg_compression",
    "repeated_jpeg_compression",
    "webp_compression",
    "ringing_artifacts",
    "contrast_loss",
    "low_contrast",
    "brightness_variation",
    "faded_text",
    "uneven_fading",
    "uneven_illumination",
    "gradient_illumination",
    "vignetting",
    "glare",
    "local_overexposure",
    "local_underexposure",
    "yellow_paper",
    "browning",
    "gray_paper",
    "paper_texture",
    "paper_fiber",
    "old_paper_texture",
    "stains",
    "water_marks",
    "foxing",
    "dust",
    "scratches",
    "ink_bleed",
    "ink_spread",
    "ink_erosion",
    "broken_strokes",
    "show_through",
    "bleed_through",
    "shadow",
    "binding_shadow",
    "corner_shadow",
    "edge_shadow",
    "edge_darkening",
    "page_curvature",
    "warping",
    "stretching",
    "non_uniform_scaling",
    "perspective",
    "rotation",
    "skew",
    "cropping",
    "padding",
    "translation",
    "edge_clipping",
    "page_misalignment",
    "book_gutter_deformation",
    "border_artifacts",
    "page_border",
    "black_edge",
    "white_edge",
    "torn_edges",
    "fold_marks",
    "creases",
    "wrinkles",
    "dot_gain",
    "toner_scatter",
    "photocopy_degradation",
    "repeated_photocopy",
    "thresholding_artifacts",
    "halftone_pattern",
    "small_font_degradation",
    "broken_characters",
    "weak_punctuation",
    "weak_dots",
    "missing_dots",
    "faint_diacritics",
    "diacritic_erosion",
    "character_erosion",
    "character_dilation",
    "disconnected_strokes",
    "merged_strokes",
    "baseline_jitter",
    "local_text_fading",
    "footnote_degradation",
    "margin_note_degradation",
    "partial_clipping",
    "double_page_scan",
    "scanner_lines",
    "punched_holes",
    "synthetic_stamp",
    "synthetic_handwritten_marks",
    "synthetic_page_number",
    "synthetic_marginal_noise",
]


def default_registry() -> DistortionRegistry:
    registry = DistortionRegistry()
    for name in DEFAULT_DISTORTIONS:
        registry.register(name, RealImageDistortion)
    registry.register("metadata_only", MetadataOnlyDistortion)
    return registry

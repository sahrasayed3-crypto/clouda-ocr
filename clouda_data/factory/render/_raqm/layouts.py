"""Deterministic per-page layout selection.

Each page draws one value from every option pool in configs/layout.yaml using
an RNG seeded from (global_seed, page_index). Re-running with the same seed
therefore reproduces the identical sequence of layouts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from .config_shim import LayoutConfigView, weighted_choice, weighted_choice_entry
from ....provenance.hashing import derive_seed

# Western digits -> Arabic-Indic digits, for synthesized page numbers.
_ARABIC_INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def to_numerals(value: int, system: str) -> str:
    """Format an integer in the requested digit shapes."""
    text = str(int(value))
    if system == "arabic_indic":
        return text.translate(_ARABIC_INDIC)
    if system == "western":
        return text
    raise ValueError(f"unknown numeral system {system!r}")


def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(float(mm) / 25.4 * dpi))


def pt_to_px(pt: float, dpi: int) -> int:
    """Typographic points (1/72 inch) to device pixels."""
    return max(1, int(round(float(pt) / 72.0 * dpi)))


@dataclass(frozen=True)
class LayoutSpec:
    """Fully resolved layout parameters for one page."""

    columns: int
    body_pt: float
    line_spacing: float
    margins_mm: tuple[float, float, float, float]  # top, right, bottom, left
    margins_name: str
    density: str
    fill: float
    indent_em: float
    para_gap_em: float
    justify: bool
    heading_scale: float
    heading_name: str
    running_header: bool
    running_footer: bool
    page_number: bool
    page_number_position: str
    numeral_system: str
    font_family: str
    ink: int
    header_rule: bool
    allow_footnote: bool

    @property
    def name(self) -> str:
        """Compact human-readable layout identifier used in the manifest."""
        return (
            f"{self.columns}col-{self.density}-{self.margins_name}-"
            f"{self.body_pt:g}pt-ls{self.line_spacing:g}-{self.font_family}"
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pool(layout_cfg: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    variation = layout_cfg.get("variation", {})
    if key not in variation:
        raise KeyError(f"layout.yaml: variation pool '{key}' is missing")
    return list(variation[key])


def pick_layout(
    config: LayoutConfigView, page_index: int, global_seed: int
) -> LayoutSpec:
    """Choose a deterministic layout for the given page index."""
    layout_cfg = config.layout
    seed = derive_seed(global_seed, f"layout:{page_index}", distortion="layout")
    rng = np.random.default_rng(seed)

    margins_entry = weighted_choice_entry(_pool(layout_cfg, "margins_mm"), rng)
    density_entry = weighted_choice_entry(_pool(layout_cfg, "density"), rng)
    heading_entry = weighted_choice_entry(_pool(layout_cfg, "heading_scale"), rng)
    font_entry = weighted_choice_entry(layout_cfg["fonts"]["pool"], rng)

    ink_lo, ink_hi = layout_cfg.get("typography", {}).get("ink_jitter", [0, 0])
    base_ink = int(layout_cfg["page"].get("ink", 20))

    variation = layout_cfg["variation"]
    footnote_p = float(variation.get("footnote_probability", 0.0))
    rule_p = float(variation.get("header_rule_probability", 0.0))

    return LayoutSpec(
        columns=int(weighted_choice(_pool(layout_cfg, "columns"), rng)),
        body_pt=float(weighted_choice(_pool(layout_cfg, "body_pt"), rng)),
        line_spacing=float(weighted_choice(_pool(layout_cfg, "line_spacing"), rng)),
        margins_mm=tuple(float(v) for v in margins_entry["value"]),  # type: ignore[arg-type]
        margins_name=str(margins_entry.get("name", "custom")),
        density=str(density_entry["value"]),
        fill=float(density_entry.get("fill", 1.0)),
        indent_em=float(weighted_choice(_pool(layout_cfg, "paragraph_indent_em"), rng)),
        para_gap_em=float(weighted_choice(_pool(layout_cfg, "paragraph_gap_em"), rng)),
        justify=bool(weighted_choice(_pool(layout_cfg, "justify"), rng)),
        heading_scale=float(heading_entry["value"]),
        heading_name=str(heading_entry.get("name", "custom")),
        running_header=bool(weighted_choice(_pool(layout_cfg, "running_header"), rng)),
        running_footer=bool(weighted_choice(_pool(layout_cfg, "running_footer"), rng)),
        page_number=bool(weighted_choice(_pool(layout_cfg, "page_number"), rng)),
        page_number_position=str(
            weighted_choice(_pool(layout_cfg, "page_number_position"), rng)
        ),
        numeral_system=str(weighted_choice(_pool(layout_cfg, "numeral_system"), rng)),
        font_family=str(font_entry["family"]),
        ink=int(np.clip(base_ink + rng.integers(int(ink_lo), int(ink_hi) + 1), 0, 90)),
        header_rule=bool(rng.random() < rule_p),
        allow_footnote=bool(rng.random() < footnote_p),
    )

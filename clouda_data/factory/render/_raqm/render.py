"""Clean page rendering with correct Arabic RTL shaping.

Uses Pillow's RAQM layout engine (HarfBuzz + FriBiDi), which performs real
Arabic cursive shaping, contextual glyph selection and the Unicode
bidirectional algorithm. No manual reshaping or bidi reversal is done here -
those hand-rolled approaches drop diacritics and mis-order mixed Arabic/English
runs.

Justification note
------------------
Justified lines are drawn token-by-token so word gaps can be stretched. That is
safe for Arabic because a space always breaks cursive joining, so a token
rendered alone is glyph-identical to the same token inside the full line. It is
*not* safe when the line contains a strong left-to-right run (Latin words),
because bidi would reorder those tokens as a group. Such lines are therefore
drawn as a single RAQM run and merely right-aligned.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from PIL import Image, ImageDraw, ImageFont

from .layouts import LayoutSpec, pt_to_px

import logging

log = logging.getLogger("clouda_data_factory.render._raqm.render")

from .pagemodel import ComposedPage, LineBox, PlacedBlock

_RAQM = ImageFont.Layout.RAQM


class RenderError(RuntimeError):
    """Raised when a page cannot be rendered faithfully."""


def has_strong_ltr(text: str) -> bool:
    """True if the text contains a strong left-to-right character.

    Digits are excluded on purpose: European digits inside Arabic text are
    bidi class EN (weak) and stay in place within the RTL token order, so they
    do not prevent token-wise justification.
    """
    return any(unicodedata.bidirectional(ch) == "L" for ch in text)


class FontBook:
    """Loads, caches and validates the Arabic-capable fonts."""

    def __init__(self, layout_cfg: Mapping, root: Path) -> None:
        fonts_cfg = layout_cfg["fonts"]
        configured = Path(fonts_cfg.get("dir", "assets/fonts"))
        # An absolute configured directory (canonical assets tree) wins;
        # a relative directory resolves against the repository root.
        self._dir = configured if configured.is_absolute() else Path(root) / configured
        self._families: dict[str, dict] = dict(fonts_cfg["families"])
        self._cache: dict[tuple[str, bool, int], ImageFont.FreeTypeFont] = {}
        self._cmap_cache: dict[tuple[str, bool], set[int]] = {}
        missing = [
            f"{name}/{style}"
            for name, spec in self._families.items()
            for style in ("regular", "bold")
            if not (self._dir / spec[style]).is_file()
        ]
        if missing:
            raise RenderError(
                f"font files not found in {self._dir}: {', '.join(missing)}"
            )

    @property
    def families(self) -> list[str]:
        return list(self._families)

    def path(self, family: str, bold: bool = False) -> Path:
        if family not in self._families:
            raise RenderError(f"unknown font family {family!r} (have {self.families})")
        spec = self._families[family]
        return self._dir / spec["bold" if bold else "regular"]

    def get(self, family: str, px: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        key = (family, bold, int(px))
        font = self._cache.get(key)
        if font is None:
            font = ImageFont.truetype(
                str(self.path(family, bold)), int(px), layout_engine=_RAQM
            )
            self._cache[key] = font
        return font

    def _cmap(self, family: str, bold: bool) -> set[int]:
        key = (family, bold)
        if key not in self._cmap_cache:
            from fontTools.ttLib import TTFont

            with TTFont(str(self.path(family, bold)), lazy=True) as tt:
                self._cmap_cache[key] = set(tt.getBestCmap())
        return self._cmap_cache[key]

    def missing_glyphs(
        self, text: str, families: Iterable[str] | None = None
    ) -> dict[str, set[str]]:
        """Characters in `text` that a family cannot render.

        Run before generation: an uncovered character would silently render as
        a .notdef box while the ground truth still claimed the real character.
        """
        report: dict[str, set[str]] = {}
        # Whitespace and zero-width formatting marks need no glyph.
        candidates = {
            ch
            for ch in set(text)
            if not ch.isspace() and unicodedata.category(ch) != "Cf"
        }
        for family in families or self.families:
            for bold in (False, True):
                cmap = self._cmap(family, bold)
                gaps = {ch for ch in candidates if ord(ch) not in cmap}
                if gaps:
                    report.setdefault(
                        f"{family}{'-bold' if bold else ''}", set()
                    ).update(gaps)
        return report


class TextMeasurer:
    """Measures shaped text widths, with a cache keyed by (font, string)."""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, str], float] = {}

    def width(self, font: ImageFont.FreeTypeFont, text: str) -> float:
        if not text:
            return 0.0
        key = (id(font), text)
        hit = self._cache.get(key)
        if hit is None:
            hit = float(font.getlength(text, direction="rtl", language="ar"))
            self._cache[key] = hit
        return hit

    def wrap(
        self,
        font: ImageFont.FreeTypeFont,
        text: str,
        max_width: float,
        first_line_width: float | None = None,
    ) -> list["WrappedLine"]:
        """Greedy word wrap that records character offsets into `text`.

        Offsets are what allow a paragraph to be split across a page boundary
        without ever reconstructing its text: the page keeps the exact
        substring `text[:line.end]` and the remainder carries on to the next
        page. Hard line breaks in the source are preserved. A token wider than
        the column gets its own line rather than being broken, so no source
        character is ever dropped or hyphenated.
        """
        if max_width <= 0:
            raise RenderError("column width must be positive")
        lines: list[WrappedLine] = []

        for match in re.finditer(r"[^\n]*", text):
            if match.start() == match.end() == len(text) and lines:
                break  # zero-width match at end of string
            hard_line, base = match.group(0), match.start()
            tokens = [
                (m.group(0), base + m.start(), base + m.end())
                for m in re.finditer(r"\S+", hard_line)
            ]
            if not tokens:
                continue

            current: list[tuple[str, int, int]] = []
            for token in tokens:
                avail = (
                    first_line_width
                    if (first_line_width is not None and not lines and not current)
                    else max_width
                )
                trial = " ".join(t[0] for t in current + [token])
                if current and self.width(font, trial) > avail:
                    lines.append(_make_line(current, False))
                    current = [token]
                else:
                    current.append(token)
            if current:
                lines.append(_make_line(current, True))

        if lines:
            lines[-1] = WrappedLine(
                lines[-1].text, lines[-1].start, lines[-1].end, True
            )
        return lines


@dataclass(frozen=True)
class WrappedLine:
    """One wrapped line: render text plus its span in the source string."""

    text: str
    start: int
    end: int
    hard_end: bool


def _make_line(tokens: list[tuple[str, int, int]], hard_end: bool) -> WrappedLine:
    return WrappedLine(
        text=" ".join(t[0] for t in tokens),
        start=tokens[0][1],
        end=tokens[-1][2],
        hard_end=hard_end,
    )


@dataclass(frozen=True)
class Frame:
    """A rectangular text area in page pixels."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


class PageRenderer:
    """Draws a `ComposedPage` onto a clean greyscale page image."""

    def __init__(self, layout_cfg: Mapping, fontbook: FontBook) -> None:
        self._cfg = layout_cfg
        self._fonts = fontbook
        self._typo = layout_cfg.get("typography", {})
        self._page_cfg = layout_cfg["page"]

    def render(self, page: ComposedPage, layout: LayoutSpec) -> Image.Image:
        background = int(self._page_cfg.get("background", 255))
        img = Image.new("L", (page.width, page.height), background)
        draw = ImageDraw.Draw(img)
        ink = int(layout.ink)

        for block in page.blocks:
            font = self._font_for(block.role, layout)
            for line in block.lines:
                self._draw_line(draw, line, font, ink)
            if block.marker:
                self._draw_marker(draw, block, layout, ink)

        for rule in page.meta.get("rules", []):
            draw.line(
                (rule["x0"], rule["y"], rule["x1"], rule["y"]),
                fill=min(255, ink + 60),
                width=max(1, int(round(rule.get("width", 1)))),
            )
        return img

    # -- internals -------------------------------------------------------
    def _font_for(self, role: str, layout: LayoutSpec) -> ImageFont.FreeTypeFont:
        dpi = int(self._page_cfg["dpi"])
        body_px = pt_to_px(layout.body_pt, dpi)
        if role == "title":
            return self._fonts.get(
                layout.font_family,
                max(1, int(round(body_px * layout.heading_scale * 1.15))),
                bold=True,
            )
        if role == "heading":
            return self._fonts.get(
                layout.font_family,
                max(1, int(round(body_px * layout.heading_scale))),
                bold=True,
            )
        if role == "footnote":
            return self._fonts.get(
                layout.font_family,
                max(
                    1,
                    int(round(body_px * float(self._typo.get("footnote_scale", 0.82)))),
                ),
            )
        if role in ("running_header", "running_footer"):
            return self._fonts.get(
                layout.font_family,
                max(
                    1,
                    int(round(body_px * float(self._typo.get("running_scale", 0.80)))),
                ),
            )
        if role == "page_number":
            return self._fonts.get(
                layout.font_family,
                max(
                    1,
                    int(
                        round(
                            body_px * float(self._typo.get("page_number_scale", 0.85))
                        )
                    ),
                ),
            )
        return self._fonts.get(layout.font_family, body_px)

    def _draw_line(
        self,
        draw: ImageDraw.ImageDraw,
        line: LineBox,
        font: ImageFont.FreeTypeFont,
        ink: int,
    ) -> None:
        if not line.text:
            return
        if line.justify and line.extra_space > 0 and not has_strong_ltr(line.text):
            self._draw_justified(draw, line, font, ink)
            return
        draw.text(
            (line.x_right, line.y_top),
            line.text,
            font=font,
            fill=ink,
            anchor="ra",
            direction="rtl",
            language="ar",
        )

    def _draw_justified(
        self,
        draw: ImageDraw.ImageDraw,
        line: LineBox,
        font: ImageFont.FreeTypeFont,
        ink: int,
    ) -> None:
        """Place tokens right-to-left with widened gaps."""
        tokens = line.text.split()
        if len(tokens) < 2:
            draw.text(
                (line.x_right, line.y_top),
                line.text,
                font=font,
                fill=ink,
                anchor="ra",
                direction="rtl",
                language="ar",
            )
            return
        space_w = float(font.getlength(" ", direction="rtl", language="ar"))
        gap = space_w + line.extra_space
        x = line.x_right
        for token in tokens:  # logical order == right-to-left visual order
            draw.text(
                (x, line.y_top),
                token,
                font=font,
                fill=ink,
                anchor="ra",
                direction="rtl",
                language="ar",
            )
            x -= float(font.getlength(token, direction="rtl", language="ar")) + gap

    def _draw_marker(
        self,
        draw: ImageDraw.ImageDraw,
        block: PlacedBlock,
        layout: LayoutSpec,
        ink: int,
    ) -> None:
        """Draw a superscript footnote reference marker."""
        if not block.lines or not block.marker:
            return
        dpi = int(self._page_cfg["dpi"])
        body_px = pt_to_px(layout.body_pt, dpi)
        scale = float(self._typo.get("footnote_marker_scale", 0.62))
        mfont = self._fonts.get(layout.font_family, max(1, int(round(body_px * scale))))
        last = block.lines[-1]
        if block.role == "footnote":
            # Marker sits to the right of the note text (start of an RTL line).
            x = last.x_right + float(mfont.getlength("  ", direction="rtl"))
            y = last.y_top
        else:
            # Marker sits at the end of the paragraph, i.e. to its left.
            x = last.x_right - last.width - 1.0
            y = last.y_top - body_px * 0.18
        draw.text(
            (x, y),
            block.marker,
            font=mfont,
            fill=ink,
            anchor="ra",
            direction="rtl",
            language="ar",
        )

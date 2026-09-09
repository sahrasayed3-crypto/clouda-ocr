"""Composition of source segments into laid-out pages.

The paginator walks the corpus as an ordered stream and fills each page's
columns. When a paragraph does not fit, it is split at a *character offset*, so
the part on this page and the part on the next are both exact substrings of the
source file. Nothing is rewritten, reflowed into new wording, or duplicated.

Reading order for a two-column Arabic page is right column first (column 0),
then left column (column 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


from .config_shim import LayoutConfigView
from .corpus import Segment, SourceDocument
from .layouts import LayoutSpec, mm_to_px, pick_layout, pt_to_px, to_numerals

import logging

log = logging.getLogger("clouda_data_factory.render._raqm.paginate")

from .pagemodel import ComposedPage, LineBox, PlacedBlock
from .render import FontBook, Frame, TextMeasurer, has_strong_ltr

log = logging.getLogger("clouda_data_factory.render._raqm.paginate")

_ROLE_FOR_KIND = {"title": "title", "heading": "heading", "paragraph": "paragraph"}


class PaginationError(RuntimeError):
    """Raised when a page cannot be composed."""


@dataclass
class _Cursor:
    """Position in the segment stream, snapshot-able for two-pass layout."""

    index: int = 0
    remainder: Segment | None = None


class SegmentStream:
    """Ordered, splittable view over all corpus segments."""

    def __init__(self, segments: Sequence[Segment], allow_cycle: bool = False) -> None:
        self._segments = list(segments)
        self._cursor = _Cursor()
        self._allow_cycle = allow_cycle
        self._cycles = 0

    @property
    def cycles(self) -> int:
        return self._cycles

    @property
    def exhausted(self) -> bool:
        return self._cursor.remainder is None and self._cursor.index >= len(
            self._segments
        )

    def snapshot(self) -> _Cursor:
        return _Cursor(self._cursor.index, self._cursor.remainder)

    def restore(self, snap: _Cursor) -> None:
        self._cursor = _Cursor(snap.index, snap.remainder)

    def peek(self) -> Segment | None:
        if self._cursor.remainder is not None:
            return self._cursor.remainder
        if self._cursor.index < len(self._segments):
            return self._segments[self._cursor.index]
        if self._allow_cycle and self._segments:
            self._cursor.index = 0
            self._cycles += 1
            log.warning(
                "source text exhausted - cycling corpus (pass %d). "
                "Supply more source text for a fully unique dataset.",
                self._cycles + 1,
            )
            return self._segments[0]
        return None

    def take(self) -> Segment | None:
        seg = self.peek()
        if seg is None:
            return None
        if self._cursor.remainder is not None:
            self._cursor.remainder = None
        else:
            self._cursor.index += 1
        return seg

    def push_remainder(self, seg: Segment) -> None:
        if self._cursor.remainder is not None:  # pragma: no cover - defensive
            raise PaginationError("segment stream already holds a remainder")
        self._cursor.remainder = seg


class Paginator:
    """Turns a corpus into a list of `ComposedPage`s."""

    def __init__(
        self,
        config: LayoutConfigView,
        fontbook: FontBook,
        measurer: TextMeasurer,
        global_seed: int,
    ) -> None:
        self._config = config
        self._fonts = fontbook
        self._measure = measurer
        self._seed = global_seed
        self._layout_cfg = config.layout
        self._page_cfg = config.layout["page"]
        self._typo = config.layout.get("typography", {})
        self._pag_cfg = config.layout.get("pagination", {})
        self._dpi = int(self._page_cfg["dpi"])
        self._width = mm_to_px(self._page_cfg["width_mm"], self._dpi)
        self._height = mm_to_px(self._page_cfg["height_mm"], self._dpi)

    # -- public ----------------------------------------------------------
    def paginate(
        self,
        docs: Sequence[SourceDocument],
        max_pages: int,
        allow_cycle: bool = False,
    ) -> list[ComposedPage]:
        segments = [seg for doc in docs for seg in doc.segments]
        if not segments:
            raise PaginationError("corpus contains no segments")
        titles = {doc.doc_id: doc.title for doc in docs}
        stream = SegmentStream(segments, allow_cycle=allow_cycle)

        pages: list[ComposedPage] = []
        section = ""
        for index in range(max_pages):
            if stream.exhausted and not allow_cycle:
                break
            layout = pick_layout(self._config, index, self._seed)
            page, section = self._compose_page(index, layout, stream, titles, section)
            if page is None:
                break
            pages.append(page)

        if len(pages) < max_pages:
            log.warning(
                "requested %d pages but the source text yielded %d. "
                "Add more text to input/ (or pass --allow-cycle to reuse it).",
                max_pages,
                len(pages),
            )
        log.info("composed %d page(s)", len(pages))
        return pages

    # -- geometry --------------------------------------------------------
    def _bands(self, layout: LayoutSpec) -> tuple[Frame, dict[str, float]]:
        mt, mr, mb, ml = (mm_to_px(v, self._dpi) for v in layout.margins_mm)
        body_px = pt_to_px(layout.body_pt, self._dpi)
        run_px = max(
            1, int(round(body_px * float(self._typo.get("running_scale", 0.8))))
        )
        pn_px = max(
            1, int(round(body_px * float(self._typo.get("page_number_scale", 0.85))))
        )

        bands: dict[str, float] = {}
        top = float(mt)
        pn_top = layout.page_number and layout.page_number_position == "top_outer"
        if pn_top:
            bands["page_number_y"] = top
            top += pn_px * 1.6
        if layout.running_header:
            bands["header_y"] = top
            top += run_px * 1.35
            if layout.header_rule:
                bands["header_rule_y"] = top
                top += run_px * 0.45

        bottom = float(self._height - mb)
        if layout.page_number and not pn_top:
            bottom -= pn_px * 1.6
            bands["page_number_y"] = bottom
        if layout.running_footer:
            bottom -= run_px * 1.5
            bands["footer_y"] = bottom

        frame = Frame(x0=float(ml), y0=top, x1=float(self._width - mr), y1=bottom)
        if frame.height <= body_px * 4:
            raise PaginationError(
                f"page frame too small for layout {layout.name}: {frame.height:.0f}px"
            )
        return frame, bands

    def _columns(self, frame: Frame, layout: LayoutSpec) -> list[Frame]:
        if layout.columns == 1:
            return [frame]
        gutter = mm_to_px(self._page_cfg.get("column_gutter_mm", 8.0), self._dpi)
        col_w = (frame.width - gutter * (layout.columns - 1)) / layout.columns
        cols = []
        for i in range(layout.columns):
            # Column 0 is the rightmost - Arabic reading order.
            x1 = frame.x1 - i * (col_w + gutter)
            cols.append(Frame(x0=x1 - col_w, y0=frame.y0, x1=x1, y1=frame.y1))
        return cols

    # -- composition -----------------------------------------------------
    def _compose_page(
        self,
        index: int,
        layout: LayoutSpec,
        stream: SegmentStream,
        titles: dict[str, str],
        section: str,
    ) -> tuple[ComposedPage | None, str]:
        frame, bands = self._bands(layout)
        body_px = pt_to_px(layout.body_pt, self._dpi)
        line_h = body_px * layout.line_spacing

        snapshot = stream.snapshot()
        blocks, new_section = self._fill_columns(frame, layout, stream, section, 0.0)
        if not blocks:
            return None, section

        # Two-pass footnote handling: lay the body out first, then - if this
        # page is flagged for a footnote and the *next* segment in the stream is
        # short enough - redo the fill with the footnote's height reserved. The
        # footnote is therefore the natural continuation of the page's text, so
        # source order is preserved exactly.
        footnote_block: PlacedBlock | None = None
        rules: list[dict] = []
        nxt = stream.peek()
        max_fn = int(self._pag_cfg.get("footnote_max_chars", 220))
        if (
            layout.allow_footnote
            and nxt is not None
            and nxt.kind == "paragraph"
            and len(nxt.text) <= max_fn
        ):
            fn_font = self._fonts.get(
                layout.font_family,
                max(
                    1,
                    int(round(body_px * float(self._typo.get("footnote_scale", 0.82)))),
                ),
            )
            fn_line_h = fn_font.size * layout.line_spacing
            fn_lines = self._measure.wrap(fn_font, nxt.text, frame.width)
            reserve = fn_line_h * len(fn_lines) + line_h * 0.9
            if reserve < frame.height * 0.4:
                stream.restore(snapshot)
                blocks, new_section = self._fill_columns(
                    frame, layout, stream, section, reserve
                )
                if blocks:
                    seg = stream.take()
                    if seg is not None:
                        y0 = frame.y1 - fn_line_h * len(fn_lines)
                        rules.append(
                            {
                                "x0": frame.x1 - frame.width * 0.38,
                                "x1": frame.x1,
                                "y": y0 - line_h * 0.45,
                                "width": self._rule_px(),
                            }
                        )
                        footnote_block = PlacedBlock(
                            role="footnote",
                            text=seg.text,
                            column=0,
                            doc_id=seg.doc_id,
                            start=seg.start,
                            end=seg.end,
                            marker=to_numerals(1, layout.numeral_system),
                            lines=[
                                LineBox(
                                    text=ln.text,
                                    x_right=frame.x1,
                                    y_top=y0 + i * fn_line_h,
                                    width=self._measure.width(fn_font, ln.text),
                                    height=fn_line_h,
                                    font_key="footnote",
                                )
                                for i, ln in enumerate(fn_lines)
                            ],
                        )
                        # Reference marker on the last body paragraph.
                        for blk in reversed(blocks):
                            if blk.role == "paragraph":
                                blk.marker = footnote_block.marker
                                break

        page = ComposedPage(
            page_id=f"page_{index + 1:06d}",
            index=index,
            width=self._width,
            height=self._height,
            dpi=self._dpi,
            layout_name=layout.name,
            font_family=layout.font_family,
            blocks=list(blocks),
        )

        # The running head names the section in effect when the page opens,
        # not one that starts partway down it.
        header_text = section
        self._add_furniture(
            page, layout, frame, bands, rules, header_text, titles, index
        )
        if footnote_block is not None:
            page.blocks.append(footnote_block)

        page.meta.update(
            {
                "rules": rules,
                "layout": layout.as_dict(),
                "columns": layout.columns,
                "density": layout.density,
                "has_footnote": footnote_block is not None,
                "has_header": layout.running_header,
                "has_footer": layout.running_footer,
                "has_page_number": layout.page_number,
                "numeral_system": layout.numeral_system,
                "justified": layout.justify,
            }
        )
        page.consumed = [
            Segment(
                doc_id=b.doc_id or "",
                index=i,
                kind="paragraph",
                level=0,
                text=b.text,
                start=b.start or 0,
                end=b.end or 0,
            )
            for i, b in enumerate(page.blocks)
            if b.from_source
        ]
        return page, new_section

    def _rule_px(self) -> float:
        return float(self._typo.get("rule_px", 1.5)) * (self._dpi / 150.0)

    def _fill_columns(
        self,
        frame: Frame,
        layout: LayoutSpec,
        stream: SegmentStream,
        section: str,
        bottom_reserve: float,
    ) -> tuple[list[PlacedBlock], str]:
        """Fill every column top-to-bottom from the stream."""
        blocks: list[PlacedBlock] = []
        current_section = section
        body_px = pt_to_px(layout.body_pt, self._dpi)
        limit_ratio = float(layout.fill)

        for col_index, col in enumerate(self._columns(frame, layout)):
            usable_bottom = col.y1 - (bottom_reserve if col_index == 0 else 0.0)
            # Density: stop early on sparse pages so they look genuinely sparse.
            stop_at = col.y0 + (usable_bottom - col.y0) * limit_ratio
            y = col.y0

            while True:
                seg = stream.peek()
                if seg is None:
                    break
                placed, y, consumed_all, split_at = self._place_segment(
                    seg, col, layout, y, stop_at, body_px
                )
                if placed is None:
                    break
                stream.take()
                if not consumed_all and split_at is not None:
                    remainder = seg.slice_text(split_at, seg.end, seg.index)
                    if remainder.text.strip():
                        stream.push_remainder(remainder)
                placed.column = col_index
                blocks.append(placed)
                if placed.role in ("title", "heading"):
                    current_section = placed.text
                if not consumed_all:
                    break

        return blocks, current_section

    def _place_segment(
        self,
        seg: Segment,
        col: Frame,
        layout: LayoutSpec,
        y: float,
        stop_at: float,
        body_px: int,
    ) -> tuple[PlacedBlock | None, float, bool, int | None]:
        """Try to place `seg` at `y`. Returns (block, new_y, consumed_all, split_offset)."""
        role = _ROLE_FOR_KIND[seg.kind]
        is_heading = role in ("title", "heading")

        if is_heading:
            scale = layout.heading_scale * (1.15 if role == "title" else 1.0)
            font = self._fonts.get(
                layout.font_family, max(1, int(round(body_px * scale))), bold=True
            )
            indent = 0.0
            gap_before = body_px * (0.9 if y > col.y0 else 0.0)
            gap_after = body_px * 0.45
        else:
            font = self._fonts.get(layout.font_family, body_px)
            indent = layout.indent_em * body_px
            gap_before = body_px * layout.para_gap_em if y > col.y0 else 0.0
            gap_after = 0.0

        line_h = font.size * layout.line_spacing
        y_start = y + gap_before
        first_w = col.width - indent
        lines = self._measure.wrap(font, seg.text, col.width, first_line_width=first_w)
        if not lines:
            return None, y, True, None

        room = stop_at - y_start
        max_lines = int(room // line_h) if line_h > 0 else 0
        if max_lines <= 0:
            return None, y, False, None

        min_orphan = int(self._pag_cfg.get("min_orphan_lines", 2))
        min_widow = int(self._pag_cfg.get("min_widow_lines", 2))
        keep_next = int(self._pag_cfg.get("heading_keep_with_next_lines", 2))

        if is_heading:
            # A heading must not be stranded at the foot of a column.
            if max_lines < len(lines) + keep_next:
                return None, y, False, None
            take = len(lines)
        elif max_lines >= len(lines):
            take = len(lines)
        else:
            take = max_lines
            if len(lines) - take < min_widow:
                take = len(lines) - min_widow
            if take < min_orphan:
                return None, y, False, None

        consumed_all = take >= len(lines)
        placed_lines = lines[:take]
        split_at = None if consumed_all else seg.start + placed_lines[-1].end
        block_text = seg.text if consumed_all else seg.text[: placed_lines[-1].end]

        boxes: list[LineBox] = []
        for i, ln in enumerate(placed_lines):
            x_right = col.x1 - (indent if (i == 0 and not is_heading) else 0.0)
            avail = col.width - (indent if (i == 0 and not is_heading) else 0.0)
            natural = self._measure.width(font, ln.text)
            justify, extra = self._justify_metrics(
                ln, layout, font, natural, avail, is_heading
            )
            boxes.append(
                LineBox(
                    text=ln.text,
                    x_right=x_right,
                    y_top=y_start + i * line_h,
                    width=natural,
                    height=line_h,
                    font_key=role,
                    justify=justify,
                    extra_space=extra,
                )
            )

        block = PlacedBlock(
            role=role,  # type: ignore
            text=block_text,
            lines=boxes,
            doc_id=seg.doc_id,
            start=seg.start,
            end=(seg.end if consumed_all else split_at),
        )
        return block, y_start + take * line_h + gap_after, consumed_all, split_at

    def _justify_metrics(
        self,
        line,
        layout: LayoutSpec,
        font,
        natural: float,
        avail: float,
        is_heading: bool,
    ) -> tuple[bool, float]:
        """Decide whether to justify this line, and by how much per gap."""
        if is_heading or not layout.justify or line.hard_end:
            return False, 0.0
        if has_strong_ltr(line.text):
            # Bidi would reorder Latin runs; draw as one shaped run instead.
            return False, 0.0
        n_tokens = len(line.text.split())
        if n_tokens < 2:
            return False, 0.0
        slack = avail - natural
        if slack <= 0:
            return False, 0.0
        extra = slack / (n_tokens - 1)
        space_w = float(font.getlength(" ", direction="rtl", language="ar"))
        cap = float(self._typo.get("max_justify_stretch", 3.0)) * space_w
        if extra > cap:
            return False, 0.0
        return True, extra

    def _add_furniture(
        self,
        page: ComposedPage,
        layout: LayoutSpec,
        frame: Frame,
        bands: dict[str, float],
        rules: list[dict],
        section: str,
        titles: dict[str, str],
        index: int,
    ) -> None:
        """Add running header/footer, page number and header rule."""
        body_px = pt_to_px(layout.body_pt, self._dpi)
        doc_id = next((b.doc_id for b in page.blocks if b.doc_id), None)
        doc_title = titles.get(doc_id or "", "")

        if layout.running_header and "header_y" in bands:
            text = (section or doc_title).strip()
            if text:
                font = self._fonts.get(
                    layout.font_family,
                    max(
                        1,
                        int(
                            round(body_px * float(self._typo.get("running_scale", 0.8)))
                        ),
                    ),
                )
                page.blocks.append(
                    self._single_line_block(
                        "running_header",
                        text,
                        font,
                        frame.x1,
                        bands["header_y"],
                        "right",
                    )
                )
            if layout.header_rule and "header_rule_y" in bands:
                rules.append(
                    {
                        "x0": frame.x0,
                        "x1": frame.x1,
                        "y": bands["header_rule_y"],
                        "width": self._rule_px(),
                    }
                )

        if layout.running_footer and "footer_y" in bands and doc_title:
            font = self._fonts.get(
                layout.font_family,
                max(
                    1, int(round(body_px * float(self._typo.get("running_scale", 0.8))))
                ),
            )
            page.blocks.append(
                self._single_line_block(
                    "running_footer",
                    doc_title,
                    font,
                    frame,
                    bands["footer_y"],
                    "center",
                )
            )

        if layout.page_number and "page_number_y" in bands:
            font = self._fonts.get(
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
            text = to_numerals(index + 1, layout.numeral_system)
            pos = layout.page_number_position
            if pos == "bottom_center":
                anchor_ref, mode = frame, "center"
            else:
                # Outer edge: alternates like a real bound book.
                outer_right = index % 2 == 0
                anchor_ref, mode = (
                    frame.x1  # type: ignore
                    if outer_right
                    else frame.x0 + self._measure.width(font, text)
                ), "right"
            page.blocks.append(
                self._single_line_block(
                    "page_number", text, font, anchor_ref, bands["page_number_y"], mode
                )
            )

    def _single_line_block(
        self, role: str, text: str, font, ref, y: float, mode: str
    ) -> PlacedBlock:
        width = self._measure.width(font, text)
        if mode == "center":
            x_right = ref.x0 + (ref.width + width) / 2.0
        else:
            x_right = float(ref)
        return PlacedBlock(
            role=role,  # type: ignore
            text=text,
            lines=[
                LineBox(
                    text=text,
                    x_right=x_right,
                    y_top=y,
                    width=width,
                    height=font.size,
                    font_key=role,
                )
            ],
        )

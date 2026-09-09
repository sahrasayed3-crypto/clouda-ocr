"""Data structures describing a composed page.

A `ComposedPage` is the single source of truth for both the rendered image and
the ground-truth text file: the renderer draws it, and the ground-truth writer
serialises it. They cannot drift apart, which is what guarantees the ground
truth matches what is on the page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .corpus import Segment

BlockRole = Literal[
    "title",
    "heading",
    "paragraph",
    "footnote",
    "running_header",
    "running_footer",
    "page_number",
]

# Roles whose text comes from the user's source files (authoritative ground
# truth) rather than being synthesized by the layout engine.
SOURCE_ROLES: frozenset[str] = frozenset({"title", "heading", "paragraph", "footnote"})


@dataclass
class LineBox:
    """One laid-out line of text, with its resolved draw geometry."""

    text: str
    x_right: float
    y_top: float
    width: float
    height: float
    font_key: str
    justify: bool = False
    # Extra pixels to distribute between words when justifying. 0 = natural.
    extra_space: float = 0.0


@dataclass
class PlacedBlock:
    """A block of text placed into a column, already broken into lines."""

    role: BlockRole
    text: str
    lines: list[LineBox] = field(default_factory=list)
    column: int = 0
    # Provenance back into the user's source text. None for synthesized blocks
    # (running heads, page numbers).
    doc_id: str | None = None
    start: int | None = None
    end: int | None = None
    # Footnote reference marker rendered in the body, e.g. "١".
    marker: str | None = None

    @property
    def from_source(self) -> bool:
        return self.role in SOURCE_ROLES and self.doc_id is not None


@dataclass
class ComposedPage:
    """Everything needed to render one page and write its ground truth."""

    page_id: str
    index: int
    width: int
    height: int
    dpi: int
    layout_name: str
    font_family: str
    blocks: list[PlacedBlock] = field(default_factory=list)
    consumed: list[Segment] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # -- ground truth ----------------------------------------------------
    def ground_truth_text(self) -> str:
        """Full page text in Arabic reading order, exactly as rendered.

        Includes running heads and page numbers, because an OCR engine reading
        this page will emit them too. Blocks are separated by a single newline;
        the text *within* each block is the untouched source substring, so no
        soft-wrap line breaks are invented here.
        """
        parts: list[str] = []
        for b in self.reading_order():
            if b.marker and b.role == "footnote":
                parts.append(f"{b.marker} {b.text}")
            elif b.marker:
                parts.append(f"{b.text}{b.marker}")
            else:
                parts.append(b.text)
        return "\n".join(parts) + "\n"

    def body_text(self) -> str:
        """Only the text that came from the user's source files."""
        return "\n".join(b.text for b in self.reading_order() if b.from_source) + "\n"

    def reading_order(self) -> list[PlacedBlock]:
        """Blocks in Arabic reading order.

        Running header first, then body columns right-to-left (column 0 is the
        rightmost), then footnotes, then footer and page number.
        """
        order = {"running_header": 0, "page_number_top": 1}
        head = [b for b in self.blocks if b.role == "running_header"]
        body = [b for b in self.blocks if b.role in ("title", "heading", "paragraph")]
        notes = [b for b in self.blocks if b.role == "footnote"]
        tail = [b for b in self.blocks if b.role in ("running_footer", "page_number")]
        body.sort(key=lambda b: (b.column, b.lines[0].y_top if b.lines else 0.0))
        del order
        return head + body + notes + tail

    def source_spans(self) -> list[dict]:
        """Provenance records: which source characters landed on this page."""
        return [
            {
                "doc_id": b.doc_id,
                "role": b.role,
                "start": b.start,
                "end": b.end,
                "chars": (
                    (b.end - b.start)
                    if (b.start is not None and b.end is not None)
                    else 0
                ),
            }
            for b in self.reading_order()
            if b.from_source
        ]

    def to_structure(self) -> dict:
        """Structured sidecar describing every block, for later scoring."""
        return {
            "page_id": self.page_id,
            "index": self.index,
            "width": self.width,
            "height": self.height,
            "dpi": self.dpi,
            "layout": self.layout_name,
            "font_family": self.font_family,
            "meta": self.meta,
            "blocks": [
                {
                    "role": b.role,
                    "column": b.column,
                    "text": b.text,
                    "origin": "source" if b.from_source else "synthesized",
                    "doc_id": b.doc_id,
                    "start": b.start,
                    "end": b.end,
                    "marker": b.marker,
                    "rendered_lines": [ln.text for ln in b.lines],
                }
                for b in self.reading_order()
            ],
        }

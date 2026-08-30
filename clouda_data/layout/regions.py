from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RegionKind(str, Enum):
    BODY = "body"
    TITLE = "title"
    HEADING = "heading"
    FOOTNOTE = "footnote"
    MARGIN = "margin"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    TABLE = "table"
    IMAGE = "image"
    ARABIC_PARAGRAPH = "arabic_paragraph"
    ENGLISH_FRAGMENT = "english_fragment"
    EMPTY = "empty"
    BINDING_EDGE = "binding_edge"


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)


@dataclass(frozen=True)
class Region:
    id: str
    kind: RegionKind
    box: Box
    reading_order: int
    priority: int = 5
    text_role: str = "reference"

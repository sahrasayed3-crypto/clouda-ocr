"""Render backend interface (MERGE_PLAN.md phase 4).

A backend turns one source document into clean output:

    render(source) -> RenderResult(clean_pdf_path, page_paths, layout_record)

Backends are interchangeable; the renderer actually used is recorded per
page in the run manifest. Both bundled backends are vendored/adapted from
the two legacy systems so their historical behavior is preserved.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RenderResult:
    clean_pdf: Path | None
    pages: list[Path]
    searchable: bool
    renderer: str
    layout: dict[str, Any] = field(default_factory=dict)


class RenderBackend(ABC):
    """One source -> clean PDF and/or clean page images."""

    name: str = "abstract"
    searchable: bool = False

    @abstractmethod
    def render(
        self,
        text: str | None,
        image_path: Path | None,
        out_dir: Path,
        style_seed: int,
        dpi: int = 150,
    ) -> RenderResult:
        """Render one document. Exactly one of text/image_path is provided."""
        raise NotImplementedError

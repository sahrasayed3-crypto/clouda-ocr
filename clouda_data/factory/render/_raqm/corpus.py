"""Source-text ingestion.

Reads the user's Arabic / mixed Arabic-English text files verbatim and slices
them into an ordered stream of `Segment`s. A segment's `text` is always an
exact substring of the decoded source document:

    segment.text == document.text[segment.start:segment.end]

Nothing here rewrites, normalises, spell-corrects, reshapes, transliterates or
infers text. There is no OCR and no model of any kind in this path - the
ground truth is the input, sliced.

Optional lightweight structural markup is recognised so that pages can carry
headings drawn from the user's own text:

    # Title
    ## Section heading
    body paragraph ...

The `#` markers themselves are treated as markup and fall outside every
segment's [start, end) span, which keeps the substring invariant intact.
Detection is automatic: if a file contains no `#` headings it is read as plain
paragraphs, so unmarked prose works with no configuration.
"""

from __future__ import annotations

import codecs
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Sequence

from ....provenance.hashing import sha256_bytes, sha256_text

import logging

log = logging.getLogger("clouda_data_factory.render._raqm.corpus")

SegmentKind = Literal["title", "heading", "paragraph"]

# Encodings tried in order. utf-8 covers virtually all modern Arabic text;
# cp1256 / iso-8859-6 are legacy Windows/Unix Arabic encodings kept as a
# fallback so an old file fails loudly rather than silently producing mojibake.
_ENCODINGS = ("utf-8", "cp1256", "iso-8859-6")

# NB: no `^` anchor - this is used with `pattern.match(text, start, end)`,
# which already anchors at `start`. A `^` would additionally demand a
# real line start and would silently fail to match indented blocks.
_HEADING_RE = re.compile(r"(#{1,6})[ \t]+(?=\S)")
_BLANK_SPLIT_RE = re.compile(r"\n[ \t]*\n")

# Characters that carry no printable ink and may legitimately be dropped when
# measuring "did we preserve every meaningful character".
_IGNORABLE = {
    "​",  # zero-width space
    "‎",  # LRM
    "‏",  # RLM
    "﻿",  # BOM
}


@dataclass(frozen=True)
class Segment:
    """One structural unit of source text, with exact provenance."""

    doc_id: str
    index: int
    kind: SegmentKind
    level: int
    text: str
    start: int
    end: int

    @property
    def char_count(self) -> int:
        return len(self.text)

    def slice_text(self, start: int, end: int, new_index: int) -> "Segment":
        """Return a sub-segment covering [start, end) of the *document*.

        Used by the paginator when a paragraph must break across a page
        boundary. The result is still an exact substring of the document.
        """
        if not (self.start <= start <= end <= self.end):
            raise ValueError(
                f"sub-slice [{start},{end}) escapes segment span [{self.start},{self.end})"
            )
        return Segment(
            doc_id=self.doc_id,
            index=new_index,
            kind=self.kind,
            level=self.level,
            text=self.text[start - self.start : end - self.start],
            start=start,
            end=end,
        )


@dataclass(frozen=True)
class SourceDocument:
    """A single input file, decoded but otherwise untouched."""

    doc_id: str
    path: Path
    text: str
    encoding: str
    byte_sha256: str
    text_sha256: str
    segments: tuple[Segment, ...]

    @property
    def title(self) -> str:
        """The document's own title segment, or its filename stem."""
        for seg in self.segments:
            if seg.kind == "title":
                return seg.text
        return self.path.stem

    def verify_segment_invariant(self) -> None:
        """Assert every segment is an exact substring at its recorded offsets.

        Raises ValueError on any violation. Called after parsing and again by
        the test-suite; cheap enough to run on every pipeline invocation.
        """
        prev_end = 0
        for seg in self.segments:
            if seg.start < prev_end:
                raise ValueError(
                    f"{self.doc_id}: segment {seg.index} overlaps the previous segment"
                )
            if self.text[seg.start : seg.end] != seg.text:
                raise ValueError(
                    f"{self.doc_id}: segment {seg.index} text does not match its offsets"
                )
            gap = self.text[prev_end : seg.start]
            if gap.strip().strip("#"):
                raise ValueError(
                    f"{self.doc_id}: non-markup text dropped before segment {seg.index}: "
                    f"{gap.strip()[:60]!r}"
                )
            prev_end = seg.end
        tail = self.text[prev_end:]
        if tail.strip().strip("#"):
            raise ValueError(
                f"{self.doc_id}: trailing text dropped: {tail.strip()[:60]!r}"
            )


class CorpusError(RuntimeError):
    """Raised when source text cannot be ingested."""


def _decode(raw: bytes, path: Path) -> tuple[str, str]:
    """Decode bytes, trying known Arabic-capable encodings in order."""
    if raw.startswith(codecs.BOM_UTF8):
        return raw.decode("utf-8-sig"), "utf-8-sig"

    last_error: Exception | None = None
    for enc in _ENCODINGS:
        try:
            text = raw.decode(enc)
        except (UnicodeDecodeError, LookupError) as exc:
            last_error = exc
            continue
        if enc not in ("utf-8", "utf-8-sig"):
            log.warning(
                "%s decoded as %s (not UTF-8) - verify the text renders as expected",
                path.name,
                enc,
            )
        return text, enc
    raise CorpusError(f"cannot decode {path} with any of {_ENCODINGS}: {last_error}")


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink [start, end) so it excludes surrounding whitespace (incl. \\r)."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def parse_segments(
    doc_id: str, text: str, detect_headings: bool = True
) -> list[Segment]:
    """Split a document into title / heading / paragraph segments.

    Blank lines separate blocks. Hard line breaks *inside* a block are
    preserved verbatim and rendered as hard breaks, so verse and address-style
    layouts survive intact.
    """
    segments: list[Segment] = []
    cursor = 0
    seen_title = False

    for raw_block in _BLANK_SPLIT_RE.split(text):
        # Locate this block in the document rather than trusting split offsets.
        block_start = text.find(raw_block, cursor)
        if block_start < 0:  # pragma: no cover - defensive
            raise CorpusError(f"{doc_id}: internal error locating block offset")
        block_end = block_start + len(raw_block)
        cursor = block_end

        start, end = _trim_span(text, block_start, block_end)
        if start >= end:
            continue

        kind: SegmentKind = "paragraph"
        level = 0
        if detect_headings:
            match = _HEADING_RE.match(text, start, end)
            if match:
                level = len(match.group(1))
                # Move the span past the '#' markup so `text[start:end]` stays exact.
                start = match.end()
                start, end = _trim_span(text, start, end)
                if start >= end:
                    continue
                if level == 1 and not seen_title:
                    kind, seen_title = "title", True
                else:
                    kind = "heading"

        segments.append(
            Segment(
                doc_id=doc_id,
                index=len(segments),
                kind=kind,
                level=level,
                text=text[start:end],
                start=start,
                end=end,
            )
        )

    return segments


def load_document(path: Path, markup: str = "auto") -> SourceDocument:
    """Ingest one source file."""
    raw = path.read_bytes()
    if not raw.strip():
        raise CorpusError(f"{path} is empty")
    text, encoding = _decode(raw, path)

    if markup == "auto":
        detect = bool(re.search(r"^#{1,6}[ \t]+\S", text, flags=re.MULTILINE))
    elif markup == "hash":
        detect = True
    elif markup == "none":
        detect = False
    else:
        raise CorpusError(f"unknown markup mode {markup!r} (use auto|hash|none)")

    doc_id = path.stem
    segments = parse_segments(doc_id, text, detect_headings=detect)
    if not segments:
        raise CorpusError(f"{path} contains no usable text segments")

    doc = SourceDocument(
        doc_id=doc_id,
        path=path,
        text=text,
        encoding=encoding,
        byte_sha256=sha256_bytes(raw),
        text_sha256=sha256_text(text),
        segments=tuple(segments),
    )
    doc.verify_segment_invariant()
    log.info(
        "ingested %s: %d segments, %d chars, encoding=%s, headings=%s",
        path.name,
        len(segments),
        len(text),
        encoding,
        "on" if detect else "off",
    )
    return doc


def load_corpus(
    input_dir: Path,
    patterns: Sequence[str] = ("*.txt", "*.md"),
    markup: str = "auto",
) -> list[SourceDocument]:
    """Ingest every source file in `input_dir`, sorted by filename.

    Files with identical byte content are de-duplicated: the first occurrence
    (by sorted filename) wins and later copies are reported and skipped, so a
    document supplied twice under different names cannot inflate the dataset.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise CorpusError(f"input directory not found: {input_dir}")

    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(input_dir.glob(pattern)))
    paths = sorted(set(paths))
    if not paths:
        raise CorpusError(
            f"no source text found in {input_dir} (looked for {', '.join(patterns)}). "
            "Place your Arabic / mixed Arabic-English .txt files there."
        )

    docs: list[SourceDocument] = []
    by_hash: dict[str, str] = {}
    for path in paths:
        try:
            doc = load_document(path, markup=markup)
        except CorpusError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface, never swallow
            raise CorpusError(f"failed to ingest {path}: {exc}") from exc

        if doc.byte_sha256 in by_hash:
            log.warning(
                "duplicate source file skipped: %s is byte-identical to %s (sha256=%s)",
                path.name,
                by_hash[doc.byte_sha256],
                doc.byte_sha256[:12],
            )
            continue
        by_hash[doc.byte_sha256] = path.name
        docs.append(doc)

    log.info(
        "corpus: %d document(s), %d duplicate(s) skipped",
        len(docs),
        len(paths) - len(docs),
    )
    return docs


def iter_segments(docs: Sequence[SourceDocument]) -> Iterator[Segment]:
    """Yield every segment across all documents, in document then source order."""
    for doc in docs:
        yield from doc.segments


def significant_chars(text: str) -> str:
    """Text reduced to characters that must survive round-tripping.

    Drops whitespace and zero-width formatting marks; keeps everything that
    puts ink on the page. Used by the exactness tests.
    """
    return "".join(
        ch
        for ch in text
        if not ch.isspace()
        and ch not in _IGNORABLE
        and unicodedata.category(ch) != "Cf"
    )

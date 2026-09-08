"""Tiny synthetic fixture dataset builder for pre-training tests.

Everything is generated programmatically at test time (no binary blobs in
Git): images are a few hundred bytes each, text files are a few lines. The
fixture exercises every interesting case:

- source_a: Arabic, English, mixed text; two pages of one document;
  a normalized-text duplicate; an exact image duplicate; an orphan text;
  a junk file and a junk directory;
- source_b: JSONL records (one malformed row, one row pointing at a
  missing image) and a CSV record pair.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

ARABIC_TEXT = "اللغة العربية جميلة وكتابتها متصلة"
ENGLISH_TEXT = "Clouda OCR training data pipeline"
MIXED_TEXT = "مراجعة Review للنصوص texts المختلطة"
DUP_TEXT = "اللغة العربية جميلة وكتابتها متصلة"
ORPHAN_TEXT = "نص بلا صورة مقترنة"


def _tiny_png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (64, 48), color)
    for x in range(0, 64, 7):
        for y in range(0, 48, 5):
            image.putpixel((x, y), (255 - color[0], 255 - color[1], 255 - color[2]))
    image.save(path, format="PNG")


def build_tiny_dataset(root: Path) -> dict[str, Path]:
    """Create the fixture dataset; returns the two source roots."""

    root.mkdir(parents=True, exist_ok=True)
    source_a = root / "fixture_source_a"
    source_b = root / "fixture_source_b"

    # --- source_a: image + sidecar text ---------------------------------
    _tiny_png(source_a / "arabic" / "ar_page-p1.png", (240, 240, 240))
    (source_a / "arabic" / "ar_page-p1.txt").write_text(
        ARABIC_TEXT + "\n", encoding="utf-8"
    )
    # Two pages of one logical document.
    for page, color in ((1, (200, 210, 220)), (2, (210, 200, 220))):
        _tiny_png(source_a / "mixed" / f"mixed_doc-p{page}.png", color)
        (source_a / "mixed" / f"mixed_doc-p{page}.txt").write_text(
            f"{MIXED_TEXT} صفحة {page}\n", encoding="utf-8"
        )
    _tiny_png(source_a / "english" / "en_note.png", (220, 230, 200))
    (source_a / "english" / "en_note.txt").write_text(
        ENGLISH_TEXT + "\n", encoding="utf-8"
    )
    # Same normalized text as ar_page, different image bytes.
    _tiny_png(source_a / "dups" / "dup_text.png", (180, 190, 200))
    (source_a / "dups" / "dup_text.txt").write_text(DUP_TEXT + "\n", encoding="utf-8")
    # Exact duplicate of en_note.png (byte-identical copy).
    (source_a / "dups").mkdir(parents=True, exist_ok=True)
    (source_a / "dups" / "dup_image.txt").write_text(
        ENGLISH_TEXT + "\n", encoding="utf-8"
    )
    _tiny_png(source_a / "dups" / "dup_image.png", (220, 230, 200))
    (source_a / "dups" / "dup_image.png").write_bytes(
        (source_a / "english" / "en_note.png").read_bytes()
    )
    # Orphan text with no paired image.
    (source_a / "orphan.txt").write_text(ORPHAN_TEXT + "\n", encoding="utf-8")
    # Junk that the scanner must ignore.
    (source_a / ".DS_Store").write_text("junk", encoding="utf-8")
    (source_a / "__pycache__" / "cached.png").parent.mkdir(parents=True, exist_ok=True)
    _tiny_png(source_a / "__pycache__" / "cached.png", (10, 10, 10))

    # --- source_b: record adapters ---------------------------------------
    (source_b / "images").mkdir(parents=True, exist_ok=True)
    _tiny_png(source_b / "images" / "rec_alpha.png", (250, 220, 210))
    _tiny_png(source_b / "images" / "rec_beta.png", (210, 250, 220))
    records = [
        {"image": "images/rec_alpha.png", "text": ARABIC_TEXT, "id": "alpha"},
        {"image": "images/rec_beta.png", "text": ENGLISH_TEXT, "id": "beta"},
    ]
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    lines.append("{not valid json")  # malformed metadata row
    lines.append(
        json.dumps({"image": "images/missing.png", "text": "نص", "id": "gamma"})
    )
    (source_b / "records.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (source_b / "pairs.csv").write_text(
        "image,text\n" "images/rec_alpha.png," + ENGLISH_TEXT + "\n",
        encoding="utf-8",
    )

    return {"source_a": source_a, "source_b": source_b}

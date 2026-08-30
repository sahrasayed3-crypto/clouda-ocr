"""Run a small local demonstration without an OCR model or external service."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pdfword.docx_export import markdown_to_docx  # noqa: E402
from pdfword.ocr_pipeline import process_pdf  # noqa: E402


def _process(name: str):
    return process_pdf(
        (FIXTURES / name).read_bytes(),
        from_page=1,
        to_page=1,
        progress_bar=None,
        status_placeholder=None,
    )[0][0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "demo")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    digital = _process("digital_text.pdf")
    scanned = _process("scanned.pdf")
    blank = _process("blank.pdf")
    (args.output_dir / "digital_text.docx").write_bytes(markdown_to_docx([digital]))
    payload = {"pages": [asdict(page) for page in (digital, scanned, blank)]}
    (args.output_dir / "page_statuses.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"DOCX: {args.output_dir / 'digital_text.docx'}")
    print(f"JSON: {args.output_dir / 'page_statuses.json'}")
    print(
        "States:",
        ", ".join(page.route_used or "unknown" for page in (digital, scanned, blank)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

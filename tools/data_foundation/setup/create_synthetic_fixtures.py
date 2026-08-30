from __future__ import annotations

from pathlib import Path

FIXTURES = {
    "arabic_body.txt": "عنوان اختباري\n\nهذا نص عربي قصير لاختبار حفظ الحقيقة النصية.\n\n1",
    "mixed_ar_en.txt": "تقرير OCR\nArabic text with English OCR sample.\nالهامش: note 1",
    "two_column.txt": "عمود أول\nسطر عربي قصير\n\nعمود ثان\nسطر آخر",
    "footnote.txt": "المتن الرئيسي يحتوي على إشارة [1].\n\n[1] حاشية قصيرة.",
    "blank.txt": "",
    "near_blank.txt": " ",
}


def main() -> int:
    root = Path("tests/fixtures/synthetic_pages")
    root.mkdir(parents=True, exist_ok=True)
    for name, content in FIXTURES.items():
        (root / name).write_text(content, encoding="utf-8")
    (root / "README.md").write_text(
        "# Synthetic Test Fixtures\n\nTiny test-only files. They are not training data and were not downloaded.\n",
        encoding="utf-8",
    )
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

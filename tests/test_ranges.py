import unittest

from pdfword.ranges import build_output_filename, parse_page_ranges


class TestRanges(unittest.TestCase):
    def test_parse_multi_ranges(self) -> None:
        ranges, pages = parse_page_ranges("21-25, 30-33")
        self.assertEqual(ranges, [(21, 25), (30, 33)])
        self.assertEqual(pages, [21, 22, 23, 24, 25, 30, 31, 32, 33])

    def test_parse_single_pages_and_ranges(self) -> None:
        ranges, pages = parse_page_ranges("21-23, 25, 26")
        self.assertEqual(ranges, [(21, 23), (25, 25), (26, 26)])
        self.assertEqual(pages, [21, 22, 23, 25, 26])

    def test_parse_invalid_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_page_ranges("10-2")

    def test_output_filename_sanitization(self) -> None:
        filename = build_output_filename('book: "fiqh/hadith"', [(1, 5), (8, 10)])
        self.assertEqual(filename, "book fiqh hadith_1-5_8-10.docx")


if __name__ == "__main__":
    unittest.main()

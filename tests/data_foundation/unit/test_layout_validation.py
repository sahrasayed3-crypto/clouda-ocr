from __future__ import annotations

import unittest

from clouda_data.layout.reading_order import reading_order_ids
from clouda_data.layout.regions import Box, Region, RegionKind
from clouda_data.layout.validators import validate_regions


class LayoutValidationTests(unittest.TestCase):
    def test_region_validation_and_order(self) -> None:
        regions = [
            Region("body", RegionKind.BODY, Box(0, 50, 100, 100), 1),
            Region("title", RegionKind.TITLE, Box(0, 0, 100, 40), 0),
        ]
        validate_regions(regions, 200, 200)
        self.assertEqual(reading_order_ids(regions), ["title", "body"])

    def test_invalid_region(self) -> None:
        with self.assertRaises(ValueError):
            validate_regions(
                [Region("bad", RegionKind.BODY, Box(0, 0, 0, 10), 0)], 100, 100
            )


if __name__ == "__main__":
    unittest.main()

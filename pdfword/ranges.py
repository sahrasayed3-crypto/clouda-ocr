import re

ALL_PAGE_MODES = {"All pages", "كل الصفحات"}
RANGE_MODES = {"Range", "نطاق محدد"}
SEPARATE_MODES = {"Separate pages", "صفحات منفصلة"}


def select_pages(
    mode: str,
    total_pages: int,
    *,
    start: int | None = None,
    end: int | None = None,
    separate: str = "",
) -> tuple[list[tuple[int, int]], list[int]]:
    """Build a validated page selection independently from parallelism."""
    if total_pages < 1:
        raise ValueError("Could not determine the PDF page count.")
    if mode in ALL_PAGE_MODES:
        return [(1, total_pages)], list(range(1, total_pages + 1))
    if mode in RANGE_MODES:
        first = int(start or 0)
        last = int(end or 0)
        if first < 1 or last < 1:
            raise ValueError("Page numbers must be positive.")
        if first > last:
            raise ValueError("Range start cannot be after range end.")
        if last > total_pages:
            raise ValueError(f"The PDF contains only {total_pages} pages.")
        return [(first, last)], list(range(first, last + 1))
    if mode in SEPARATE_MODES:
        values = [int(value) for value in re.findall(r"\d+", separate or "")]
        if not values:
            raise ValueError("Enter at least one page number.")
        pages = list(dict.fromkeys(values))
        if any(page < 1 for page in pages):
            raise ValueError("Page numbers must be positive.")
        if any(page > total_pages for page in pages):
            raise ValueError(f"The PDF contains only {total_pages} pages.")
        return [(page, page) for page in pages], pages
    raise ValueError("Unknown page selection mode.")


def parse_page_ranges(ranges_text: str) -> tuple[list[tuple[int, int]], list[int]]:
    text = (ranges_text or "").replace("،", ",").replace("طŒ", ",").strip()
    if not text:
        raise ValueError("Enter at least one range.")

    ranges: list[tuple[int, int]] = []
    ordered_pages: list[int] = []
    seen = set()
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        numbers = [int(value) for value in re.findall(r"\d+", part)]
        if len(numbers) == 1:
            a = b = numbers[0]
        elif len(numbers) >= 2:
            a, b = numbers[0], numbers[1]
        else:
            raise ValueError(f"Invalid range: {part}")
        if a <= 0 or b <= 0 or a > b:
            raise ValueError(f"Invalid range: {part}")
        ranges.append((a, b))
        for page in range(a, b + 1):
            if page not in seen:
                ordered_pages.append(page)
                seen.add(page)
    return ranges, ordered_pages


def build_output_filename(book_name: str, ranges: list[tuple[int, int]]) -> str:
    safe = re.sub(r'[<>:""/\\\\|?*]+', " ", (book_name or "").strip())
    safe = re.sub(r"\s+", " ", safe).strip() or "document"
    ranges_part = "_".join([f"{a}-{b}" for a, b in ranges]) or "all"
    return f"{safe}_{ranges_part}.docx"

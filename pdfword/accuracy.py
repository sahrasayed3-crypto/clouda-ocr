import re
import unicodedata
from typing import Sequence

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
LATIN_RE = re.compile(r"[A-Za-z]")
ARABIC_TOKEN_RE = re.compile(r"[\u0600-\u06FF]+")
CONTROL_RE = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]")
PRIVATE_USE_RE = re.compile(r"[\uE000-\uF8FF]")
COMBINING_RE = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
SYMBOL_RE = re.compile(r"[^\w\s\u0600-\u06FF.,;:!?؟،؛ـ()\[\]{}\"'«»/@+\-=%]")


def _safe_percent(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 2)


def _duplicate_line_ratio(lines: list[str]) -> float:
    normalized = [re.sub(r"\s+", " ", line).strip() for line in lines if line.strip()]
    if len(normalized) < 3:
        return 0.0
    return (len(normalized) - len(set(normalized))) / max(1, len(normalized))


def analyze_text_corruption(
    text: str,
    *,
    estimated_text_quality: float | None = None,
    expected_non_empty: bool = False,
) -> dict:
    """Detect broad text corruption without using a fixed bad-word blacklist."""
    value = normalize_for_accuracy(text or "")
    stripped = value.strip()
    chars = [c for c in stripped if not c.isspace()]
    total_chars = max(1, len(chars))
    arabic_chars = ARABIC_RE.findall(stripped)
    arabic_count = len(arabic_chars)
    words = re.findall(r"\S+", stripped)
    arabic_words = [w for w in words if ARABIC_RE.search(w)]
    lines = stripped.splitlines()

    replacement_count = stripped.count("\ufffd")
    private_use_count = len(PRIVATE_USE_RE.findall(stripped))
    control_count = len(CONTROL_RE.findall(stripped))
    combining_count = len(COMBINING_RE.findall(stripped))
    symbol_count = len(SYMBOL_RE.findall(stripped))
    duplicate_ratio = _duplicate_line_ratio(lines)
    embedded_noise_words = [
        word
        for word in arabic_words
        if re.search(
            r"[\u0600-\u06FF][A-Za-z0-9_@#$%^&*+=<>\\|~]{1,}[\u0600-\u06FF]", word
        )
    ]
    malformed_words = [
        word
        for word in arabic_words
        if len(
            re.findall(
                r"[^\u0600-\u06FF\u064B-\u065F\u0670\u06D6-\u06ED0-9.,:;!?؟،؛()\[\]{}«»\"'\-/]",
                word,
            )
        )
        > max(1, len(word) // 3)
    ]
    isolated_arabic_tokens = [
        w for w in arabic_words if len(ARABIC_TOKEN_RE.sub("", w)) == 0 and len(w) == 1
    ]
    detached_diacritics = len(
        re.findall(r"(?<![\u0621-\u064A])[\u064B-\u065F]", stripped)
    )

    mojibake_chars = sum(1 for c in arabic_chars if c in {"ط", "ظ"})
    mojibake_ratio = mojibake_chars / max(1, arabic_count)
    mojibake_like = (
        arabic_count >= 20
        and mojibake_ratio >= 0.34
        and len(re.findall(r"(?:ط[^\s]{1,3}|ظ[^\s]{1,3})", stripped)) >= 5
    )
    punctuation_noise = len(re.findall(r"[.,;:!?؟،؛]{3,}", stripped))
    repeated_garbage = len(re.findall(r"(.{1,8})\1{4,}", stripped))
    symbol_density = symbol_count / total_chars
    combining_density = combining_count / total_chars
    isolated_ratio = len(isolated_arabic_tokens) / max(1, len(arabic_words))

    critical_flags: list[str] = []
    reasons: list[str] = []
    if expected_non_empty and not stripped:
        critical_flags.append("empty_output_from_non_empty_page")
        reasons.append("No text was extracted from a page expected to contain text.")
    if replacement_count:
        critical_flags.append("replacement_characters")
        reasons.append("Replacement characters indicate Unicode decoding loss.")
    if private_use_count:
        critical_flags.append("private_use_glyphs")
        reasons.append(
            "Private-use glyphs usually indicate broken embedded font mapping."
        )
    if control_count:
        critical_flags.append("control_characters")
        reasons.append("Control characters were found inside extracted text.")
    if embedded_noise_words:
        critical_flags.append("noise_inside_arabic_words")
        reasons.append("Digits or symbols appear inside Arabic word bodies.")
    if mojibake_like:
        critical_flags.append("mojibake_like_arabic")
        reasons.append("Arabic text has a strong mojibake-like glyph pattern.")
    if symbol_density > 0.12:
        critical_flags.append("symbol_dominated_text")
        reasons.append("Symbol density is too high for readable text.")
    if combining_density > 0.22 or detached_diacritics > 5:
        critical_flags.append("abnormal_combining_marks")
        reasons.append("Combining marks are detached or abnormally dense.")
    if repeated_garbage:
        critical_flags.append("repeated_garbage")
        reasons.append("Short text spans are repeated excessively.")
    if duplicate_ratio >= 0.35:
        critical_flags.append("large_duplicate_sections")
        reasons.append("Duplicate line ratio is high.")

    unicode_integrity = (
        100.0
        - min(55.0, replacement_count * 18.0)
        - min(55.0, private_use_count * 14.0)
    )
    unicode_integrity -= min(30.0, control_count * 8.0) + min(
        35.0, combining_density * 120.0
    )
    arabic_integrity = 100.0 - min(45.0, len(embedded_noise_words) * 18.0)
    arabic_integrity -= min(35.0, len(malformed_words) * 8.0)
    arabic_integrity -= min(
        35.0, mojibake_ratio * 80.0 if mojibake_like else mojibake_ratio * 15.0
    )
    arabic_integrity -= min(25.0, isolated_ratio * 80.0)
    structure = (
        100.0 - min(45.0, duplicate_ratio * 100.0) - min(30.0, punctuation_noise * 10.0)
    )
    structure -= min(30.0, repeated_garbage * 12.0)

    readability = (
        (unicode_integrity * 0.35) + (arabic_integrity * 0.45) + (structure * 0.20)
    )
    if estimated_text_quality is not None:
        readability = min(readability, max(0.0, float(estimated_text_quality)) + 8.0)
    coverage = (
        0.0
        if expected_non_empty and not stripped
        else min(100.0, 35.0 + min(65.0, len(stripped) / 4.0))
    )
    hallucination_risk = (
        65.0
        if repeated_garbage or symbol_density > 0.08
        else (20.0 if stripped else 0.0)
    )
    numeric_quality = (
        (readability * 0.45)
        + (unicode_integrity * 0.20)
        + (arabic_integrity * 0.20)
        + (structure * 0.10)
        + (coverage * 0.05)
    )
    if estimated_text_quality is not None:
        numeric_quality = min(float(estimated_text_quality), numeric_quality)
    accept = (
        not critical_flags
        and numeric_quality >= 90.0
        and readability >= 88.0
        and coverage >= 20.0
    )

    samples = embedded_noise_words[:3] + malformed_words[:3]
    if mojibake_like:
        samples.extend(re.findall(r"\S*?[طظ]\S*", stripped)[:3])
    return {
        "is_corrupted": bool(critical_flags),
        "estimated_text_quality": _safe_percent(numeric_quality),
        "readability_score": _safe_percent(readability),
        "unicode_integrity_score": _safe_percent(unicode_integrity),
        "arabic_word_integrity_score": _safe_percent(arabic_integrity),
        "structure_score": _safe_percent(structure),
        "layout_consistency_score": _safe_percent(structure),
        "coverage_score": _safe_percent(coverage),
        "duplication_risk": _safe_percent(duplicate_ratio * 100.0),
        "missing_text_risk": _safe_percent(100.0 - coverage),
        "hallucination_risk": _safe_percent(hallucination_risk),
        "critical_failure_flags": critical_flags,
        "detected_reasons": reasons,
        "suspicious_samples": samples[:8],
        "recommended_route": (
            "accept" if accept else ("ocr_or_cloud" if stripped else "manual_review")
        ),
        "accept_or_reject": "accept" if accept else "reject",
    }


def final_acceptance_decision(
    text: str,
    *,
    estimated_text_quality: float | None,
    threshold: float = 90.0,
    expected_non_empty: bool = False,
) -> dict:
    diagnostics = analyze_text_corruption(
        text,
        estimated_text_quality=estimated_text_quality,
        expected_non_empty=expected_non_empty,
    )
    threshold = max(90.0, float(threshold))
    indeterminate_quality = estimated_text_quality is None
    accepted = (
        not indeterminate_quality
        and diagnostics["estimated_text_quality"] >= threshold
        and diagnostics["accept_or_reject"] == "accept"
        and not diagnostics["critical_failure_flags"]
    )
    if accepted:
        reason = None
    elif indeterminate_quality:
        reason = "indeterminate_quality_score"
    elif diagnostics["critical_failure_flags"]:
        reason = "critical_text_corruption: " + ", ".join(
            diagnostics["critical_failure_flags"]
        )
    else:
        reason = f"estimated_text_quality below {threshold:.2f}%"
    return {
        "accepted": accepted,
        "requires_manual_review": not accepted,
        "estimated_text_quality": diagnostics["estimated_text_quality"],
        "review_reason": reason,
        "diagnostics": diagnostics,
    }


def levenshtein_distance(seq1: Sequence[str], seq2: Sequence[str]) -> int:
    if len(seq1) < len(seq2):
        seq1, seq2 = seq2, seq1
    prev = list(range(len(seq2) + 1))
    for i, a in enumerate(seq1, start=1):
        curr = [i]
        for j, b in enumerate(seq2, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (0 if a == b else 1)
            curr.append(min(ins, delete, replace))
        prev = curr
    return prev[-1]


def normalize_for_accuracy(text: str) -> str:
    """Apply only lossless normalization allowed for acceptance metrics."""
    value = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", value)


def estimate_quality_components(
    text: str,
    *,
    expected_pages: int = 1,
    converted_pages: int = 1,
    base_text_score: float | None = None,
) -> dict:
    """Calculate transparent heuristic scores when no ground truth exists."""
    value = (text or "").strip()
    chars = [c for c in value if not c.isspace()]
    useful = [c for c in chars if c.isalnum() or ARABIC_RE.match(c)]
    useful_ratio = len(useful) / max(1, len(chars))
    replacement_penalty = min(25.0, value.count("\ufffd") * 5.0)
    length_component = min(1.0, len(value) / max(40.0, converted_pages * 250.0))
    heuristic_text = max(
        0.0,
        min(
            100.0,
            (useful_ratio * 70.0) + (length_component * 30.0) - replacement_penalty,
        ),
    )
    text_score = (
        heuristic_text
        if base_text_score is None
        else max(0.0, min(100.0, float(base_text_score)))
    )
    completeness = max(
        0.0, min(100.0, (converted_pages / max(1, expected_pages)) * 100.0)
    )
    has_known_script = bool(ARABIC_RE.search(value) or LATIN_RE.search(value))
    direction = 95.0 if value and has_known_script else (60.0 if value else 0.0)
    paragraph_count = len([p for p in re.split(r"\n\s*\n", value) if p.strip()])
    layout = (
        min(
            100.0,
            55.0 + min(30.0, paragraph_count * 5.0) + (15.0 if "\n" in value else 0.0),
        )
        if value
        else 0.0
    )
    final = (
        (text_score * 0.70)
        + (completeness * 0.15)
        + (direction * 0.10)
        + (layout * 0.05)
    )
    return {
        "text_quality": round(text_score, 2),
        "completeness": round(completeness, 2),
        "direction_quality": round(direction, 2),
        "layout_quality": round(layout, 2),
        "final_quality": round(max(0.0, min(100.0, final)), 2),
        "label": "درجة جودة تقديرية",
    }


def compute_accuracy_metrics(reference_text: str, extracted_text: str) -> dict:
    ref = normalize_for_accuracy(reference_text)
    hyp = normalize_for_accuracy(extracted_text)

    ref_words = ref.split()
    hyp_words = hyp.split()
    ref_chars = list(ref)
    hyp_chars = list(hyp)

    word_errors = levenshtein_distance(ref_words, hyp_words) if ref_words else 0
    char_errors = levenshtein_distance(ref_chars, hyp_chars) if ref_chars else 0

    wer = (word_errors / max(1, len(ref_words))) * 100.0
    cer = (char_errors / max(1, len(ref_chars))) * 100.0

    return {
        "wer": wer,
        "cer": cer,
        "character_accuracy": max(0.0, 100.0 - cer),
        "word_accuracy": max(0.0, 100.0 - wer),
        "char_accuracy": max(0.0, 100.0 - cer),
        "ref_words": len(ref_words),
        "hyp_words": len(hyp_words),
        "reference_text": ref,
        "extracted_text": hyp,
        "word_errors": word_errors,
        "char_errors": char_errors,
    }


def evaluate_pages_against_references(
    page_results: Sequence, references: Sequence[str]
) -> dict:
    """Return per-page and document-level real accuracy metrics.

    This is intentionally separate from heuristic quality scores and OCR engine
    confidence. It requires reference text and should not be used when ground
    truth is unavailable.
    """
    per_page = []
    for index, page in enumerate(page_results):
        expected = references[index] if index < len(references) else ""
        extracted = getattr(page, "markdown", str(page))
        metrics = compute_accuracy_metrics(expected, extracted)
        per_page.append(
            {
                "page_no": getattr(page, "page_no", index + 1),
                "expected_text": metrics["reference_text"],
                "extracted_text": metrics["extracted_text"],
                "cer": metrics["cer"],
                "wer": metrics["wer"],
                "character_accuracy": metrics["char_accuracy"],
                "word_accuracy": metrics["word_accuracy"],
                "heuristic_quality_score": getattr(page, "text_quality_score", None),
                "ocr_engine_confidence": None,
                "route_used": getattr(page, "route_used", None),
                "engine_used": getattr(page, "model_used", None),
                "model_used": getattr(page, "model_used", None),
                "requires_manual_review": bool(
                    getattr(page, "requires_manual_review", False)
                ),
            }
        )

    document_expected = "\n\n".join(references)
    document_extracted = "\n\n".join(
        getattr(page, "markdown", str(page)) for page in page_results
    )
    document = compute_accuracy_metrics(document_expected, document_extracted)
    valid_pages = [row for row in per_page if row["expected_text"].strip()]
    aggregate = {
        "pages": per_page,
        "document": {
            "expected_text": document["reference_text"],
            "extracted_text": document["extracted_text"],
            "cer": document["cer"],
            "wer": document["wer"],
            "character_accuracy": document["char_accuracy"],
            "word_accuracy": document["word_accuracy"],
        },
        "aggregate": {
            "page_count": len(per_page),
            "referenced_page_count": len(valid_pages),
            "avg_cer": (
                (sum(row["cer"] for row in valid_pages) / len(valid_pages))
                if valid_pages
                else None
            ),
            "avg_wer": (
                (sum(row["wer"] for row in valid_pages) / len(valid_pages))
                if valid_pages
                else None
            ),
            "avg_character_accuracy": (
                (
                    sum(row["character_accuracy"] for row in valid_pages)
                    / len(valid_pages)
                )
                if valid_pages
                else None
            ),
            "avg_word_accuracy": (
                (sum(row["word_accuracy"] for row in valid_pages) / len(valid_pages))
                if valid_pages
                else None
            ),
        },
    }
    return aggregate

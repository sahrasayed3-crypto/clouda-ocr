from __future__ import annotations

import io
import json
from types import SimpleNamespace

from PIL import Image

from pdfword.ocr_self_review import (
    OCRIssueCode,
    OCRReviewResult,
    OCRReviewVerdict,
    OCRPageState,
    ReReadResult,
    ReReadBudget,
    RenderedPageContext,
    ReviewRegion,
    make_rendered_page_context,
    crop_review_region,
    reconcile_ocr_results,
    review_first_pass,
    validate_and_bound_regions,
)
from pdfword.engines import OCRBox, OCRResult, OCR_STATUS_SUCCEEDED


def _png_bytes(width: int = 40, height: int = 60) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(output, format="PNG")
    return output.getvalue()


def test_render_context_binds_exact_png_dimensions_and_identity() -> None:
    first = make_rendered_page_context(3, _png_bytes())
    second = make_rendered_page_context(3, _png_bytes())

    assert first.page_no == 3
    assert (first.width_px, first.height_px) == (40, 60)
    assert first.coordinate_space == "image_pixels"
    assert first.render_identity == second.render_identity


def test_safe_diagnostics_omit_internal_text_hash_and_geometry() -> None:
    context = make_rendered_page_context(1, _png_bytes())
    region = ReviewRegion(
        region_id="region-1",
        page_no=1,
        bbox_px=(1, 2, 20, 30),
        render_identity=context.render_identity,
        image_width_px=40,
        image_height_px=60,
        reason_codes=(OCRIssueCode.UNICODE_CORRUPTION,),
        priority="high",
    )
    result = OCRReviewResult(
        verdict=OCRReviewVerdict.REREAD_REQUIRED,
        issue_codes=(OCRIssueCode.UNICODE_CORRUPTION,),
        suspicious_regions=(region,),
        safe_diagnostics=(("first_pass_characters", 11),),
        selective_reread_justified=True,
        manual_review_required=False,
    )

    serialized = json.dumps(result.to_diagnostics())
    assert "secret text" not in serialized
    assert context.render_identity not in serialized
    assert "bbox" not in serialized


def _analysis(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "arabic_integrity_risk": False,
        "suspicious_fragmentation": False,
        "pathological_arabic_spacing_count": 0,
        "unicode_corruption_count": 0,
        "reading_order_risk": False,
        "text_coverage_ratio": None,
        "visible_image_operations": None,
        "warnings": (),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_clean_first_pass_without_confidence_is_accepted() -> None:
    review = review_first_pass(
        OCRResult(
            engine_name="fake",
            status=OCR_STATUS_SUCCEEDED,
            text="A readable first-pass OCR result with enough words to be useful.",
            confidence=None,
        ),
        _analysis(),
        make_rendered_page_context(1, _png_bytes()),
    )

    assert review.verdict is OCRReviewVerdict.ACCEPTED
    assert not review.issue_codes


def test_unicode_corruption_is_categorical_review_reason() -> None:
    review = review_first_pass(
        OCRResult(
            engine_name="fake",
            status=OCR_STATUS_SUCCEEDED,
            text="broken \ufffd output",
        ),
        _analysis(),
        make_rendered_page_context(1, _png_bytes()),
    )

    assert OCRIssueCode.UNICODE_CORRUPTION in review.issue_codes
    assert review.verdict is OCRReviewVerdict.REVIEW_REQUIRED


def test_verified_current_render_box_justifies_selective_reread() -> None:
    context = make_rendered_page_context(1, _png_bytes())
    review = review_first_pass(
        OCRResult(
            engine_name="fake",
            status=OCR_STATUS_SUCCEEDED,
            text="broken \ufffd output",
            boxes=(
                OCRBox(
                    bbox=(1.0, 2.0, 30.0, 40.0),
                    metadata={
                        "coordinate_space": "image_pixels",
                        "render_identity": context.render_identity,
                        "image_width_px": 40,
                        "image_height_px": 60,
                    },
                ),
            ),
        ),
        _analysis(),
        context,
    )

    assert review.verdict is OCRReviewVerdict.REREAD_REQUIRED
    assert review.selective_reread_justified
    assert len(review.suspicious_regions) == 1


def test_stale_region_is_rejected_without_creating_a_crop() -> None:
    context = make_rendered_page_context(1, _png_bytes())
    stale = ReviewRegion(
        region_id="stale",
        page_no=1,
        bbox_px=(1, 1, 20, 20),
        render_identity="other-render",
        image_width_px=40,
        image_height_px=60,
        reason_codes=(OCRIssueCode.UNICODE_CORRUPTION,),
        priority="high",
    )

    assert validate_and_bound_regions((stale,), context, ReReadBudget()) == ()


def test_overlapping_regions_are_deduplicated_and_crop_is_bounded() -> None:
    image = _png_bytes()
    context = make_rendered_page_context(1, image)
    regions = tuple(
        ReviewRegion(
            region_id=name,
            page_no=1,
            bbox_px=bbox,
            render_identity=context.render_identity,
            image_width_px=40,
            image_height_px=60,
            reason_codes=(OCRIssueCode.UNICODE_CORRUPTION,),
            priority="high",
        )
        for name, bbox in (("one", (1, 1, 30, 50)), ("two", (2, 2, 31, 51)))
    )

    bounded = validate_and_bound_regions(regions, context, ReReadBudget())

    assert [region.region_id for region in bounded] == ["one"]
    with Image.open(io.BytesIO(crop_review_region(image, bounded[0], context))) as crop:
        assert crop.size == (29, 49)


def test_reread_replaces_only_the_verified_box_text_and_retains_provenance() -> None:
    context = make_rendered_page_context(1, _png_bytes())
    region = ReviewRegion(
        region_id="r1",
        page_no=1,
        bbox_px=(1, 1, 30, 40),
        render_identity=context.render_identity,
        image_width_px=40,
        image_height_px=60,
        reason_codes=(OCRIssueCode.UNICODE_CORRUPTION,),
        priority="high",
        source_text="bad \ufffd",
    )
    review = OCRReviewResult(
        verdict=OCRReviewVerdict.REREAD_REQUIRED,
        issue_codes=(OCRIssueCode.UNICODE_CORRUPTION,),
        suspicious_regions=(region,),
        safe_diagnostics=(),
        selective_reread_justified=True,
        manual_review_required=False,
    )

    result = reconcile_ocr_results(
        "before bad \ufffd after",
        review,
        (ReReadResult("r1", "fake", True, "fixed"),),
    )

    assert result.state is OCRPageState.ACCEPTED_AFTER_SELECTIVE_REREAD
    assert result.text == "before fixed after"
    assert result.provenance == ("selective_reread:r1",)


def test_reread_without_a_verified_text_boundary_requires_review() -> None:
    review = OCRReviewResult(
        verdict=OCRReviewVerdict.REREAD_REQUIRED,
        issue_codes=(OCRIssueCode.UNICODE_CORRUPTION,),
        suspicious_regions=(
            ReviewRegion(
                "r1", 1, (1, 1, 30, 40), "render", 40, 60,
                (OCRIssueCode.UNICODE_CORRUPTION,), "high",
            ),
        ),
        safe_diagnostics=(),
        selective_reread_justified=True,
        manual_review_required=False,
    )

    result = reconcile_ocr_results("bad \ufffd", review, (ReReadResult("r1", "fake", True, "fixed"),))

    assert result.state is OCRPageState.REVIEW_REQUIRED
    assert OCRIssueCode.REREAD_DISAGREEMENT in result.reason_codes

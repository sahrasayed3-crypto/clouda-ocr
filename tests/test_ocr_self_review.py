from __future__ import annotations

import io
import json

from PIL import Image

from pdfword.ocr_self_review import (
    OCRIssueCode,
    OCRReviewResult,
    OCRReviewVerdict,
    RenderedPageContext,
    ReviewRegion,
    make_rendered_page_context,
)


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

import io
import re

from pypdf import PdfReader

from .constants import MODEL_FAST
from .ocr_pipeline import encode_image_to_base64, render_pdf_page_to_png_bytes
from .provider_client import get_provider_client


def extract_pdf_reference_text(pdf_bytes: bytes, page_numbers: list[int]) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        parts = []
        for page_no in page_numbers:
            idx = page_no - 1
            txt = reader.pages[idx].extract_text() or ""
            if txt.strip():
                parts.append(txt)
        return "\n".join(parts).strip()
    except Exception:
        return ""


def resolve_reference_text(
    pdf_bytes: bytes,
    page_numbers: list[int],
    manual_file_bytes: bytes | None,
    manual_text: str,
) -> tuple[str, str]:
    # source label: manual_file | manual_text | auto_pdf | none
    if manual_file_bytes:
        text = manual_file_bytes.decode("utf-8", errors="ignore").strip()
        if text:
            return text, "manual_file"

    if manual_text.strip():
        return manual_text.strip(), "manual_text"

    auto_ref = extract_pdf_reference_text(pdf_bytes, page_numbers)
    if auto_ref:
        return auto_ref, "auto_pdf"

    return "", "none"


def _parse_score(raw: str) -> float | None:
    nums = re.findall(r"\d+", raw or "")
    if not nums:
        return None
    val = float(nums[0])
    return max(0.0, min(100.0, val))


def estimate_ai_fidelity_score(
    api_key: str,
    pdf_bytes: bytes,
    page_results: list,
    judge_model: str = MODEL_FAST,
    provider_name: str = "openrouter",
) -> float | None:
    if not page_results:
        return None

    judge_prompt = (
        "You are a strict OCR fidelity judge. "
        "Given a page image and text-only OCR output for that same page, score transcription faithfulness from 0 to 100. "
        "Criteria: exact wording, numbers, punctuation, and text-only reading order. "
        "Do not reward Markdown table reconstruction or image descriptions. "
        "Output only one integer number, nothing else."
    )
    user_template = (
        "Evaluate this text-only OCR output against the page image and output only one integer 0-100:\n\n"
        "{markdown}"
    )

    if len(page_results) <= 3:
        sample = page_results
    else:
        sample = [
            page_results[0],
            page_results[len(page_results) // 2],
            page_results[-1],
        ]
    scores: list[float] = []
    provider_client = get_provider_client(provider_name, primary_api_key=api_key)

    for item in sample:
        try:
            image_bytes = render_pdf_page_to_png_bytes(
                pdf_bytes=pdf_bytes, page_no=item.page_no, dpi=220
            )
            image_b64 = encode_image_to_base64(image_bytes)
            raw = provider_client.chat_with_image(
                model=judge_model,
                system_prompt=judge_prompt,
                user_text=user_template.format(markdown=item.markdown[:6000]),
                image_b64=image_b64,
                max_tokens=8,
                temperature=0.0,
            )
            score = _parse_score(raw)
            if score is not None:
                scores.append(score)
        except Exception:
            continue

    if not scores:
        return None
    return sum(scores) / len(scores)

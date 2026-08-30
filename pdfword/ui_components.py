import html
from pathlib import Path

STATUS_META = {
    "pending": {
        "label": "بانتظار المعالجة",
        "tone": "warning",
        "description": "تم استلام المستند وسينتقل إلى المعالجة عند توفر عامل.",
    },
    "processing": {
        "label": "قيد المعالجة",
        "tone": "info",
        "description": "المستند قيد العمل. يمكنك مغادرة الصفحة والعودة لاحقًا.",
    },
    "finalizing": {
        "label": "تجهيز النتيجة",
        "tone": "info",
        "description": "اكتملت المعالجة الأساسية ويجري تجهيز ملف Word.",
    },
    "completed": {
        "label": "جاهز للتنزيل",
        "tone": "success",
        "description": "اكتمل التحويل ويمكن تنزيل ملف Word الآن.",
    },
    "manual_review": {
        "label": "تحتاج مراجعة",
        "tone": "warning",
        "description": (
            "لم تتمكن المعالجة الآلية من اعتماد نتيجة نهائية لهذا المستند. "
            "حدّث قائمة المستندات لاحقًا أو احذف المستند إن لم تعد تحتاجه."
        ),
    },
    "failed": {
        "label": "تعذر التحويل",
        "tone": "danger",
        "description": "لم يكتمل التحويل. يمكنك إعادة المحاولة.",
    },
    "cancelled": {
        "label": "أُلغي التحويل",
        "tone": "neutral",
        "description": "أُوقفت معالجة هذا المستند بناءً على طلبك.",
    },
}

STATUS_LABELS = {key: value["label"] for key, value in STATUS_META.items()}


def status_meta(status: str) -> dict[str, str]:
    """Return the canonical, user-facing presentation for a backend status."""
    return STATUS_META.get(
        status,
        {
            "label": "حالة غير معروفة",
            "tone": "neutral",
            "description": "تعذر تحديد حالة المستند. حدّث الصفحة أو حاول لاحقًا.",
        },
    ).copy()


def document_state_alert(status: str) -> tuple[str, str] | None:
    """Keep the signature as the single message for every known document state."""
    if status in STATUS_META:
        return None
    return "info", status_meta(status)["description"]


def document_capabilities(status: str) -> dict[str, bool]:
    """Infer only actions supported by the public document API."""
    known = status in STATUS_META
    return {
        "download": status == "completed",
        "retry": status in {"failed", "cancelled"},
        "cancel": status in {"pending", "processing"},
        "delete": known
        and status in {"completed", "failed", "cancelled", "manual_review"},
    }


def prepared_download_is_available(
    status: str, prepared_download: object, job_id: str
) -> bool:
    """Prevent a cached result button from surviving a later status change."""
    return bool(
        document_capabilities(status)["download"]
        and isinstance(prepared_download, dict)
        and prepared_download.get("job_id") == job_id
        and prepared_download.get("data")
    )


def admin_user_rows(users: list[dict]) -> list[dict[str, str]]:
    """Expose a concise Arabic admin table without incidental backend fields."""
    role_labels = {"admin": "مسؤول", "user": "مستخدم"}
    status_labels = {
        "active": "نشط",
        "disabled": "معطّل",
        "deletion_pending": "بانتظار الحذف",
        "deleted": "محذوف",
    }
    return [
        {
            "معرّف المستخدم": str(user.get("user_id") or ""),
            "البريد الإلكتروني": str(user.get("normalized_email") or ""),
            "الاسم": str(user.get("display_name") or ""),
            "الدور": role_labels.get(str(user.get("role") or ""), "مستخدم"),
            "الحالة": status_labels.get(str(user.get("status") or ""), "غير معروفة"),
            "البريد مفعّل": "نعم" if user.get("email_verified") else "لا",
        }
        for user in users
    ]


def user_error_message(status_code: int | None, context: str = "action") -> str:
    """Map transport outcomes to safe Arabic guidance without raw backend detail."""
    messages = {
        400: "تعذر قبول الطلب. راجع البيانات وحاول مرة أخرى.",
        401: "انتهت جلسة الدخول. سجّل الدخول مرة أخرى.",
        403: "ليست لديك صلاحية لتنفيذ هذا الإجراء.",
        404: "لم يعد العنصر المطلوب متاحًا.",
        409: "تغيّرت حالة المستند ولا يمكن تنفيذ الإجراء الآن. حدّث الصفحة.",
        413: "يتجاوز الملف حد الحجم أو عدد الصفحات المسموح.",
        429: "وصلت إلى حد الاستخدام الحالي. حاول لاحقًا.",
    }
    if status_code in messages:
        return messages[status_code]
    if status_code is None and context == "upload_uncertain":
        return (
            "لم تصل استجابة مؤكدة بعد الإرسال. راجع قائمة المستندات قبل "
            "إعادة المحاولة لتجنب تكرار الملف."
        )
    if status_code is None:
        return "تعذر الاتصال بالخدمة. تحقق من الاتصال وحاول مرة أخرى."
    if context == "documents":
        return "تعذر تحميل المستندات الآن. حاول تحديث الصفحة."
    if context == "upload":
        return "تعذر إرسال الملف للتحويل. حاول مرة أخرى."
    return "تعذر تنفيذ الإجراء. حاول مرة أخرى."


def guest_job_state(
    payload: dict,
    *,
    filename: str,
    fallback_pages: int,
) -> dict:
    """Keep the guest result token with the canonical job state for download."""
    job_id = str(payload.get("job_id") or "")
    if not job_id:
        raise ValueError("Guest response is missing a job identifier.")
    return {
        "job_id": job_id,
        "status": str(payload.get("status") or "pending"),
        "page_count": int(payload.get("page_count") or fallback_pages),
        "result_token": str(payload.get("result_token") or ""),
        "filename": filename,
    }


STATUS_LABEL_TRANSLATIONS = {
    "System": "النظام",
    "Worker": "العامل",
    "Direct PDF text": "استخراج النص المباشر",
    "Future OCR": "OCR القادم",
}

STATUS_STATE_TRANSLATIONS = {
    "ready": "جاهز",
    "busy": "مشغول",
    "offline": "غير متصل",
    "available": "متاح",
    "connected": "متصل",
    "unavailable": "غير متاح",
}


def load_styles(path: str | Path = "assets/styles.css") -> str:
    p = Path(path)
    return f"<style>{p.read_text(encoding='utf-8')}</style>" if p.is_file() else ""


def status_badge(label: str, state: str) -> str:
    safe_label = html.escape(label)
    safe_state = html.escape(state)
    display_label = html.escape(STATUS_LABEL_TRANSLATIONS.get(label, label))
    display_state = html.escape(STATUS_STATE_TRANSLATIONS.get(state, state))
    return (
        f'<span class="status-badge status-{html.escape(state)}">'
        f'<span class="status-dot" aria-hidden="true"></span>'
        f'<span>{display_label}<em class="sr-only">{safe_label}</em></span>'
        f'<strong>{display_state}<em class="sr-only">{safe_state}</em></strong>'
        "</span>"
    )


def status_strip(
    status: dict, *, direct_text_ready: bool = True, future_ocr_ready: bool = False
) -> str:
    worker_ready = status.get("worker_state") in {"ready", "busy"}
    badges = [
        status_badge("System", "ready" if status.get("server") else "offline"),
        status_badge(
            "Worker",
            status.get("worker_state", "offline") if worker_ready else "offline",
        ),
        status_badge("Direct PDF text", "ready" if direct_text_ready else "offline"),
        status_badge("Future OCR", "ready" if future_ocr_ready else "offline"),
    ]
    return (
        '<section class="status-center" aria-label="System status">'
        '<div class="status-center-title">مركز الحالة'
        '<em class="sr-only">Status Center</em></div>'
        '<div class="status-strip">' + "".join(badges) + "</div></section>"
    )


def page_header(title: str, subtitle: str = "") -> str:
    return (
        '<section class="page-header page-heading">'
        f"<h1>{html.escape(title)}</h1>"
        f"<p>{html.escape(subtitle)}</p>"
        "</section>"
    )


def section_title(title: str, description: str = "", step: int | None = None) -> str:
    step_html = f'<span class="section-step">{step}</span>' if step is not None else ""
    return (
        '<div class="section-title section-heading">'
        f"{step_html}<div><h2>{html.escape(title)}</h2>"
        f"<p>{html.escape(description)}</p></div></div>"
    )


def document_transformation_signature(
    *, status: str = "idle", filename: str = "", compact: bool = False
) -> str:
    """Render Clouda's honest PDF-to-DOCX visual signature.

    The component only visualizes a known backend state, or ``demo`` when used as
    an explicitly explanatory product illustration. It never invents progress,
    extraction values, or intermediate processing stages.
    """
    state_classes = {
        "idle": "transform-state-idle",
        "pending": "transform-state-active",
        "processing": "transform-state-active",
        "finalizing": "transform-state-active",
        "completed": "transform-state-completed",
        "failed": "transform-state-attention",
        "manual_review": "transform-state-review",
        "cancelled": "transform-state-neutral",
        "demo": "transform-state-demo",
    }
    safe_status = status if status in state_classes else "idle"
    explanatory = safe_status == "demo"
    active = safe_status in {"pending", "processing", "finalizing"}
    state_class = state_classes[safe_status]
    compact_class = " signature-compact" if compact else ""
    if explanatory:
        label = "من PDF إلى Word"
        description = "تمثيل بصري لمسار التحويل المدعوم، وليس حالة ملف مرفوع."
    elif safe_status == "idle":
        label = "مسار التحويل"
        description = "PDF بنص قابل للتحديد إلى ملف Word قابل للتحرير."
    else:
        presentation = status_meta(safe_status)
        label = presentation["label"]
        description = presentation["description"]
    if active and "يمكنك مغادرة الصفحة" not in description:
        description = f"{description} يمكنك مغادرة الصفحة والعودة لاحقًا."

    if explanatory:
        accessibility = (
            ' role="group" aria-label="عرض توضيحي لمسار التحويل من PDF إلى Word"'
        )
    elif safe_status == "idle":
        accessibility = ' role="group" aria-label="مسار التحويل من PDF إلى Word"'
    else:
        accessibility = ' role="status" aria-live="polite"'
    explanatory_attr = ' data-explanatory="true"' if explanatory else ""
    filename_html = (
        f'<span class="signature-filename" dir="auto">{html.escape(filename)}</span>'
        if filename
        else ""
    )
    return (
        f'<section class="document-signature {state_class}{compact_class}" '
        f'data-status="{safe_status}"{explanatory_attr}{accessibility}>'
        '<div class="signature-copy">'
        f"<strong>{html.escape(label)}</strong><p>{html.escape(description)}</p>"
        "</div>"
        '<div class="signature-canvas" aria-hidden="true">'
        '<div class="signature-document document-source">'
        '<span class="signature-format">PDF</span>'
        '<div class="signature-sheet"><i></i><i></i><i></i><i></i></div>'
        f"{filename_html}</div>"
        '<div class="signature-bridge">'
        '<span class="recognition-sweep"></span>'
        '<div class="structure-lines"><i></i><i></i><i></i></div>'
        '<span class="signature-arrow">←</span>'
        "</div>"
        '<div class="signature-document document-target">'
        '<span class="signature-format">DOCX</span>'
        '<div class="signature-sheet"><i></i><i></i><i></i><i></i></div>'
        '<span class="signature-ready-mark">✓</span>'
        "</div></div></section>"
    )


def document_summary_strip(*, active: int, completed: int, attention: int) -> str:
    """Return a compact document-first overview instead of dashboard KPI cards."""
    values = (
        (max(0, active), "قيد العمل", "summary-active"),
        (max(0, completed), "جاهزة للتنزيل", "summary-completed"),
        (max(0, attention), "تحتاج انتباهك", "summary-attention"),
    )
    items = "".join(
        f'<span class="{tone}">{count} {label}</span>' for count, label, tone in values
    )
    return (
        '<section class="document-summary-strip" aria-label="ملخص المستندات">'
        '<span class="summary-label">الآن في مساحة مستنداتك</span>'
        f"<div>{items}</div></section>"
    )


def file_summary(filename: str, size_bytes: int, page_count: int | None = None) -> str:
    size_mb = size_bytes / (1024 * 1024)
    pages = "عدد الصفحات غير معروف" if page_count is None else f"{page_count} صفحة"
    return (
        '<div class="file-summary">'
        '<div class="file-icon">PDF</div>'
        '<div class="file-meta">'
        f'<strong dir="auto">{html.escape(filename)}</strong>'
        '<div class="file-facts">'
        f"<span>{size_mb:.2f} MB</span><span>{html.escape(pages)}</span>"
        "</div></div></div>"
    )


def processing_panel(
    *,
    stage: str,
    progress: int | None,
    completed_pages: int,
    total_pages: int,
    elapsed: str,
    last_update: str = "",
    current_page: int | None = None,
) -> str:
    progress_text = "قيد العمل" if progress is None else f"{progress}%"
    width = "100%" if progress is None else f"{max(0, min(100, progress))}%"
    current = (
        "" if current_page is None else f"<span>الصفحة الحالية: {current_page}</span>"
    )
    progress_track = (
        '<div class="progress-track progress-indeterminate"><span></span></div>'
        if progress is None
        else f'<div class="progress-track"><span style="width:{width}"></span></div>'
    )
    return (
        '<section class="processing-panel">'
        '<div class="processing-head">'
        f"<div><small>المرحلة الحالية</small><strong>{html.escape(stage)}</strong></div>"
        f"<b>{html.escape(progress_text)}</b></div>"
        f"{progress_track}"
        '<div class="processing-grid">'
        f"<span>الصفحات: {completed_pages}/{total_pages}</span>"
        f"{current}"
        f"<span>الزمن المنقضي: {html.escape(elapsed)}</span>"
        f"<span>آخر تحديث: {html.escape(last_update)}</span>"
        "</div></section>"
    )


def progress_steps(active_index: int = 0) -> str:
    labels = [
        "تحليل الملف",
        "تجهيز الصفحات",
        "استخراج النص",
        "فحص OCR القادم",
        "تنسيق Word",
        "إنشاء DOCX",
        "مراجعة نهائية",
    ]
    items = []
    for index, label in enumerate(labels):
        state = (
            "step-done"
            if index < active_index
            else "step-active" if index == active_index else "step-todo"
        )
        items.append(
            f'<li class="progress-step {state}">'
            f"<span>{index + 1}</span>{html.escape(label)}</li>"
        )
    return '<ol class="progress-steps">' + "".join(items) + "</ol>"


def empty_state(title: str, description: str) -> str:
    return (
        '<div class="empty-state">'
        f"<strong>{html.escape(title)}</strong>"
        f"<p>{html.escape(description)}</p>"
        "</div>"
    )


def document_card(doc: dict) -> str:
    status = doc.get("status", "pending")
    presentation = status_meta(status)
    status_label = presentation["label"]
    filename = doc.get("original_pdf_name", "ملف غير معروف")
    page_count = doc.get("page_count")
    created = doc.get("created_at", "")
    doc_id = doc.get("job_id", doc.get("id", ""))
    if page_count == 1:
        pages_text = "صفحة واحدة"
    elif page_count == 2:
        pages_text = "صفحتان"
    elif isinstance(page_count, int) and 3 <= page_count <= 10:
        pages_text = f"{page_count} صفحات"
    elif page_count:
        pages_text = f"{page_count} صفحة"
    else:
        pages_text = ""
    status_class = f"status-tone-{presentation['tone']}"
    created_display = created[:16].replace("T", " ") if created else ""
    return (
        f'<div class="doc-card" data-doc-id="{html.escape(doc_id)}">'
        f'<div class="doc-card-icon">PDF</div>'
        f'<div class="doc-card-body">'
        f'<div class="doc-card-name" dir="auto">{html.escape(filename)}</div>'
        f'<div class="doc-card-meta">'
        f"{html.escape(pages_text)}"
        f'{"<span>•</span>" if pages_text and created_display else ""}'
        f"{html.escape(created_display)}"
        f"</div></div>"
        f'<div class="doc-card-status {html.escape(status_class)}">'
        f"{html.escape(status_label)}</div>"
        f"</div>"
    )


def document_list(documents: list[dict]) -> str:
    if not documents:
        return empty_state("لا توجد تحويلات", "لم يتم إجراء أي تحويلات بعد.")
    cards = "".join(document_card(doc) for doc in documents)
    return f'<div class="doc-list">{cards}</div>'


def ocr_unavailable_notice() -> str:
    return (
        '<div class="ocr-notice">'
        '<div class="ocr-notice-icon">⏳</div>'
        "<div>"
        '<strong class="ocr-notice-title">'
        "خدمة التعرف الضوئي غير متاحة حاليًا"
        "</strong>"
        '<p class="ocr-notice-text">'
        "تم استلام الملف وتجهيزه للمعالجة، لكن خدمة التعرف الضوئي على النصوص "
        "غير متاحة حاليًا. سيتم معالجة الملف تلقائيًا عند توفر الخدمة."
        "</p></div></div>"
    )


def auth_required_notice() -> str:
    return (
        '<div class="auth-notice">'
        '<strong class="auth-notice-title">'
        "تسجيل الدخول مطلوب"
        "</strong>"
        '<p class="auth-notice-text">'
        "سجّل دخولك لإرسال ملفات PDF للتحويل وعرض سجل التحويلات."
        "</p></div>"
    )


def landing_hero() -> str:
    return (
        '<div class="clouda-hero">'
        '<div class="clouda-hero-content">'
        "<h1>&#1581;&#1608;&#1617;&#1604; &#1605;&#1587;&#1578;&#1606;&#1583;&#1575;&#1578;&#1603;"
        " &#1575;&#1604;&#1593;&#1585;&#1576;&#1610;&#1577; &#1573;&#1604;&#1609;"
        " <em>&#1605;&#1604;&#1601;&#1575;&#1578; &#1602;&#1575;&#1576;&#1604;&#1577;"
        " &#1604;&#1604;&#1578;&#1581;&#1585;&#1610;&#1585;</em></h1>"
        "<p>&#1575;&#1585;&#1601;&#1593; &#1605;&#1604;&#1601; PDF"
        " &#1608;&#1575;&#1581;&#1589;&#1604; &#1593;&#1604;&#1609;"
        " &#1605;&#1587;&#1578;&#1606;&#1583; Word"
        " &#1605;&#1606;&#1587;&#1617;&#1602; &#1576;&#1583;&#1602;&#1577;"
        " &#1593;&#1575;&#1604;&#1610;&#1577;&#8202;&#8212;&#8202;"
        "&#1575;&#1604;&#1606;&#1589; &#1575;&#1604;&#1593;&#1585;&#1576;&#1610;"
        " &#1608;&#1575;&#1604;&#1578;&#1606;&#1587;&#1610;&#1602;"
        " &#1605;&#1581;&#1601;&#1608;&#1592;&#1575;&#1606;.</p>"
        '<div class="clouda-hero-trust">'
        "<span>"
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">'
        '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>'
        " &#1582;&#1589;&#1608;&#1589;&#1610;&#1577; &#1578;&#1575;&#1605;&#1577;</span>"
        "<span>"
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">'
        '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>'
        " &#1578;&#1581;&#1608;&#1610;&#1604; &#1601;&#1608;&#1585;&#1610;</span>"
        "</div></div>"
        '<div class="doc-transform-panel">'
        '<div class="scan-line"></div>'
        '<div class="doc-transform-header">'
        '<span class="doc-transform-label">DOCUMENT TRANSFORM</span>'
        '<span class="doc-transform-badge">&#1580;&#1575;&#1607;&#1586;</span>'
        "</div>"
        '<div class="doc-transform-body">'
        '<div class="doc-page">'
        '<div class="doc-page-title">PDF</div>'
        '<div class="doc-line" style="width:85%"></div>'
        '<div class="doc-line" style="width:70%"></div>'
        '<div class="doc-line" style="width:90%"></div>'
        '<div class="doc-line" style="width:55%"></div>'
        '<div class="doc-line" style="width:78%"></div>'
        "</div>"
        '<div class="doc-transform-arrow">'
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">'
        '<path d="M5 12h14M12 5l7 7-7 7"/></svg>'
        "</div>"
        '<div class="doc-page doc-output">'
        '<div class="doc-page-title">DOCX</div>'
        '<div class="doc-line doc-line-ar" style="--w:85%;--i:0;width:85%"></div>'
        '<div class="doc-line doc-line-ar" style="--w:70%;--i:1;width:70%"></div>'
        '<div class="doc-line doc-line-ar" style="--w:90%;--i:2;width:90%"></div>'
        '<div class="doc-line doc-line-ar" style="--w:55%;--i:3;width:55%"></div>'
        '<div class="doc-line doc-line-ar" style="--w:78%;--i:4;width:78%"></div>'
        "</div>"
        "</div></div></div>"
    )


def how_it_works() -> str:
    icons = [
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>',
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
    ]
    steps = [
        (
            "&#1575;&#1585;&#1601;&#1593; &#1575;&#1604;&#1605;&#1604;&#1601;",
            "&#1575;&#1582;&#1578;&#1585; PDF",
        ),
        (
            "&#1605;&#1593;&#1575;&#1604;&#1580;&#1577; &#1584;&#1603;&#1610;&#1577;",
            "&#1578;&#1581;&#1604;&#1610;&#1604; &#1575;&#1604;&#1576;&#1606;&#1610;&#1577;",
        ),
        (
            "&#1578;&#1583;&#1602;&#1610;&#1602; &#1575;&#1604;&#1580;&#1608;&#1583;&#1577;",
            "&#1605;&#1585;&#1575;&#1580;&#1593;&#1577; &#1570;&#1604;&#1610;&#1577;",
        ),
        (
            "&#1581;&#1605;&#1617;&#1604; &#1575;&#1604;&#1606;&#1578;&#1610;&#1580;&#1577;",
            "Word &#1580;&#1575;&#1607;&#1586;",
        ),
    ]
    items = ""
    for i, (title, desc) in enumerate(steps):
        items += (
            '<div class="process-step">'
            f'<div class="process-step-icon">{icons[i]}</div>'
            f"<strong>{title}</strong>"
            f"<p>{desc}</p>"
            "</div>"
        )
    return f'<div class="process-flow">{items}</div>'


def features_grid() -> str:
    icons = [
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 7V4a2 2 0 0 1 2-2h8.5L20 7.5V20a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-3"/><polyline points="14 2 14 8 20 8"/><line x1="2" y1="13" x2="10" y2="13"/><line x1="2" y1="17" x2="8" y2="17"/></svg>',
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
    ]
    features = [
        (
            "&#1583;&#1593;&#1605; &#1603;&#1575;&#1605;&#1604; &#1604;&#1604;&#1593;&#1585;&#1576;&#1610;&#1577;",
            "&#1605;&#1581;&#1585;&#1603; &#1605;&#1578;&#1582;&#1589;&#1589; &#1601;&#1610; &#1575;&#1604;&#1606;&#1589;&#1608;&#1589; &#1575;&#1604;&#1593;&#1585;&#1576;&#1610;&#1577; &#1608;&#1575;&#1604;&#1605;&#1582;&#1578;&#1604;&#1591;&#1577;&#8202;&#8212;&#8202;&#1610;&#1581;&#1578;&#1601;&#1592; &#1576;&#1575;&#1604;&#1575;&#1578;&#1580;&#1575;&#1607; &#1608;&#1575;&#1604;&#1578;&#1588;&#1603;&#1610;&#1604; &#1608;&#1575;&#1604;&#1578;&#1606;&#1587;&#1610;&#1602;.",
        ),
        (
            "&#1587;&#1585;&#1593;&#1577; &#1601;&#1575;&#1574;&#1602;&#1577;",
            "&#1578;&#1581;&#1608;&#1610;&#1604; &#1575;&#1604;&#1589;&#1601;&#1581;&#1575;&#1578; &#1601;&#1610; &#1579;&#1608;&#1575;&#1606;&#1613; &#1605;&#1593;&#1583;&#1608;&#1583;&#1577; &#1576;&#1601;&#1590;&#1604; &#1575;&#1604;&#1605;&#1593;&#1575;&#1604;&#1580;&#1577; &#1575;&#1604;&#1605;&#1578;&#1608;&#1575;&#1586;&#1610;&#1577;.",
        ),
        (
            "&#1582;&#1589;&#1608;&#1589;&#1610;&#1577; &#1578;&#1575;&#1605;&#1577;",
            "&#1605;&#1604;&#1601;&#1575;&#1578;&#1603; &#1605;&#1581;&#1605;&#1610;&#1577; &#1608;&#1610;&#1578;&#1605; &#1581;&#1584;&#1601;&#1607;&#1575; &#1578;&#1604;&#1602;&#1575;&#1574;&#1610;&#1611;&#1575; &#1576;&#1593;&#1583; &#1575;&#1604;&#1605;&#1593;&#1575;&#1604;&#1580;&#1577;.",
        ),
    ]
    items = ""
    for i, (title, desc) in enumerate(features):
        items += (
            '<div class="feature-row">'
            f'<div class="feature-icon">{icons[i]}</div>'
            '<div class="feature-content">'
            f"<strong>{title}</strong>"
            f"<p>{desc}</p>"
            "</div></div>"
        )
    return (
        '<div class="features-section">'
        '<div class="features-section-title">'
        "&#1604;&#1605;&#1575;&#1584;&#1575; CLOUDA</div>"
        f"{items}</div>"
    )


def mobile_nav(current: str, options: list[str], labels: dict[str, str]) -> str:
    items = ""
    for opt in options:
        active = " active" if opt == current else ""
        label = html.escape(labels.get(opt, opt))
        items += f'<span class="mobile-nav-item{active}">{label}</span>'
    return (
        '<div class="mobile-nav"><div class="mobile-nav-inner">' f"{items}</div></div>"
    )


def upload_label() -> str:
    return (
        '<p class="upload-label">'
        "&#1575;&#1582;&#1578;&#1585; &#1605;&#1604;&#1601; PDF"
        " &#1608;&#1575;&#1581;&#1583;&#1611;&#1575; &#8212;"
        " &#1575;&#1604;&#1581;&#1583; &#1575;&#1604;&#1571;&#1602;&#1589;&#1609;"
        " &#1633;&#1632;&#1632; &#1605;&#1610;&#1594;&#1575;&#1576;&#1575;&#1610;&#1578;"
        "</p>"
    )


def site_footer() -> str:
    return (
        '<footer class="site-footer">'
        "<p>CLOUDA &#8212; &#1578;&#1581;&#1608;&#1610;&#1604;"
        " &#1575;&#1604;&#1605;&#1587;&#1578;&#1606;&#1583;&#1575;&#1578;"
        " &#1576;&#1584;&#1603;&#1575;&#1569;</p>"
        "</footer>"
    )

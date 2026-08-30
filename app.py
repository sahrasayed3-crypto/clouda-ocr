from __future__ import annotations

import html
import os
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components

from pdfword.database import Database
from pdfword.health import system_health
from pdfword.limits import ProcessingLimits, inspect_pdf_upload, limits_from_env
from pdfword.settings import (
    DEFAULT_SETTINGS,
    load_settings,
    runtime_settings,
    save_settings,
)
from pdfword.ui_components import (
    admin_user_rows,
    auth_required_notice,
    document_capabilities,
    document_card,
    document_state_alert,
    document_summary_strip,
    document_transformation_signature,
    empty_state,
    file_summary,
    guest_job_state,
    load_styles,
    page_header,
    prepared_download_is_available,
    section_title,
    status_meta,
    status_strip,
    user_error_message,
)
from pdfword.ui_status import fetch_system_status

NAV_HOME = "الرئيسية"
NAV_UPLOAD = "رفع مستند"
NAV_DOCUMENTS = "المستندات"
NAV_ACCOUNT = "الحساب"
NAV_GUEST = "تجربة سريعة"
NAV_ADMIN = "الإدارة"


@st.cache_resource
def get_database() -> Database:
    return Database()


database = get_database()
runtime = runtime_settings()

st.set_page_config(
    page_title="Clouda PDF",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="auto",
)
styles = load_styles()
if styles:
    st.markdown(styles, unsafe_allow_html=True)


FIREBASE_COMPONENT = components.declare_component(
    "clouda_firebase_auth",
    path=str(Path(__file__).parent / "pdfword" / "streamlit_firebase_auth"),
)


def _api_base_url() -> str:
    return st.session_state.get("api_base_url", runtime.server_base_url).rstrip("/")


def _api_headers(csrf: bool = False) -> dict[str, str]:
    headers: dict[str, str] = {}
    if csrf and st.session_state.get("csrf_token"):
        headers["X-CSRF-Token"] = st.session_state["csrf_token"]
    return headers


def _guest_api_headers(csrf: bool = False) -> dict[str, str]:
    headers: dict[str, str] = {}
    if csrf and st.session_state.get("guest_csrf_token"):
        headers["X-CSRF-Token"] = st.session_state["guest_csrf_token"]
    return headers


def _api_cookies() -> dict[str, str]:
    cookies: dict[str, str] = {}
    if st.session_state.get("clouda_session"):
        cookies["clouda_session"] = st.session_state["clouda_session"]
    if st.session_state.get("clouda_guest"):
        cookies["clouda_guest"] = st.session_state["clouda_guest"]
    return cookies


def _api_request(method: str, path: str, **kwargs):
    timeout = kwargs.pop("timeout", 20)
    return requests.request(
        method,
        f"{_api_base_url()}{path}",
        headers=kwargs.pop("headers", None),
        cookies=_api_cookies(),
        timeout=timeout,
        **kwargs,
    )


def _status_code(exc: requests.RequestException) -> int | None:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def _auth_user() -> dict:
    user = st.session_state.get("auth_user")
    return user if isinstance(user, dict) else {}


def _is_authenticated() -> bool:
    return bool(st.session_state.get("clouda_session") and _auth_user())


def _is_admin() -> bool:
    return _is_authenticated() and _auth_user().get("role") == "admin"


def _visible_navigation() -> list[str]:
    if not _is_authenticated():
        return [NAV_HOME, NAV_GUEST, NAV_ACCOUNT]
    pages = [NAV_HOME, NAV_UPLOAD, NAV_DOCUMENTS, NAV_ACCOUNT]
    if _is_admin():
        pages.append(NAV_ADMIN)
    return pages


def _navigate(page: str) -> None:
    st.session_state["_requested_page"] = page
    st.rerun()


def _request_mobile_navigation() -> None:
    page = st.session_state.get("mobile_navigation")
    if page:
        st.session_state["_requested_page"] = page


def _load_documents(*, show_error: bool = True) -> list[dict] | None:
    try:
        response = _api_request("GET", "/user/documents")
        response.raise_for_status()
        payload = response.json()
        documents = payload.get("documents", [])
        return documents if isinstance(documents, list) else []
    except requests.RequestException as exc:
        if show_error:
            st.error(user_error_message(_status_code(exc), "documents"))
        return None


def _safe_docx_name(document: dict) -> str:
    name = str(document.get("original_pdf_name") or "result.pdf")
    stem = name[:-4] if name.lower().endswith(".pdf") else name
    return f"{stem}.docx"


def _store_auth_response(response) -> dict:
    response.raise_for_status()
    payload = response.json()
    session_cookie = response.cookies.get("clouda_session")
    if session_cookie:
        st.session_state["clouda_session"] = session_cookie
    if payload.get("csrf_token"):
        st.session_state["csrf_token"] = payload["csrf_token"]
    st.session_state["auth_user"] = payload.get("user", payload)
    return payload


def _auth_mode() -> str:
    try:
        response = _api_request("GET", "/auth/firebase/config", timeout=5)
        if response.status_code == 200:
            return response.json().get("mode", "firebase_staging")
    except requests.RequestException:
        pass
    return "local"


def _firebase_auth_component(api_base_url: str) -> dict | None:
    try:
        response = _api_request("GET", "/auth/firebase/config", timeout=10)
        response.raise_for_status()
        firebase_config = response.json()["config"]
    except (requests.RequestException, KeyError, TypeError):
        st.error("تعذر تحميل إعدادات تسجيل الدخول الآن.")
        return None
    component_value = FIREBASE_COMPONENT(
        firebase_config=firebase_config,
        api_base_url=api_base_url.rstrip("/"),
        default=None,
        key="clouda_firebase_auth",
        height=500,
    )
    return component_value if isinstance(component_value, dict) else None


def _consume_firebase_auth_bridge(bridge_token: str) -> None:
    if not bridge_token:
        return
    try:
        response = _api_request(
            "POST",
            "/auth/streamlit/bridge",
            json={"bridge_token": bridge_token},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        st.session_state["clouda_session"] = payload["session_id"]
        st.session_state["csrf_token"] = payload["csrf_token"]
        st.session_state["auth_user"] = payload["user"]
        st.success("تم تسجيل الدخول بنجاح.")
        _navigate(NAV_HOME)
    except (requests.RequestException, KeyError, TypeError):
        st.error("تعذر إكمال تسجيل الدخول. حاول مرة أخرى.")


def _logout_current_session() -> None:
    try:
        response = _api_request("POST", "/auth/logout", headers=_api_headers(True))
        response.raise_for_status()
    except requests.RequestException:
        st.warning("أُغلقت الجلسة المحلية، وتعذر تأكيد الإغلاق مع الخادم.")
    finally:
        for key in (
            "clouda_session",
            "csrf_token",
            "auth_user",
            "firebase_logout_requested",
            "selected_document_id",
            "document_download",
        ):
            st.session_state.pop(key, None)
    _navigate(NAV_HOME)


system_state = fetch_system_status(
    runtime.server_base_url,
    runtime.worker_api_key,
    requests,
    redis_available=False,
).as_dict()


def _service_summary() -> tuple[str, str]:
    if system_state.get("server") and system_state.get("worker_state") in {
        "ready",
        "busy",
    }:
        return "الخدمة جاهزة", "success"
    if system_state.get("server"):
        return "المعالجة قد تنتظر", "warning"
    return "تعذر التحقق من الخدمة", "danger"


service_label, service_tone = _service_summary()
user_label = _auth_user().get("display_name") or _auth_user().get("email") or ""
user_html = (
    f'<span class="header-user" dir="auto">{html.escape(str(user_label))}</span>'
    if user_label
    else ""
)
st.markdown(
    '<header class="app-header">'
    '<div class="app-brand"><span class="app-brand-mark" aria-hidden="true">C</span>'
    "<div><strong>Clouda PDF</strong><small>تحويل المستندات العربية</small></div></div>"
    '<div class="header-meta">'
    f'{user_html}<span class="service-pill service-{service_tone}">'
    f'<span aria-hidden="true"></span>{html.escape(service_label)}</span>'
    "</div></header>",
    unsafe_allow_html=True,
)


visible_navigation = _visible_navigation()
requested_page = st.session_state.pop("_requested_page", None)
if requested_page in visible_navigation:
    st.session_state["main_navigation"] = requested_page
if st.session_state.get("main_navigation") not in visible_navigation:
    st.session_state["main_navigation"] = visible_navigation[0]

with st.sidebar:
    st.markdown('<div class="sidebar-brand">CLOUDA</div>', unsafe_allow_html=True)
    st.caption("PDF إلى DOCX قابل للتحرير")
    current_page = st.radio(
        "التنقل",
        visible_navigation,
        key="main_navigation",
        label_visibility="collapsed",
    )
    st.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)
    if _is_authenticated():
        st.caption(str(_auth_user().get("email", "")))
    else:
        st.caption("واجهة عربية واضحة وآمنة")

if st.session_state.get("mobile_navigation") != current_page:
    st.session_state["mobile_navigation"] = current_page
st.radio(
    "التنقل السريع",
    visible_navigation,
    horizontal=True,
    key="mobile_navigation",
    label_visibility="collapsed",
    on_change=_request_mobile_navigation,
)


def render_public_home() -> None:
    with st.container(key="public_hero"):
        hero_copy, hero_demo = st.columns(
            [1.04, 0.96], gap="large", vertical_alignment="center"
        )
        with hero_copy:
            st.markdown(
                '<section class="public-hero-copy">'
                '<p class="eyebrow">مساحة تحويل مستندات عربية</p>'
                "<h1>من PDF إلى Word، بوضوح من الرفع حتى التنزيل.</h1>"
                "<p>حوّل النص القابل للتحديد داخل ملف PDF إلى مستند Word قابل للتحرير، "
                "وتابع حالة ملفك الحقيقية دون نسب أو وعود مصطنعة.</p>"
                '<div class="public-hero-facts" aria-label="إمكانات التحويل">'
                "<span>PDF واحد كامل</span><span>حالة معالجة حقيقية</span>"
                "<span>DOCX عند الاكتمال</span></div>"
                "</section>",
                unsafe_allow_html=True,
            )
            primary, secondary = st.columns(2)
            if primary.button("تسجيل الدخول", type="primary", width="stretch"):
                _navigate(NAV_ACCOUNT)
            if secondary.button("تجربة ملف صغير", width="stretch"):
                _navigate(NAV_GUEST)
        with hero_demo:
            transformation_demo = document_transformation_signature(
                status="demo", compact=True
            )
            st.markdown(
                f'<div class="public-hero-demo">{transformation_demo}</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        section_title("مسار وثيقة واحد", "ثلاث نقاط واضحة من الاختيار إلى النتيجة."),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="public-capability-list">'
        "<div><span>01</span><p><strong>اختر الملف</strong> PDF واحد ضمن الحدود الظاهرة.</p></div>"
        "<div><span>02</span><p><strong>تابع الحالة</strong> تحديث حقيقي من خدمة المعالجة.</p></div>"
        "<div><span>03</span><p><strong>نزّل النتيجة</strong> ملف Word بعد اكتمال التحويل.</p></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.info(
        "لا يدّعي Clouda حاليًا معالجة OCR للصفحات المصورة أو إعادة بناء الجداول والصور."
    )


def render_dashboard() -> None:
    if not _is_authenticated():
        render_public_home()
        return
    name = _auth_user().get("display_name") or _auth_user().get("email") or ""
    st.markdown(
        page_header("مساحة العمل", f"مرحبًا {name}. هذه أهم حالات مستنداتك الآن."),
        unsafe_allow_html=True,
    )
    upload_col, library_col, _ = st.columns([1, 1, 2])
    if upload_col.button("رفع مستند جديد", type="primary", width="stretch"):
        _navigate(NAV_UPLOAD)
    if library_col.button("عرض المستندات", width="stretch"):
        _navigate(NAV_DOCUMENTS)

    documents = _load_documents()
    if documents is None:
        return
    active_statuses = {"pending", "processing", "finalizing"}
    active = sum(1 for doc in documents if doc.get("status") in active_statuses)
    completed = sum(1 for doc in documents if doc.get("status") == "completed")
    attention = sum(
        1 for doc in documents if doc.get("status") in {"failed", "manual_review"}
    )
    st.markdown(
        document_summary_strip(
            active=active,
            completed=completed,
            attention=attention,
        ),
        unsafe_allow_html=True,
    )

    if not documents:
        st.markdown(
            empty_state(
                "لا توجد مستندات بعد",
                "ابدأ برفع أول ملف PDF، وستظهر حالته ونتيجته هنا.",
            ),
            unsafe_allow_html=True,
        )
        return

    needs_attention = [
        doc for doc in documents if doc.get("status") in {"failed", "manual_review"}
    ]
    if needs_attention:
        st.markdown(
            section_title("تحتاج إجراءً", "مستندات لم تصل إلى نتيجة قابلة للتنزيل."),
            unsafe_allow_html=True,
        )
        for index, document in enumerate(needs_attention[:3]):
            st.markdown(document_card(document), unsafe_allow_html=True)
            if st.button(
                "عرض التفاصيل",
                key=f"dashboard-attention-{index}-{document.get('job_id', '')}",
            ):
                st.session_state["selected_document_id"] = document.get("job_id", "")
                _navigate(NAV_DOCUMENTS)

    st.markdown(
        section_title("أحدث المستندات", "آخر ما أرسلته إلى Clouda."),
        unsafe_allow_html=True,
    )
    for index, document in enumerate(documents[:5]):
        st.markdown(document_card(document), unsafe_allow_html=True)
        if st.button(
            "فتح المستند",
            key=f"dashboard-recent-{index}-{document.get('job_id', '')}",
        ):
            st.session_state["selected_document_id"] = document.get("job_id", "")
            _navigate(NAV_DOCUMENTS)


def render_upload() -> None:
    st.markdown(
        page_header("رفع مستند", "أرسل ملف PDF كاملًا للتحويل إلى Word."),
        unsafe_allow_html=True,
    )
    if not _is_authenticated():
        st.markdown(auth_required_notice(), unsafe_allow_html=True)
        if st.button("الانتقال إلى تسجيل الدخول", type="primary"):
            _navigate(NAV_ACCOUNT)
        return

    limits = limits_from_env()
    limit_mb = min(limits.max_upload_bytes, limits.max_pdf_bytes) // (1024 * 1024)
    st.markdown(
        '<div class="upload-guidance">'
        "<strong>قبل الرفع</strong>"
        f"<p>ملف PDF واحد، حتى {limit_mb} ميغابايت و{limits.max_pdf_pages} صفحة. "
        "يمكنك إزالة الملف أو استبداله قبل الإرسال من أداة الاختيار.</p></div>",
        unsafe_allow_html=True,
    )
    if system_state.get("server") and system_state.get("worker_state") == "offline":
        st.warning(
            "الخادم متاح، لكن عامل المعالجة غير متصل الآن؛ قد يبقى الملف منتظرًا."
        )
    elif not system_state.get("server"):
        st.info("تعذر التحقق من الخدمة حاليًا. لن يُرسل الملف إلا عند نجاح الاتصال.")

    uploaded_file = st.file_uploader(
        "اختر ملف PDF",
        type=["pdf"],
        accept_multiple_files=False,
        max_upload_size=limit_mb,
        help=f"الحد الأقصى {limit_mb} ميغابايت و{limits.max_pdf_pages} صفحة.",
    )
    validation_error = ""
    page_count = 0
    size_bytes = 0
    if uploaded_file is not None:
        try:
            size_bytes, page_count = inspect_pdf_upload(uploaded_file, limits=limits)
        except ValueError as exc:
            validation_error = str(exc)
        st.markdown(
            file_summary(uploaded_file.name, uploaded_file.size, page_count or None),
            unsafe_allow_html=True,
        )
        if validation_error:
            st.error(validation_error)
        else:
            st.success(
                "الملف صالح للإرسال. سيُحوّل كامل المستند وفق إمكانات الخدمة الحالية."
            )
    else:
        st.markdown(
            empty_state(
                "لم تختر ملفًا", "اختر ملف PDF لعرض حجمه وعدد صفحاته قبل الإرسال."
            ),
            unsafe_allow_html=True,
        )

    submit = st.button(
        "إرسال للتحويل",
        type="primary",
        disabled=uploaded_file is None or bool(validation_error),
        width="stretch",
    )
    if submit and uploaded_file is not None:
        try:
            with st.spinner("جارٍ إرسال الملف بأمان…"):
                response = _api_request(
                    "POST",
                    "/user/documents/upload",
                    headers=_api_headers(True),
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    },
                )
                response.raise_for_status()
                document = response.json()
            st.session_state["selected_document_id"] = document["job_id"]
            st.session_state["upload_success"] = "تم استلام الملف وإرساله للمعالجة."
            _navigate(NAV_DOCUMENTS)
        except requests.Timeout:
            st.session_state["upload_warning"] = user_error_message(
                None, "upload_uncertain"
            )
            _navigate(NAV_DOCUMENTS)
        except (requests.RequestException, KeyError, TypeError) as exc:
            status_code = (
                _status_code(exc)
                if isinstance(exc, requests.RequestException)
                else None
            )
            st.error(user_error_message(status_code, "upload"))


def _refresh_selected_document(document: dict) -> dict:
    job_id = str(document.get("job_id") or "")
    if not job_id:
        return document
    try:
        response = _api_request("GET", f"/user/documents/{job_id}")
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else document
    except requests.RequestException:
        return document


def _run_document_action(method: str, path: str, success_message: str) -> bool:
    try:
        response = _api_request(method, path, headers=_api_headers(True))
        response.raise_for_status()
        st.success(success_message)
        return True
    except requests.RequestException as exc:
        st.error(user_error_message(_status_code(exc), "action"))
        return False


def render_document_detail(document: dict) -> None:
    document = _refresh_selected_document(document)
    job_id = str(document.get("job_id") or "")
    status = str(document.get("status") or "")
    presentation = status_meta(status)
    capabilities = document_capabilities(status)
    filename = str(document.get("original_pdf_name") or "مستند")
    page_count = int(document.get("page_count") or 0)
    created = str(document.get("created_at") or "")[:16].replace("T", " ")

    st.markdown('<div class="detail-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<section class="document-detail-head">'
        '<div><p class="eyebrow">تفاصيل المستند</p>'
        f'<h2 dir="auto">{html.escape(filename)}</h2></div>'
        f'<span class="status-chip status-tone-{presentation["tone"]}">'
        f'{html.escape(presentation["label"])}</span></section>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<dl class="document-facts">'
        "<div><dt>الحالة</dt>"
        f'<dd>{html.escape(presentation["label"])}</dd></div>'
        "<div><dt>الصفحات</dt>"
        f'<dd>{page_count if page_count else "—"}</dd></div>'
        "<div><dt>تاريخ الرفع</dt>"
        f'<dd dir="auto">{html.escape(created or "—")}</dd></div>'
        "</dl>",
        unsafe_allow_html=True,
    )
    st.markdown(
        document_transformation_signature(status=status, filename=filename),
        unsafe_allow_html=True,
    )
    state_alert = document_state_alert(status)
    if state_alert:
        alert_kind, alert_message = state_alert
        if alert_kind == "error":
            st.error(alert_message)
        elif alert_kind == "warning":
            st.warning(alert_message)
        elif alert_kind == "success":
            st.success(alert_message)
        else:
            st.info(alert_message)

    action_cols = st.columns(4)
    if action_cols[0].button("تحديث الحالة", key=f"refresh-{job_id}"):
        st.rerun()
    if capabilities["download"] and action_cols[1].button(
        "تحميل ملف Word", type="primary", key=f"prepare-download-{job_id}"
    ):
        try:
            response = _api_request("GET", f"/user/documents/{job_id}/download")
            response.raise_for_status()
            st.session_state["document_download"] = {
                "job_id": job_id,
                "data": response.content,
                "name": _safe_docx_name(document),
            }
        except requests.RequestException as exc:
            st.error(user_error_message(_status_code(exc), "download"))
    if capabilities["retry"] and action_cols[1].button(
        "إعادة المحاولة", type="primary", key=f"retry-{job_id}"
    ):
        if _run_document_action(
            "POST",
            f"/user/documents/{job_id}/retry",
            "أُعيد المستند إلى قائمة المعالجة.",
        ):
            st.rerun()
    if capabilities["cancel"] and action_cols[2].button(
        "إلغاء المعالجة", key=f"cancel-{job_id}"
    ):
        if _run_document_action(
            "POST", f"/user/documents/{job_id}/cancel", "تم إلغاء المعالجة."
        ):
            st.rerun()

    prepared_download = st.session_state.get("document_download")
    if isinstance(prepared_download, dict) and prepared_download_is_available(
        status, prepared_download, job_id
    ):
        st.download_button(
            "حفظ ملف Word",
            prepared_download["data"],
            file_name=prepared_download["name"],
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            type="primary",
            key=f"save-download-{job_id}",
        )

    if capabilities["delete"]:
        with st.expander("حذف المستند من القائمة"):
            confirmed = st.checkbox(
                "أفهم أن المستند سيُخفى من قائمتي ولا يمكن التراجع من الواجهة.",
                key=f"delete-confirm-{job_id}",
            )
            if st.button(
                "تأكيد الحذف",
                type="secondary",
                disabled=not confirmed,
                key=f"delete-{job_id}",
            ):
                if _run_document_action(
                    "DELETE", f"/user/documents/{job_id}", "حُذف المستند من القائمة."
                ):
                    st.session_state.pop("selected_document_id", None)
                    st.session_state.pop("document_download", None)
                    st.rerun()


def render_documents() -> None:
    st.markdown(
        page_header("المستندات", "ابحث عن مستند وافتح تفاصيله أو نتيجته."),
        unsafe_allow_html=True,
    )
    if not _is_authenticated():
        st.markdown(auth_required_notice(), unsafe_allow_html=True)
        return
    if st.session_state.pop("upload_success", None):
        st.success("تم استلام الملف وإرساله للمعالجة.")
    upload_warning = st.session_state.pop("upload_warning", None)
    if upload_warning:
        st.warning(upload_warning)

    documents = _load_documents()
    if documents is None:
        return
    search_col, filter_col, refresh_col = st.columns([2, 1, 1])
    query = (
        search_col.text_input(
            "البحث باسم الملف",
            placeholder="اكتب جزءًا من اسم الملف",
            key="document_search",
        )
        .strip()
        .casefold()
    )
    status_options = ["all", *list({str(doc.get("status") or "") for doc in documents})]
    status_filter = filter_col.selectbox(
        "الحالة",
        status_options,
        format_func=lambda value: (
            "كل الحالات" if value == "all" else status_meta(value)["label"]
        ),
        key="document_status_filter",
    )
    if refresh_col.button("تحديث القائمة", width="stretch"):
        st.rerun()

    filtered = [
        doc
        for doc in documents
        if (not query or query in str(doc.get("original_pdf_name") or "").casefold())
        and (status_filter == "all" or doc.get("status") == status_filter)
    ]
    if not documents:
        st.markdown(
            empty_state("لا توجد مستندات بعد", "ارفع أول مستند لبدء مكتبتك."),
            unsafe_allow_html=True,
        )
        if st.button("رفع مستند", type="primary"):
            _navigate(NAV_UPLOAD)
        return
    if not filtered:
        st.markdown(
            empty_state("لا توجد نتائج", "غيّر عبارة البحث أو مرشح الحالة."),
            unsafe_allow_html=True,
        )
    else:
        for index, document in enumerate(filtered):
            st.markdown(document_card(document), unsafe_allow_html=True)
            if st.button(
                "عرض التفاصيل",
                key=f"document-open-{index}-{document.get('job_id', '')}",
            ):
                st.session_state["selected_document_id"] = document.get("job_id", "")
                st.session_state.pop("document_download", None)
                st.rerun()

    selected_id = str(st.session_state.get("selected_document_id") or "")
    if selected_id:
        selected = next(
            (doc for doc in documents if str(doc.get("job_id") or "") == selected_id),
            None,
        )
        if selected is not None:
            render_document_detail(selected)
        else:
            st.session_state.pop("selected_document_id", None)


def _guest_limits() -> ProcessingLimits:
    max_bytes = int(os.getenv("CLOUDA_GUEST_MAX_BYTES", str(10 * 1024 * 1024)))
    max_pages = int(os.getenv("CLOUDA_GUEST_MAX_PAGES", "5"))
    return ProcessingLimits(
        max_upload_bytes=max_bytes,
        max_pdf_bytes=max_bytes,
        max_pdf_pages=max_pages,
    )


def render_guest_trial() -> None:
    limits = _guest_limits()
    limit_mb = limits.max_upload_bytes // (1024 * 1024)
    st.markdown(
        page_header(
            "تجربة سريعة",
            f"حوّل ملف PDF واحدًا بلا حساب، حتى {limits.max_pdf_pages} صفحات و{limit_mb} ميغابايت.",
        ),
        unsafe_allow_html=True,
    )
    if not st.session_state.get("clouda_guest"):
        st.info("ابدأ جلسة مؤقتة قبل اختيار الملف. تسمح الجلسة بمهمة نشطة واحدة.")
        if st.button("بدء التجربة", type="primary"):
            try:
                response = _api_request("POST", "/guest/session")
                response.raise_for_status()
                guest_cookie = response.cookies.get("clouda_guest")
                payload = response.json()
                if guest_cookie:
                    st.session_state["clouda_guest"] = guest_cookie
                st.session_state["guest_csrf_token"] = payload.get("csrf_token", "")
                st.rerun()
            except requests.RequestException as exc:
                st.error(user_error_message(_status_code(exc), "action"))
        return

    guest_file = st.file_uploader(
        "اختر ملف PDF للتجربة",
        type=["pdf"],
        key="guest_pdf",
        max_upload_size=limit_mb,
        help=f"حتى {limits.max_pdf_pages} صفحات و{limit_mb} ميغابايت.",
    )
    guest_error = ""
    guest_pages = 0
    if guest_file is not None:
        try:
            _, guest_pages = inspect_pdf_upload(guest_file, limits=limits)
            st.markdown(
                file_summary(guest_file.name, guest_file.size, guest_pages),
                unsafe_allow_html=True,
            )
        except ValueError as exc:
            guest_error = str(exc)
            st.error(guest_error)
    if (
        st.button(
            "إرسال ملف التجربة",
            type="primary",
            disabled=guest_file is None or bool(guest_error),
        )
        and guest_file is not None
    ):
        try:
            response = _api_request(
                "POST",
                "/guest/trial",
                headers=_guest_api_headers(True),
                files={
                    "file": (guest_file.name, guest_file.getvalue(), "application/pdf")
                },
            )
            response.raise_for_status()
            payload = response.json()
            st.session_state["guest_job"] = guest_job_state(
                payload,
                filename=guest_file.name,
                fallback_pages=guest_pages,
            )
            st.session_state.pop("guest_download", None)
            st.success("تم استلام الملف. استخدم تحديث الحالة لمعرفة النتيجة.")
        except (requests.RequestException, KeyError, TypeError) as exc:
            status_code = (
                _status_code(exc)
                if isinstance(exc, requests.RequestException)
                else None
            )
            st.error(user_error_message(status_code, "upload"))

    job = st.session_state.get("guest_job")
    if not isinstance(job, dict) or not job.get("job_id"):
        return
    job_id = str(job["job_id"])
    if st.button("تحديث حالة التجربة", key="guest-refresh"):
        try:
            response = _api_request("GET", f"/guest/jobs/{job_id}")
            response.raise_for_status()
            job.update(response.json())
            st.session_state["guest_job"] = job
        except requests.RequestException as exc:
            st.error(user_error_message(_status_code(exc), "documents"))
    presentation = status_meta(str(job.get("status") or "pending"))
    st.markdown(
        '<div class="guest-status" role="status">'
        f'<strong>{html.escape(presentation["label"])}</strong>'
        f'<p>{html.escape(presentation["description"])}</p></div>',
        unsafe_allow_html=True,
    )
    if job.get("status") == "completed":
        token = str(job.get("result_token") or "")
        if not token:
            st.error("تعذر العثور على رمز تنزيل النتيجة لهذه الجلسة.")
            return
        if st.button("تحميل نتيجة التجربة", type="primary"):
            try:
                response = _api_request(
                    "GET",
                    f"/guest/jobs/{job_id}/download",
                    params={"token": token},
                )
                response.raise_for_status()
                st.session_state["guest_download"] = response.content
            except requests.RequestException as exc:
                st.error(user_error_message(_status_code(exc), "download"))
        if st.session_state.get("guest_download"):
            st.download_button(
                "حفظ ملف Word",
                st.session_state["guest_download"],
                file_name=_safe_docx_name(
                    {"original_pdf_name": job.get("filename", "result.pdf")}
                ),
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary",
            )


def render_account() -> None:
    st.markdown(
        page_header("الحساب", "تسجيل الدخول وإدارة الجلسة الحالية."),
        unsafe_allow_html=True,
    )
    auth_user = _auth_user()
    if st.session_state.get("firebase_logout_requested"):
        _logout_current_session()
    if auth_user:
        display_name = auth_user.get("display_name") or auth_user.get("email") or ""
        st.markdown(
            '<section class="account-summary">'
            f'<p class="eyebrow">الحساب الحالي</p><h2 dir="auto">{html.escape(str(display_name))}</h2>'
            f'<p dir="ltr">{html.escape(str(auth_user.get("email", "")))}</p></section>',
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        if cols[0].button("تحديث بيانات الحساب", width="stretch"):
            try:
                response = _api_request("GET", "/auth/me")
                response.raise_for_status()
                st.session_state["auth_user"] = response.json()
                st.success("تم تحديث بيانات الحساب.")
            except requests.RequestException as exc:
                st.error(user_error_message(_status_code(exc), "action"))
        if cols[1].button("تسجيل الخروج", width="stretch"):
            _logout_current_session()
        with st.expander("طلب حذف الحساب"):
            confirmed = st.checkbox(
                "أفهم أن الطلب سيُراجع وقد يؤدي إلى حذف الحساب وبياناته.",
                key="delete-account-confirm",
            )
            if st.button("إرسال طلب الحذف", disabled=not confirmed):
                try:
                    response = _api_request(
                        "POST", "/auth/delete-account", headers=_api_headers(True)
                    )
                    response.raise_for_status()
                    st.warning("تم إرسال طلب الحذف للمراجعة.")
                except requests.RequestException as exc:
                    st.error(user_error_message(_status_code(exc), "action"))
        return

    if _auth_mode() != "local":
        firebase_event = _firebase_auth_component(_api_base_url())
        if firebase_event and firebase_event.get("type") == "clouda_auth_bridge":
            _consume_firebase_auth_bridge(str(firebase_event.get("bridge_token", "")))
        elif (
            firebase_event and firebase_event.get("type") == "firebase_logout_requested"
        ):
            st.session_state["firebase_logout_requested"] = True
            _logout_current_session()
        return

    login_tab, register_tab, reset_tab = st.tabs(
        ["تسجيل الدخول", "إنشاء حساب", "استعادة كلمة المرور"]
    )
    with login_tab:
        with st.form("login-form"):
            login_email = st.text_input(
                "البريد الإلكتروني", key="login_email_input", autocomplete="email"
            )
            login_password = st.text_input(
                "كلمة المرور",
                type="password",
                key="login_pass_input",
                autocomplete="current-password",
            )
            if st.form_submit_button("تسجيل الدخول", type="primary"):
                try:
                    _store_auth_response(
                        _api_request(
                            "POST",
                            "/auth/dev/login",
                            json={"email": login_email, "password": login_password},
                        )
                    )
                    _navigate(NAV_HOME)
                except requests.RequestException as exc:
                    code = _status_code(exc)
                    if code in {400, 401}:
                        st.error("البريد الإلكتروني أو كلمة المرور غير صحيحة.")
                    else:
                        st.error(user_error_message(code, "action"))
    with register_tab:
        with st.form("register-form"):
            register_email = st.text_input(
                "البريد الإلكتروني", key="register_email", autocomplete="email"
            )
            register_password = st.text_input(
                "كلمة المرور",
                type="password",
                key="register_password",
                autocomplete="new-password",
                help="استخدم كلمة مرور قوية وفق متطلبات الخدمة.",
            )
            if st.form_submit_button("إنشاء الحساب", type="primary"):
                try:
                    response = _api_request(
                        "POST",
                        "/auth/dev/register",
                        json={"email": register_email, "password": register_password},
                    )
                    response.raise_for_status()
                    st.success("تم إنشاء الحساب. فعّل البريد قبل تسجيل الدخول.")
                except requests.RequestException as exc:
                    st.error(user_error_message(_status_code(exc), "action"))
        if st.button("تفعيل البريد في بيئة التطوير"):
            try:
                response = _api_request(
                    "POST", "/auth/dev/verify-email", json={"email": register_email}
                )
                response.raise_for_status()
                st.success("تم تفعيل البريد في بيئة التطوير.")
            except requests.RequestException as exc:
                st.error(user_error_message(_status_code(exc), "action"))
    with reset_tab:
        with st.form("password-reset-start-form"):
            reset_email = st.text_input(
                "البريد الإلكتروني", key="reset_email_input", autocomplete="email"
            )
            if st.form_submit_button("بدء الاستعادة", type="primary"):
                try:
                    response = _api_request(
                        "POST",
                        "/auth/dev/password-reset/start",
                        json={"email": reset_email},
                    )
                    response.raise_for_status()
                    st.session_state["_reset_token"] = response.json().get(
                        "reset_token", ""
                    )
                    st.success("تم بدء استعادة كلمة المرور.")
                except requests.RequestException as exc:
                    st.error(user_error_message(_status_code(exc), "action"))
        reset_token = st.session_state.get("_reset_token", "")
        if reset_token:
            with st.form("password-reset-complete-form"):
                new_password = st.text_input(
                    "كلمة المرور الجديدة",
                    type="password",
                    autocomplete="new-password",
                )
                if st.form_submit_button("حفظ كلمة المرور الجديدة"):
                    try:
                        response = _api_request(
                            "POST",
                            "/auth/dev/password-reset/complete",
                            json={
                                "reset_token": reset_token,
                                "new_password": new_password,
                            },
                        )
                        response.raise_for_status()
                        st.session_state.pop("_reset_token", None)
                        st.success("تم تغيير كلمة المرور.")
                    except requests.RequestException as exc:
                        st.error(user_error_message(_status_code(exc), "action"))


def _verified_admin() -> bool:
    if not _is_admin():
        return False
    try:
        response = _api_request("GET", "/auth/me")
        response.raise_for_status()
        profile = response.json()
        if profile.get("role") != "admin":
            return False
        st.session_state["auth_user"] = profile
        return True
    except requests.RequestException:
        return False


def render_admin() -> None:
    st.markdown(
        page_header("الإدارة", "المستخدمون وإعدادات التشغيل وتشخيص الخدمة."),
        unsafe_allow_html=True,
    )
    if not _verified_admin():
        st.error("تعذر التحقق من صلاحية المسؤول. لن تُعرض أدوات الإدارة.")
        return
    admin_section = st.radio(
        "قسم الإدارة",
        ["المستخدمون", "إعدادات التشغيل", "حالة الخدمة"],
        horizontal=True,
        key="admin_section",
    )
    if admin_section == "المستخدمون":
        try:
            response = _api_request("GET", "/admin/users")
            response.raise_for_status()
            users = response.json().get("users", [])
            st.dataframe(admin_user_rows(users), width="stretch", hide_index=True)
        except requests.RequestException as exc:
            st.error(user_error_message(_status_code(exc), "action"))
        st.markdown("#### تغيير حالة مستخدم")
        user_id = st.text_input("معرّف المستخدم", key="admin_target_user_id")
        status = st.selectbox(
            "الحالة الجديدة",
            ["active", "disabled", "deletion_pending", "deleted"],
            format_func=lambda value: {
                "active": "نشط",
                "disabled": "معطّل",
                "deletion_pending": "بانتظار الحذف",
                "deleted": "محذوف",
            }[value],
        )
        if st.button("تطبيق الحالة", type="primary", disabled=not user_id):
            try:
                response = _api_request(
                    "POST",
                    f"/admin/users/{user_id}/status",
                    headers=_api_headers(True),
                    json={"status": status},
                )
                response.raise_for_status()
                st.success("تم تحديث حالة المستخدم.")
            except requests.RequestException as exc:
                st.error(user_error_message(_status_code(exc), "action"))
    elif admin_section == "إعدادات التشغيل":
        settings = load_settings(database)
        with st.form("settings-form"):
            acceptance = st.number_input(
                "حد القبول",
                0.0,
                100.0,
                float(settings["acceptance_threshold"]),
            )
            max_pages = st.number_input(
                "الحد الأقصى لصفحات PDF",
                1,
                5000,
                int(settings["max_pdf_pages"]),
            )
            storage_root = st.text_input(
                "مسار التخزين", value=str(settings["storage_root"])
            )
            enabled_engines = st.multiselect(
                "محركات الاستخراج المفعّلة",
                list(DEFAULT_SETTINGS["enabled_engines"]),
                default=[
                    engine
                    for engine in settings["enabled_engines"]
                    if engine in DEFAULT_SETTINGS["enabled_engines"]
                ],
            )
            if st.form_submit_button("حفظ الإعدادات", type="primary"):
                save_settings(
                    database,
                    {
                        "acceptance_threshold": acceptance,
                        "max_pdf_pages": max_pages,
                        "storage_root": storage_root,
                        "enabled_engines": enabled_engines,
                    },
                )
                st.success("تم حفظ الإعدادات.")
    elif admin_section == "حالة الخدمة":
        st.markdown(
            status_strip(system_state, direct_text_ready=True, future_ocr_ready=False),
            unsafe_allow_html=True,
        )
        health = system_health(database, runtime.storage_root)
        health_status = health.get("status", "unknown")
        st.metric(
            "الحالة العامة",
            {"healthy": "سليمة", "degraded": "متدهورة", "unhealthy": "متعطلة"}.get(
                health_status, "غير معروفة"
            ),
        )
        checks = health.get("checks", {})
        if checks:
            st.dataframe(
                [
                    {
                        "الفحص": name,
                        "الحالة": (
                            "سليم"
                            if isinstance(value, dict) and value.get("ok")
                            else "يحتاج مراجعة"
                        ),
                    }
                    for name, value in checks.items()
                ],
                width="stretch",
                hide_index=True,
            )
        from pdfword.teacher_pipeline.service import safe_status

        teacher_status = safe_status(database)
        with st.expander("مسار المعلّمين الخامل"):
            st.write(f"وضع المسار: {teacher_status['pipeline_mode']}")
            blockers = teacher_status.get("activation_blockers", [])
            if blockers:
                st.warning(f"عوائق التفعيل المسجلة: {len(blockers)}")


if current_page == NAV_HOME:
    render_dashboard()
elif current_page == NAV_UPLOAD:
    render_upload()
elif current_page == NAV_DOCUMENTS:
    render_documents()
elif current_page == NAV_ACCOUNT:
    render_account()
elif current_page == NAV_GUEST:
    render_guest_trial()
elif current_page == NAV_ADMIN:
    render_admin()

st.markdown(
    '<footer class="site-footer"><p>Clouda PDF — مساحة واضحة لتحويل المستندات.</p></footer>',
    unsafe_allow_html=True,
)

import io
from html.parser import HTMLParser
from pathlib import Path

import pytest
from pypdf import PdfWriter
from streamlit.testing.v1 import AppTest

import pdfword.ui_components as ui
import pdfword.limits as ui_limits
from pdfword.limits import ProcessingLimits
from pdfword.ranges import select_pages
from pdfword.ui_components import (
    file_summary,
    load_styles,
    processing_panel,
    status_strip,
)
from pdfword.ui_status import UiSystemStatus, fetch_system_status, parse_health_payload

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.mark.parametrize(
    ("status", "label", "tone"),
    [
        ("pending", "بانتظار المعالجة", "warning"),
        ("processing", "قيد المعالجة", "info"),
        ("finalizing", "تجهيز النتيجة", "info"),
        ("completed", "جاهز للتنزيل", "success"),
        ("manual_review", "تحتاج مراجعة", "warning"),
        ("failed", "تعذر التحويل", "danger"),
        ("cancelled", "أُلغي التحويل", "neutral"),
    ],
)
def test_status_meta_canonically_maps_real_backend_states(status, label, tone):
    meta = ui.status_meta(status)
    assert meta["label"] == label
    assert meta["tone"] == tone


def test_unknown_status_is_neutral_and_does_not_gain_actions():
    assert ui.status_meta("unexpected")["tone"] == "neutral"
    assert ui.document_capabilities("unexpected") == {
        "download": False,
        "retry": False,
        "cancel": False,
        "delete": False,
    }


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (
            "pending",
            {"download": False, "retry": False, "cancel": True, "delete": False},
        ),
        (
            "processing",
            {"download": False, "retry": False, "cancel": True, "delete": False},
        ),
        (
            "finalizing",
            {"download": False, "retry": False, "cancel": False, "delete": False},
        ),
        (
            "completed",
            {"download": True, "retry": False, "cancel": False, "delete": True},
        ),
        (
            "failed",
            {"download": False, "retry": True, "cancel": False, "delete": True},
        ),
        (
            "cancelled",
            {"download": False, "retry": True, "cancel": False, "delete": True},
        ),
        (
            "manual_review",
            {"download": False, "retry": False, "cancel": False, "delete": True},
        ),
    ],
)
def test_document_capabilities_follow_backend_status_transitions(status, expected):
    assert ui.document_capabilities(status) == expected


@pytest.mark.parametrize(
    ("status", "state_class"),
    [
        ("idle", "transform-state-idle"),
        ("pending", "transform-state-active"),
        ("processing", "transform-state-active"),
        ("finalizing", "transform-state-active"),
        ("completed", "transform-state-completed"),
        ("failed", "transform-state-attention"),
        ("manual_review", "transform-state-review"),
        ("cancelled", "transform-state-neutral"),
    ],
)
def test_document_transformation_signature_uses_only_real_states(status, state_class):
    rendered = ui.document_transformation_signature(
        status=status,
        filename="تقرير Q3 <نهائي>.pdf",
    )

    assert state_class in rendered
    assert f'data-status="{status}"' in rendered
    assert "document-source" in rendered
    assert "recognition-sweep" in rendered
    assert "structure-lines" in rendered
    assert "document-target" in rendered
    assert "تقرير Q3 &lt;نهائي&gt;.pdf" in rendered
    assert "<نهائي>" not in rendered
    assert "%" not in rendered
    assert "المرحلة" not in rendered
    assert "confidence" not in rendered.casefold()


def test_active_document_transformation_signature_is_a_live_region():
    rendered = ui.document_transformation_signature(status="processing")

    assert 'role="status"' in rendered
    assert 'aria-live="polite"' in rendered
    assert "يمكنك مغادرة الصفحة والعودة لاحقًا" in rendered


def test_explanatory_transformation_signature_is_not_live_processing():
    rendered = ui.document_transformation_signature(status="demo", compact=True)

    assert 'data-explanatory="true"' in rendered
    assert 'role="group"' in rendered
    assert 'role="status"' not in rendered
    assert "وليس حالة ملف مرفوع" in rendered
    assert "signature-compact" in rendered
    assert "%" not in rendered


def test_manual_review_state_gives_only_supported_next_steps():
    rendered = ui.document_transformation_signature(status="manual_review")

    assert "حدّث قائمة المستندات لاحقًا" in rendered
    assert "احذف المستند" in rendered


def test_document_summary_strip_is_one_compact_accessible_surface():
    rendered = ui.document_summary_strip(active=2, completed=4, attention=1)

    assert rendered.count("document-summary-strip") == 1
    assert 'aria-label="ملخص المستندات"' in rendered
    assert "2 قيد العمل" in rendered
    assert "4 جاهزة للتنزيل" in rendered
    assert "1 تحتاج انتباهك" in rendered


@pytest.mark.parametrize("status", list(ui.STATUS_META))
def test_known_document_state_does_not_duplicate_the_signature_with_an_alert(status):
    assert ui.document_state_alert(status) is None


def test_unknown_document_state_keeps_one_safe_information_alert():
    alert = ui.document_state_alert("unexpected")

    assert alert is not None
    assert alert[0] == "info"
    assert alert[1] == ui.status_meta("unexpected")["description"]


@pytest.mark.parametrize(
    ("status_code", "context", "expected"),
    [
        (401, "documents", "انتهت جلسة الدخول. سجّل الدخول مرة أخرى."),
        (403, "action", "ليست لديك صلاحية لتنفيذ هذا الإجراء."),
        (413, "upload", "يتجاوز الملف حد الحجم أو عدد الصفحات المسموح."),
        (429, "upload", "وصلت إلى حد الاستخدام الحالي. حاول لاحقًا."),
        (None, "upload", "تعذر الاتصال بالخدمة. تحقق من الاتصال وحاول مرة أخرى."),
        (
            None,
            "upload_uncertain",
            "لم تصل استجابة مؤكدة بعد الإرسال. راجع قائمة المستندات قبل إعادة المحاولة لتجنب تكرار الملف.",
        ),
    ],
)
def test_user_error_messages_are_actionable_and_do_not_expose_exceptions(
    status_code, context, expected
):
    assert ui.user_error_message(status_code, context) == expected


def test_document_card_uses_public_job_id_and_escapes_mixed_filename():
    rendered = ui.document_card(
        {
            "job_id": "job-42",
            "original_pdf_name": '<bdo dir="ltr">بحث 2026.pdf</bdo>',
            "status": "completed",
            "page_count": 8,
            "created_at": "2026-08-16T10:30:00+00:00",
        }
    )
    assert 'data-doc-id="job-42"' in rendered
    assert "<bdo" not in rendered
    assert "&lt;bdo" in rendered
    assert "8 صفحات" in rendered
    assert "جاهز للتنزيل" in rendered


def test_guest_job_state_preserves_download_token_and_real_status():
    state = ui.guest_job_state(
        {
            "job_id": "guest-42",
            "status": "pending",
            "page_count": 4,
            "result_token": "one-time-token",
        },
        filename="بحث.pdf",
        fallback_pages=1,
    )
    assert state == {
        "job_id": "guest-42",
        "status": "pending",
        "page_count": 4,
        "result_token": "one-time-token",
        "filename": "بحث.pdf",
    }


def test_prepared_download_is_hidden_after_document_leaves_completed_state():
    prepared = {"job_id": "job-42", "data": b"docx", "name": "result.docx"}

    assert ui.prepared_download_is_available("completed", prepared, "job-42")
    assert not ui.prepared_download_is_available("failed", prepared, "job-42")
    assert not ui.prepared_download_is_available("completed", prepared, "job-99")


def test_admin_user_rows_are_arabic_and_limit_the_visible_fields():
    rows = ui.admin_user_rows(
        [
            {
                "user_id": "user-42",
                "normalized_email": "qa@example.test",
                "display_name": "QA",
                "role": "admin",
                "status": "active",
                "email_verified": True,
                "internal_note": "must-not-render",
            }
        ]
    )

    assert rows == [
        {
            "معرّف المستخدم": "user-42",
            "البريد الإلكتروني": "qa@example.test",
            "الاسم": "QA",
            "الدور": "مسؤول",
            "الحالة": "نشط",
            "البريد مفعّل": "نعم",
        }
    ]


def test_guest_job_state_rejects_response_without_job_id():
    with pytest.raises(ValueError, match="job identifier"):
        ui.guest_job_state({}, filename="بحث.pdf", fallback_pages=1)


def _pdf_file(page_count: int = 1, *, encrypted: bool = False) -> io.BytesIO:
    output = io.BytesIO()
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=200, height=200)
    if encrypted:
        writer.encrypt("test-password")
    writer.write(output)
    output.seek(0)
    return output


def test_upload_inspection_returns_real_bytes_and_pages_and_resets_stream():
    uploaded = _pdf_file(3)
    size, pages = ui_limits.inspect_pdf_upload(
        uploaded,
        limits=ProcessingLimits(max_upload_bytes=1024 * 1024, max_pdf_pages=5),
    )
    assert size == len(uploaded.getvalue())
    assert pages == 3
    assert uploaded.tell() == 0


def test_upload_inspection_rejects_password_protected_pdf():
    with pytest.raises(ValueError, match="محمي بكلمة مرور"):
        ui_limits.inspect_pdf_upload(_pdf_file(encrypted=True))


def test_upload_inspection_rejects_oversized_and_page_limited_pdf():
    uploaded = _pdf_file(2)
    with pytest.raises(ValueError, match="حجم الملف"):
        ui_limits.inspect_pdf_upload(
            uploaded,
            limits=ProcessingLimits(max_upload_bytes=16, max_pdf_pages=5),
        )
    with pytest.raises(ValueError, match="عدد صفحات"):
        ui_limits.inspect_pdf_upload(
            uploaded,
            limits=ProcessingLimits(max_upload_bytes=1024 * 1024, max_pdf_pages=1),
        )


def _offline_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "ui-navigation.sqlite3"))
    monkeypatch.setenv("SERVER_BASE_URL", "http://127.0.0.1:59999")
    monkeypatch.setenv("WORKER_API_KEY", "test-only-worker-key")
    return AppTest.from_file(APP_PATH, default_timeout=20)


def _navigation(app: AppTest):
    return next(widget for widget in app.radio if widget.label == "التنقل")


def test_visitor_navigation_exposes_only_public_destinations(tmp_path, monkeypatch):
    app = _offline_app(tmp_path, monkeypatch)
    app.run()
    assert not app.exception
    assert list(_navigation(app).options) == ["الرئيسية", "تجربة سريعة", "الحساب"]


def test_authenticated_navigation_is_task_focused_and_hides_admin(
    tmp_path, monkeypatch
):
    app = _offline_app(tmp_path, monkeypatch)
    app.session_state["auth_user"] = {
        "email": "user@example.com",
        "display_name": "مستخدم",
        "role": "user",
    }
    app.session_state["clouda_session"] = "session-for-ui-test"
    app.run()
    assert not app.exception
    assert list(_navigation(app).options) == [
        "الرئيسية",
        "رفع مستند",
        "المستندات",
        "الحساب",
    ]


def test_admin_navigation_adds_one_conditional_workspace(tmp_path, monkeypatch):
    app = _offline_app(tmp_path, monkeypatch)
    app.session_state["auth_user"] = {
        "email": "admin@example.com",
        "display_name": "مسؤول",
        "role": "admin",
    }
    app.session_state["clouda_session"] = "admin-session-for-ui-test"
    app.run()
    assert not app.exception
    assert list(_navigation(app).options) == [
        "الرئيسية",
        "رفع مستند",
        "المستندات",
        "الحساب",
        "الإدارة",
    ]


def test_admin_sections_render_lazily_instead_of_running_all_diagnostics():
    source = Path("app.py").read_text(encoding="utf-8")
    admin_source = source.split("def render_admin()", 1)[1].split(
        "if current_page == NAV_HOME", 1
    )[0]

    assert "st.tabs(" not in admin_source
    assert 'key="admin_section"' in admin_source


def test_upload_page_does_not_offer_backend_unsupported_controls(tmp_path, monkeypatch):
    app = _offline_app(tmp_path, monkeypatch)
    app.session_state["auth_user"] = {
        "email": "user@example.com",
        "display_name": "مستخدم",
        "role": "user",
    }
    app.session_state["clouda_session"] = "session-for-ui-test"
    app.run()
    _navigation(app).set_value("رفع مستند").run()
    assert not app.exception
    assert len(app.file_uploader) == 1
    assert all(widget.label != "Selection mode" for widget in app.radio)
    assert all(widget.label != "اسم ملف Word الناتج" for widget in app.text_input)


def test_each_file_picker_displays_its_real_upload_limit():
    source = Path("app.py").read_text(encoding="utf-8")

    assert source.count("max_upload_size=limit_mb") == 2


def test_range_10_to_15_has_six_pages():
    ranges, pages = select_pages("Range", 20, start=10, end=15)
    assert ranges == [(10, 15)]
    assert pages == [10, 11, 12, 13, 14, 15]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": 5, "end": 2},
        {"start": 0, "end": 2},
        {"start": 1, "end": 21},
    ],
)
def test_invalid_ranges_are_rejected(kwargs):
    with pytest.raises(ValueError):
        select_pages("Range", 20, **kwargs)


def test_page_selection_is_independent_from_parallelism():
    _, pages = select_pages("Separate pages", 20, separate="1, 4, 9")
    parallel_pages = 2
    assert pages == [1, 4, 9]
    assert parallel_pages != len(pages)


def test_worker_and_tools_status_are_visible_without_secrets():
    status = parse_health_payload(
        {
            "status": "ok",
            "redis_available": True,
            "cloud_available": False,
            "workers": [{"worker_name": "windows-worker-1", "state": "ready"}],
        }
    )
    html = status_strip(
        status.as_dict(), direct_text_ready=True, future_ocr_ready=False
    )
    assert "Worker" in html
    assert "Direct PDF text" in html
    assert "Future OCR" in html
    assert "WORKER_API_KEY" not in html
    assert "OPENROUTER_API_KEY" not in html


def test_offline_worker_status():
    status = parse_health_payload(
        {
            "status": "ok",
            "redis_available": True,
            "workers": [],
            "cloud_available": False,
        }
    )
    assert status.worker_state == "offline"
    assert status.cloud is False


def test_legacy_server_uses_redis_and_marks_cloud_unknown():
    status = parse_health_payload(
        {"status": "ok", "role": "server"},
        legacy_status=True,
        redis_available=True,
    )
    assert status.server is True
    assert status.redis is True
    assert status.worker_state == "ready"
    assert status.cloud is None
    assert status.service_status_outdated is True
    html = status_strip(status.as_dict())
    assert "Status Center" in html
    assert "Worker" in html


def test_status_fetch_detects_legacy_404():
    class Response:
        def __init__(self, status_code, payload=None):
            self.status_code = status_code
            self._payload = payload or {}

        def raise_for_status(self):
            if self.status_code >= 400 and self.status_code != 404:
                raise RuntimeError(self.status_code)

        def json(self):
            return self._payload

    class Requester:
        @staticmethod
        def get(url, **_kwargs):
            if url.endswith("/internal/health"):
                return Response(200, {"status": "ok", "role": "server"})
            return Response(404)

    status = fetch_system_status(
        "http://server",
        "test-key",
        Requester,
        redis_available=True,
    )
    assert status.worker_state == "ready"
    assert status.cloud is None
    assert status.service_status_outdated is True


def test_status_fetch_falls_back_to_public_health_for_local_processing():
    class Response:
        def __init__(self, status_code, payload=None):
            self.status_code = status_code
            self._payload = payload or {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(self.status_code)

        def json(self):
            return self._payload

    class Requester:
        @staticmethod
        def get(url, **_kwargs):
            if url.endswith("/internal/health"):
                return Response(401)
            return Response(
                200,
                {
                    "status": "ok",
                    "role": "server",
                    "local_processing_enabled": True,
                },
            )

    status = fetch_system_status("http://server", "stale-key", Requester)
    assert status.server is True
    assert status.worker_state == "ready"
    assert status.worker_count == 1


def test_ui_html_escapes_uploaded_filename():
    rendered = file_summary('<script>alert("x")</script>.pdf', 100, 1)
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert 'dir="auto"' in rendered


def test_styles_support_rtl_and_mobile():
    css = Path("assets/styles.css").read_text(encoding="utf-8")
    assert "max-width:800px" in css or "@media (max-width: 800px)" in css
    assert "max-width:430px" in css or "@media (max-width: 430px)" in css
    assert "--c-primary" in css
    assert "radial-gradient" not in css
    assert "prefers-reduced-motion" in css
    assert "min-height: 2.75rem" in css
    assert "--space-1" in css
    assert "--font-arabic" in css
    assert "unicode-bidi: plaintext" in css
    assert ".document-detail-head" in css
    assert ".honest-processing" in css
    assert '[data-testid="stHeader"]' in css
    assert "height: 0;" not in css
    assert "backdrop-filter" not in css
    assert '[data-testid="stSidebar"][aria-expanded="false"]' in css
    assert "overflow: hidden !important" in css
    assert "visibility: hidden !important" in css
    assert ".st-key-mobile_navigation" in css
    assert ".st-key-admin_section" in css
    assert 'key="mobile_navigation"' in Path("app.py").read_text(encoding="utf-8")
    assert "@media (max-width: 1024px)" in css
    assert "@media (max-width: 768px)" in css
    assert "@media (max-width: 390px)" in css
    rendered = load_styles()
    assert rendered.startswith("<style>")
    key_prefix = "sk" + "-or-"
    assert key_prefix not in rendered


def test_premium_motion_css_has_one_coherent_accessible_motion_system():
    css = Path("assets/styles.css").read_text(encoding="utf-8")

    assert "--motion-instant: 100ms" in css
    assert "--motion-micro: 160ms" in css
    assert "--motion-component: 220ms" in css
    assert "--motion-major: 280ms" in css
    assert "--ease-natural: cubic-bezier(.2, .8, .2, 1)" in css
    assert "--ease-emphasized: cubic-bezier(.16, 1, .3, 1)" in css
    assert ".document-signature" in css
    assert ".recognition-sweep" in css
    assert "@keyframes recognition-scan" in css
    assert ".transform-state-completed" in css
    assert ".transform-state-attention" in css
    assert ".transform-state-review" in css
    assert ".document-summary-strip" in css
    assert ".st-key-public_hero" in css
    assert ".public-hero-copy {\n  max-width: 41rem;\n  text-align: right;" in css
    assert ".document-facts" in css
    assert '[data-testid="stFileUploader"]:hover' in css
    assert '[data-testid="stFileUploader"]:focus-within' in css
    assert ".stButton > button:active" in css
    assert "transform: translateY(-1px)" in css
    reduced_motion = css.split("@media (prefers-reduced-motion: reduce)", 1)[1]
    assert ".recognition-sweep" in reduced_motion
    assert "animation: none !important" in reduced_motion
    assert "transform: none !important" in reduced_motion
    for spatial_selector in (
        ".stButton > button:hover",
        ".stDownloadButton > button:hover",
        '[data-testid="stFormSubmitButton"] > button:hover',
        '[data-testid="stFileUploader"]:hover',
        ".doc-card:hover",
        ".transform-state-completed .document-target .signature-sheet",
    ):
        assert spatial_selector in reduced_motion
    assert "structure-breathe" not in css


def test_streamlit_native_controls_use_the_same_light_palette():
    config = Path(".streamlit/config.toml").read_text(encoding="utf-8")

    assert 'base = "light"' in config
    assert 'primaryColor = "#1f4e79"' in config
    assert 'textColor = "#18212b"' in config


def test_firebase_auth_fields_have_visible_associated_labels():
    class FormParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.inputs: set[str] = set()
            self.labels: set[str] = set()

        def handle_starttag(self, tag, attrs):
            values = dict(attrs)
            if tag == "input" and values.get("id"):
                self.inputs.add(values["id"])
            if tag == "label" and values.get("for"):
                self.labels.add(values["for"])

    parser = FormParser()
    parser.feed(
        Path("pdfword/streamlit_firebase_auth/index.html").read_text(encoding="utf-8")
    )
    assert parser.inputs == {
        "email",
        "password",
        "signup-email",
        "signup-password",
        "reset-email",
    }
    assert parser.labels == parser.inputs


def test_firebase_auth_component_matches_premium_motion_and_control_states():
    markup = Path("pdfword/streamlit_firebase_auth/index.html").read_text(
        encoding="utf-8"
    )

    assert "--motion-instant: 100ms" in markup
    assert "--motion-micro: 160ms" in markup
    assert "--motion-component: 220ms" in markup
    assert ".auth-btn:disabled" in markup
    assert "transform: translateY(-1px)" in markup
    assert "transform: translateY(0)" in markup
    reduced_motion = markup.split("@media (prefers-reduced-motion: reduce)", 1)[1]
    assert "transform: none !important" in reduced_motion


def test_streamlit_auth_guest_and_admin_pages_are_backend_connected():
    source = Path("app.py").read_text(encoding="utf-8-sig")

    assert "/auth/dev/register" in source
    assert "/auth/dev/login" in source
    assert "/auth/logout" in source
    assert "/guest/session" in source
    assert "/guest/trial" in source
    assert "/admin/users" in source
    assert "/admin/users/{user_id}/status" in source
    assert "/auth/dev/password-reset/start" in source
    assert "/auth/dev/password-reset/complete" in source
    assert "/user/documents/upload" in source
    assert "create_job_storage" not in source
    assert "atomic_write_stream" not in source
    assert "database.create_conversion" not in source
    assert "database.list_conversions" not in source
    assert "database.statistics(username)" not in source
    assert "informational only" not in source


def test_streamlit_composes_the_premium_signature_without_dashboard_kpis():
    source = Path("app.py").read_text(encoding="utf-8-sig")

    assert "document_transformation_signature" in source
    assert 'status="demo"' in source
    assert "document_summary_strip(" in source
    assert 'key="public_hero"' in source
    assert "document-facts" in source
    assert source.index('primary.button("تسجيل الدخول"') < source.index(
        "transformation_demo = document_transformation_signature("
    )
    assert "metric_cols = st.columns(3)" not in source
    assert "facts[0].metric" not in source
    assert "honest-processing" not in source


def test_windows_bat_wrappers_use_script_directory_and_bypass_policy():
    for name, script in {
        "start_clouda_all.bat": "start_clouda_all.ps1",
        "stop_clouda_all.bat": "stop_clouda_all.ps1",
    }.items():
        content = Path(name).read_text(encoding="utf-8")
        assert "%~dp0" in content
        assert "-ExecutionPolicy Bypass" in content
        assert script in content
        assert "D:\\clouda" not in content
        assert "E:\\clouda" not in content
        assert "F:\\clouda_merged_work" not in content
        assert "F:\\clouda_merged_state" not in content


def test_windows_powershell_launchers_default_to_unified_relative_roots():
    for name in ("start_clouda_all.ps1", "run_server.ps1"):
        content = Path(name).read_text(encoding="utf-8")
        assert "$env:CLOUDA_PROJECT_ROOT" in content
        assert "$env:CLOUDA_STATE_HOME" in content
        assert (
            "Join-Path $root '_state'" in content
            or "Join-Path $PSScriptRoot '_state'" in content
        )
        assert "E:\\clouda" not in content
        assert "F:\\clouda_merged_work" not in content
        assert "F:\\clouda_merged_state" not in content


def test_processing_panel_is_user_facing_and_accessible():
    rendered = processing_panel(
        stage="Digital text extraction",
        progress=42,
        completed_pages=2,
        total_pages=5,
        elapsed="12 seconds",
        last_update="12:30:04",
    )
    assert "Digital text extraction" in rendered
    assert "42%" in rendered
    assert ".env" not in rendered
    assert "Redis" not in rendered


def test_processing_panel_shows_live_indeterminate_state():
    rendered = processing_panel(
        stage="Digital text extraction",
        progress=None,
        completed_pages=0,
        total_pages=5,
        elapsed="3 seconds",
        last_update="12:30:04",
    )
    assert "indeterminate" in rendered
    assert "12:30:04" in rendered


def test_status_dataclass_defaults_are_safe():
    status = UiSystemStatus()
    assert status.worker_state == "offline"
    assert status.cloud is False


def test_streamlit_conversion_page_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "ui.sqlite3"))
    monkeypatch.setenv("SERVER_BASE_URL", "http://127.0.0.1:59999")
    monkeypatch.setenv("WORKER_API_KEY", "test-only-worker-key")
    app = AppTest.from_file(APP_PATH, default_timeout=20)
    app.run()
    assert not app.exception
    assert len(app.file_uploader) == 0
    assert not any(widget.key == "login_username" for widget in app.text_input)
    labels = [widget.label for widget in app.radio]
    assert "التنقل" in labels
    visible_text = " ".join(
        str(element.value)
        for element in [*app.markdown, *app.caption, *app.info, *app.warning]
    )
    assert "OPENROUTER_API_KEY" not in visible_text
    assert "WORKER_API_KEY" not in visible_text
    navigation = next(widget for widget in app.radio if widget.label == "التنقل")
    navigation.set_value("الحساب").run()
    assert not app.exception

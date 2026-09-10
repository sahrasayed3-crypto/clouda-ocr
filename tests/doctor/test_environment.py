"""Runtime, import-resolution, and dependency check tests (Phases 3-5).

Import-resolution scenarios monkeypatch ``importlib.import_module`` inside
``clouda_data.doctor.environment`` to simulate:
- correct current worktree
- editable install pointing at another worktree
- mixed roots across canonical packages
- import failure
"""

from __future__ import annotations

import sys
from types import ModuleType


import clouda_data.doctor.environment as env
from clouda_data.doctor.environment import (
    DIST_TO_IMPORT,
    check_dependency_group,
    check_import_resolution,
    check_runtime,
    installed_version,
    python_supported,
)

# ---------------------------------------------------------------------------
# Phase 3 — runtime
# ---------------------------------------------------------------------------


def test_python_supported_current_interpreter():
    ok, requires = python_supported({})
    assert requires == ">=3.11,<3.12"
    # The suite itself runs on the project's supported range.
    assert ok == (sys.version_info >= (3, 11) and sys.version_info < (3, 12))


def test_python_supported_bounds():
    assert (
        python_supported({"project": {"requires-python": ">=3.11,<3.12"}})[1]
        == ">=3.11,<3.12"
    )


def test_check_runtime_reports_platform_facts():
    checks = check_runtime({})
    by_id = {c.id: c for c in checks}
    version_check = by_id["core.python-version"]
    details = version_check.details
    assert details["os"]
    assert details["machine"]
    assert details["executable"]
    assert "python" in details
    # interpreter check is informational
    assert by_id["core.interpreter"].required is False


# ---------------------------------------------------------------------------
# Phase 4 — import resolution (monkeypatched module origins)
# ---------------------------------------------------------------------------


def _fake_module(package: str, file_path: str) -> ModuleType:
    module = ModuleType(package)
    module.__file__ = file_path
    return module


def test_import_resolution_correct_worktree(clean_worktree_root, monkeypatch):
    def fake_import(name, *a, **k):
        if name in env.CANONICAL_PACKAGES:
            return _fake_module(name, str(clean_worktree_root / name / "__init__.py"))
        return __import__(name)

    monkeypatch.setattr(env.importlib, "import_module", fake_import)
    checks = check_import_resolution(clean_worktree_root)
    check = checks[0]
    assert check.status.value == "PASS"
    assert check.details["expected_source_root"] == str(clean_worktree_root)
    assert check.details["actual_import_roots"] == [str(clean_worktree_root)]


def test_import_resolution_wrong_worktree(clean_worktree_root, tmp_path, monkeypatch):
    other_tree = tmp_path / "OTHER_WORKTREE"
    other_tree.mkdir()

    def fake_import(name, *a, **k):
        if name in env.CANONICAL_PACKAGES:
            return _fake_module(name, str(other_tree / name / "__init__.py"))
        return __import__(name)

    monkeypatch.setattr(env.importlib, "import_module", fake_import)
    checks = check_import_resolution(clean_worktree_root)
    check = checks[0]
    assert check.status.value == "WARN"
    assert check.details["actual_import_roots"] == [str(other_tree)]
    assert "pip install -e ." in (check.remediation or "")


def test_import_resolution_mixed_roots(clean_worktree_root, tmp_path, monkeypatch):
    tree_a = tmp_path / "TREE_A"
    tree_b = tmp_path / "TREE_B"

    def fake_import(name, *a, **k):
        if name == "clouda_data":
            return _fake_module(name, str(tree_a / name / "__init__.py"))
        if name in env.CANONICAL_PACKAGES:
            return _fake_module(name, str(tree_b / name / "__init__.py"))
        return __import__(name)

    monkeypatch.setattr(env.importlib, "import_module", fake_import)
    checks = check_import_resolution(clean_worktree_root)
    check = checks[0]
    assert check.status.value == "FAIL"
    assert check.details["consistent"] is False
    assert len(check.details["actual_import_roots"]) == 2


def test_import_resolution_import_error(clean_worktree_root, monkeypatch):
    def fake_import(name, *a, **k):
        if name == "clouda_training":
            raise ModuleNotFoundError("shadowed by stale install")
        if name in env.CANONICAL_PACKAGES:
            return _fake_module(name, str(clean_worktree_root / name / "__init__.py"))
        return __import__(name)

    monkeypatch.setattr(env.importlib, "import_module", fake_import)
    checks = check_import_resolution(clean_worktree_root)
    check = checks[0]
    assert check.status.value == "FAIL"
    assert "clouda_training" in check.message


def test_import_resolution_stale_site_packages(
    clean_worktree_root, tmp_path, monkeypatch
):
    site_packages = tmp_path / "site-packages"

    def fake_import(name, *a, **k):
        if name in env.CANONICAL_PACKAGES:
            return _fake_module(name, str(site_packages / name / "__init__.py"))
        return __import__(name)

    monkeypatch.setattr(env.importlib, "import_module", fake_import)
    checks = check_import_resolution(clean_worktree_root)
    check = checks[0]
    assert check.status.value == "WARN"
    assert check.details["actual_import_roots"] == [str(site_packages)]


def test_import_resolution_detects_lab_from_another_worktree(
    clean_worktree_root, tmp_path, monkeypatch
):
    other_tree = tmp_path / "OTHER_LAB_WORKTREE"

    def fake_import(name, *a, **k):
        root = other_tree if name == "clouda_lab" else clean_worktree_root
        if name in env.CANONICAL_PACKAGES or name == "clouda_lab":
            return _fake_module(name, str(root / name / "__init__.py"))
        return __import__(name)

    monkeypatch.setattr(env.importlib, "import_module", fake_import)
    check = check_import_resolution(clean_worktree_root)[0]
    assert check.status is env.DoctorStatus.FAIL
    assert str(other_tree) in check.details["actual_import_roots"]


# ---------------------------------------------------------------------------
# Phase 5 — dependency groups
# ---------------------------------------------------------------------------


def test_dependency_group_installed_and_missing():
    # PyYAML is installed in any environment able to run this repo's tests.
    checks = check_dependency_group("training", ("PyYAML==6.0.2",), required=True)
    assert len(checks) == 1
    check = checks[0]
    assert check.details["distribution"] == "PyYAML"
    assert check.details["requirement"] == "PyYAML==6.0.2"
    assert check.status is env.DoctorStatus.PASS

    missing = check_dependency_group(
        "worker", ("definitely-not-a-real-dist-xyz>=1.0",), required=False
    )
    assert missing[0].status is env.DoctorStatus.WARN
    assert 'pip install -e ".[worker]"' in missing[0].remediation


def test_dependency_required_missing_is_fail():
    checks = check_dependency_group(
        "core", ("definitely-not-a-real-dist-xyz",), required=True
    )
    assert checks[0].status is env.DoctorStatus.FAIL


def test_installed_version_missing_returns_none():
    assert installed_version("definitely-not-a-real-dist-xyz") is None


def test_dist_to_import_mapping():
    assert DIST_TO_IMPORT["Pillow"] == "PIL"
    assert DIST_TO_IMPORT["opencv-python-headless"] == "cv2"
    assert DIST_TO_IMPORT["PyYAML"] == "yaml"
    assert DIST_TO_IMPORT["WeasyPrint"] == "weasyprint"


def test_build_dependency_sections_uses_pyproject_extras(clean_worktree_root):
    (clean_worktree_root / "pyproject.toml").write_text(
        '[project]\nname="x"\n'
        "[project.optional-dependencies]\n"
        'training = ["PyYAML==6.0.2", "custom-extra-pkg>=1,<2"]\n',
        encoding="utf-8",
    )
    data = env._load_pyproject(clean_worktree_root)
    sections = env.build_dependency_sections(data)
    by_id = {s.id: s for s in sections}
    assert "dependencies-training" in by_id
    names = [c.details["distribution"] for c in by_id["dependencies-training"].checks]
    assert names == ["PyYAML", "custom-extra-pkg"]


def test_current_dependency_model_has_backend_groups(clean_worktree_root):
    data = env._load_pyproject(clean_worktree_root)
    section_ids = {section.id for section in env.build_dependency_sections(data)}
    assert {
        "dependencies-results",
        "dependencies-lab",
        "dependencies-training-data",
        "dependencies-training",
    } <= section_ids


def test_python_version_fail_when_outside_range(monkeypatch):
    # Plain tuple: python_supported() does tuple comparisons on version_info.
    monkeypatch.setattr(env.sys, "version_info", (3, 13, 0), raising=False)
    ok, requires = python_supported({})
    assert ok is False
    assert requires == ">=3.11,<3.12"


def test_python_version_supported_when_in_range(monkeypatch):
    monkeypatch.setattr(env.sys, "version_info", (3, 11, 5), raising=False)
    ok, _ = python_supported({})
    assert ok is True

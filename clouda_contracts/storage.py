from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import unquote, urlparse


class StorageSecurityError(ValueError):
    """Raised when a configured or resolved path crosses a storage boundary."""


_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")


def validate_relative_components(path: Path) -> None:
    """Reject names with ambiguous or dangerous cross-platform semantics."""
    for component in path.parts:
        normalized = unicodedata.normalize("NFC", component)
        if normalized != component:
            raise StorageSecurityError("Storage URI path must use NFC normalization.")
        if (
            not component
            or component in {".", ".."}
            or _CONTROL_CHARACTERS.search(component)
            or ":" in component
            or component.endswith((" ", "."))
            or len(component) > 255
        ):
            raise StorageSecurityError("Storage URI contains an unsafe path component.")
        stem = component.split(".", 1)[0].upper()
        if stem in _WINDOWS_RESERVED_NAMES:
            raise StorageSecurityError("Storage URI uses a reserved device name.")


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _inside(root: Path, candidate: Path) -> Path:
    resolved_root = _resolved(root)
    resolved_candidate = _resolved(candidate)
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise StorageSecurityError(
            f"Path crosses the configured storage boundary: {resolved_candidate}"
        ) from exc
    return resolved_candidate


def _default_state_home(environment: Mapping[str, str]) -> Path:
    configured = environment.get("CLOUDA_STATE_HOME", "").strip()
    if configured:
        return _resolved(Path(configured))
    if os.name == "nt":
        local = environment.get("LOCALAPPDATA", "").strip()
        base = Path(local) if local else Path.home() / "AppData" / "Local"
        return _resolved(base / "Clouda")
    xdg = environment.get("XDG_STATE_HOME", "").strip()
    base = Path(xdg) if xdg else Path.home() / ".local" / "state"
    return _resolved(base / "clouda")


@dataclass(frozen=True)
class StorageRoots:
    runtime_root: Path
    dataset_root: Path
    artifact_root: Path
    model_root: Path
    cache_root: Path
    database_path: Path
    dataset_manifest_database_path: Path
    read_only: bool = True

    @classmethod
    def from_env(
        cls,
        environment: Mapping[str, str] | None = None,
        *,
        read_only: bool = True,
        create: bool = False,
    ) -> "StorageRoots":
        values = os.environ if environment is None else environment
        home = _default_state_home(values)

        def configured(name: str, fallback: Path) -> Path:
            raw = values.get(name, "").strip()
            return _resolved(Path(raw)) if raw else _resolved(fallback)

        explicit_database = (
            values.get("CLOUDA_DATABASE_PATH", "").strip()
            or values.get("DATABASE_PATH", "").strip()
        )
        runtime_fallback = (
            _resolved(Path(explicit_database)).parent
            if explicit_database
            else home / "runtime"
        )
        runtime_root = configured("CLOUDA_RUNTIME_ROOT", runtime_fallback)
        dataset_root = configured("CLOUDA_DATASET_ROOT", home / "datasets")
        artifact_root = configured("CLOUDA_ARTIFACT_ROOT", home / "artifacts")
        model_root = configured("CLOUDA_MODEL_ROOT", home / "models")
        cache_root = configured("CLOUDA_CACHE_ROOT", home / "cache")
        database_path = configured(
            "CLOUDA_DATABASE_PATH",
            (
                Path(values.get("DATABASE_PATH", "").strip())
                if values.get("DATABASE_PATH", "").strip()
                else runtime_root / "databases" / "clouda.sqlite3"
            ),
        )
        manifest_database = configured(
            "CLOUDA_DATASET_MANIFEST_DATABASE_PATH",
            dataset_root / "manifests" / "dataset_manifest.sqlite3",
        )
        roots = cls(
            runtime_root=runtime_root,
            dataset_root=dataset_root,
            artifact_root=artifact_root,
            model_root=model_root,
            cache_root=cache_root,
            database_path=database_path,
            dataset_manifest_database_path=manifest_database,
            read_only=read_only,
        )
        roots.validate()
        if create:
            roots.ensure_directories()
        return roots

    def validate(self) -> None:
        roots = [
            self.runtime_root,
            self.dataset_root,
            self.artifact_root,
            self.model_root,
            self.cache_root,
        ]
        normalized = [_resolved(path) for path in roots]
        if len(set(normalized)) != len(normalized):
            raise StorageSecurityError("Storage roots must be distinct.")
        _inside(self.runtime_root, self.database_path)
        _inside(self.dataset_root, self.dataset_manifest_database_path)

    def ensure_directories(self) -> None:
        if self.read_only:
            raise PermissionError("Cannot create storage roots in read-only mode.")
        for path in (
            self.runtime_root,
            self.dataset_root,
            self.artifact_root,
            self.model_root,
            self.cache_root,
            self.database_path.parent,
            self.dataset_manifest_database_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def root_for_scheme(self, scheme: str) -> Path:
        roots = {
            "runtime": self.runtime_root,
            "dataset": self.dataset_root,
            "artifact": self.artifact_root,
            "model": self.model_root,
            "cache": self.cache_root,
        }
        try:
            return roots[scheme]
        except KeyError as exc:
            raise StorageSecurityError(
                f"Unsupported storage URI scheme: {scheme}"
            ) from exc

    def resolve_uri(self, value: str) -> Path:
        parsed = urlparse(value)
        if parsed.scheme not in {"runtime", "dataset", "artifact", "model", "cache"}:
            raise StorageSecurityError(f"Unsupported storage URI: {value}")
        if parsed.params or parsed.query or parsed.fragment:
            raise StorageSecurityError(
                "Storage URIs cannot contain params, query, or fragment."
            )
        relative_text = "/".join(
            part for part in (parsed.netloc, parsed.path.lstrip("/")) if part
        )
        relative = Path(unquote(relative_text.replace("\\", "/")))
        if relative.is_absolute() or ".." in relative.parts:
            raise StorageSecurityError("Storage URI contains an unsafe path.")
        validate_relative_components(relative)
        root = self.root_for_scheme(parsed.scheme)
        return _inside(root, root / relative)

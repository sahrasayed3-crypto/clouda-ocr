"""Safe artifact resolution: logical URI -> local path.

Artifact references persisted in canonical records are portable
(``dataset://...``, ``artifact://...``). Resolution happens only against
explicitly configured roots; traversal outside a root (``..``, absolute paths,
Windows drives, UNC) is rejected. Machine-local absolute paths are never
embedded in canonical records — when a source carries one, it is quarantined
under ``source_private`` (see :mod:`clouda_data.results.identity`).
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

from .identity import ArtifactRef


class ArtifactResolutionError(ValueError):
    """Raised when an artifact cannot be resolved safely."""


class ArtifactResolver:
    """Resolve portable artifact URIs against configured roots."""

    SCHEMES = ("dataset", "artifact", "model", "runtime", "cache")

    def __init__(
        self,
        roots: dict[str, str | Path],
        *,
        must_exist: bool = False,
    ) -> None:
        self._roots: dict[str, Path] = {}
        for scheme, root in roots.items():
            scheme = scheme.strip().lower()
            if scheme not in self.SCHEMES:
                raise ArtifactResolutionError(f"Unsupported scheme: {scheme}")
            self._roots[scheme] = Path(root).expanduser().resolve(strict=False)
        self._must_exist = must_exist

    def resolve_uri(self, uri: str) -> Path:
        parsed = urlparse(uri)
        scheme = (parsed.scheme or "").lower()
        if scheme not in self._roots:
            raise ArtifactResolutionError(
                f"No root configured for scheme {scheme!r}: {uri}"
            )
        if parsed.params or parsed.query or parsed.fragment:
            raise ArtifactResolutionError(f"URI contains params/query/fragment: {uri}")
        relative_text = "/".join(
            part for part in (parsed.netloc, parsed.path.lstrip("/")) if part
        ).replace("\\", "/")
        if not parsed.netloc and parsed.path.startswith("/"):
            # ``scheme:///abs`` is malformed: artifact URIs are always relative
            # (a single leading segment lands in netloc, not path).
            raise ArtifactResolutionError(f"Artifact URI must be relative: {uri}")
        relative = PurePosixPath(unquote(relative_text))
        if relative.is_absolute() or not relative.parts:
            raise ArtifactResolutionError(f"Artifact URI must be relative: {uri}")
        if any(part in ("..", "") for part in relative.parts):
            raise ArtifactResolutionError(f"Unsafe path in artifact URI: {uri}")
        if any(":" in part for part in relative.parts):
            raise ArtifactResolutionError(f"Windows drive in artifact URI: {uri}")
        root = self._roots[scheme]
        candidate = (root / Path(*relative.parts)).resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ArtifactResolutionError(
                f"Path crosses the storage boundary: {uri}"
            ) from exc
        if self._must_exist and not candidate.exists():
            raise ArtifactResolutionError(f"Artifact not found locally: {uri}")
        return candidate

    def resolve(self, artifact: ArtifactRef) -> Path:
        return self.resolve_uri(artifact.uri)


def resolver_from_env(
    environment: dict[str, str] | None = None, *, must_exist: bool = False
) -> ArtifactResolver:
    """Build a resolver from CLOUDA_* root environment variables."""

    import os

    values = os.environ if environment is None else environment
    roots: dict[str, str | Path] = {}
    for scheme in ArtifactResolver.SCHEMES:
        raw = values.get(f"CLOUDA_{scheme.upper()}_ROOT", "").strip()
        if raw:
            roots[scheme] = raw
    if not roots:
        raise ArtifactResolutionError(
            "No CLOUDA_*_ROOT environment variables configured."
        )
    return ArtifactResolver(roots, must_exist=must_exist)

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import socket
import ssl
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from tarfile import is_tarfile
from typing import Any
from urllib.parse import urljoin, urlparse

from .registry import assert_source_download_allowed, get_source

TWO_GB = 2 * 1024 * 1024 * 1024
DEFAULT_SAMPLE_LIMIT = 100 * 1024 * 1024
PRIVATE_DOWNLOAD_ENV = "CLOUDA_ALLOW_PRIVATE_DOWNLOADS"
INSECURE_DOWNLOAD_ENV = "CLOUDA_ALLOW_INSECURE_DOWNLOADS"


@dataclass(frozen=True)
class DownloadedFile:
    url: str
    path: str
    size_bytes: int
    checksum_sha256: str
    duplicate_of: str | None = None
    archive_valid: bool | None = None


@dataclass(frozen=True)
class DownloadResult:
    source_id: str
    ok: bool
    dry_run: bool
    files: list[DownloadedFile] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def safe_filename(name: str) -> str:
    value = Path(urlparse(name).path).name or name
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return safe[:180] or "downloaded_file"


@lru_cache(maxsize=1)
def trusted_ssl_context() -> ssl.SSLContext | None:
    """Use the Windows certificate store when Python's bundled roots are incomplete."""
    if not hasattr(ssl, "enum_certificates"):
        return None
    pem_chunks: list[str] = []
    for store_name in ("ROOT", "CA"):
        try:
            certificates = ssl.enum_certificates(store_name)
        except Exception:
            continue
        for certificate, encoding, _trust in certificates:
            if encoding == "x509_asn":
                pem_chunks.append(ssl.DER_cert_to_PEM_cert(certificate))
    if not pem_chunks:
        return None
    cafile = Path(tempfile.gettempdir()) / "arabic_ocr_windows_ca.pem"
    cafile.write_text("\n".join(pem_chunks), encoding="ascii")
    return ssl.create_default_context(cafile=str(cafile))


def _enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def validate_download_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise PermissionError("Dataset downloads require an HTTP(S) URL")
    if parsed.username or parsed.password or parsed.fragment:
        raise PermissionError("Dataset download URL contains forbidden components")
    if parsed.scheme != "https" and not _enabled(INSECURE_DOWNLOAD_ENV):
        raise PermissionError("Plain HTTP dataset downloads require explicit opt-in")
    allow_private = _enabled(PRIVATE_DOWNLOAD_ENV)
    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(
                parsed.hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
            )
        }
    except socket.gaierror as exc:
        raise PermissionError(
            "Dataset download host cannot be resolved safely"
        ) from exc
    if not addresses:
        raise PermissionError("Dataset download host has no resolved addresses")
    for address in addresses:
        value = ipaddress.ip_address(str(address).split("%", 1)[0])
        if (
            value.is_private
            or value.is_loopback
            or value.is_link_local
            or value.is_multicast
            or value.is_reserved
            or value.is_unspecified
        ) and not allow_private:
            raise PermissionError("Private or local dataset destinations are blocked")
    return url


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urljoin(req.full_url, newurl)
        validate_download_url(target)
        return super().redirect_request(req, fp, code, msg, headers, target)


def _urlopen(request: urllib.request.Request, *, timeout: int):
    validate_download_url(request.full_url)
    context = trusted_ssl_context()
    handlers: list[urllib.request.BaseHandler] = [
        urllib.request.ProxyHandler({}),
        _SafeRedirectHandler(),
    ]
    if context is not None:
        handlers.append(urllib.request.HTTPSHandler(context=context))
    opener = urllib.request.build_opener(*handlers)
    return opener.open(request, timeout=timeout)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_length(url: str, timeout: int = 20) -> int | None:
    validate_download_url(url)
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "arabic-ocr-dataset-prep/0.1"}
    )
    try:
        with _urlopen(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None


def validate_archive(path: Path) -> bool | None:
    suffix = path.suffix.lower()
    if suffix == ".zip":
        try:
            with zipfile.ZipFile(path) as archive:
                return archive.testzip() is None
        except zipfile.BadZipFile:
            return False
    if suffix in {".tar", ".gz", ".bz2", ".xz"}:
        return is_tarfile(path)
    return None


def disk_free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return usage.free


def _existing_checksums(root: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for manifest_path in (root / "data/manifests/download_manifests").glob("*.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in payload.get("files", []):
            checksums[item["checksum_sha256"]] = item["path"]
    return checksums


def download_http(
    url: str,
    destination: Path,
    *,
    max_bytes: int,
    retries: int = 3,
    backoff_seconds: float = 0.25,
    expected_sha256: str | None = None,
) -> DownloadedFile:
    validate_download_url(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    remote_size = content_length(url)
    if remote_size is not None and remote_size > max_bytes:
        raise PermissionError(f"Remote file exceeds limit: {remote_size} > {max_bytes}")
    downloaded = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "arabic-ocr-dataset-prep/0.1"}
    if downloaded:
        headers["Range"] = f"bytes={downloaded}-"
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=headers)
            with _urlopen(request, timeout=30) as response:
                append = bool(downloaded and getattr(response, "status", None) == 206)
                with partial.open("ab" if append else "wb") as handle:
                    if downloaded and not append:
                        downloaded = 0
                    while True:
                        chunk = response.read(1024 * 256)
                        if not chunk:
                            break
                        handle.write(chunk)
                        if partial.stat().st_size > max_bytes:
                            raise PermissionError("Download exceeded byte limit.")
            break
        except (urllib.error.URLError, TimeoutError, PermissionError):
            if attempt == retries - 1:
                raise
            time.sleep(backoff_seconds * (2**attempt))
    actual_size = partial.stat().st_size
    if actual_size > max_bytes:
        raise PermissionError("Downloaded file exceeds byte limit.")
    actual_sha = sha256_file(partial)
    if expected_sha256 and expected_sha256 != actual_sha:
        raise ValueError("Downloaded checksum mismatch.")
    partial.replace(destination)
    return DownloadedFile(
        url=url,
        path=str(destination),
        size_bytes=actual_size,
        checksum_sha256=actual_sha,
        archive_valid=validate_archive(destination),
    )


def _asset_url(source: dict[str, Any], asset: dict[str, Any]) -> str:
    url = asset.get("url")
    if url:
        return url
    method = source.get("download_method")
    if method == "huggingface":
        repo = source["repository_url"].split("/datasets/")[-1]
        path = asset["path"]
        return f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"
    if method == "internet_archive":
        identifier = source["metadata"].get("internet_archive_identifier")
        path = asset["path"]
        return f"https://archive.org/download/{identifier}/{path}"
    raise ValueError("Asset must include url for this download method.")


def download_dataset_sample(
    source_id: str,
    *,
    project_root: Path,
    registry_path: str | Path = "data/manifests/dataset_registry.json",
    max_bytes: int = DEFAULT_SAMPLE_LIMIT,
    dry_run: bool = False,
) -> DownloadResult:
    if max_bytes > TWO_GB:
        raise PermissionError("Hard safety limit: max_bytes cannot exceed 2 GB.")
    source = get_source(source_id, project_root / registry_path)
    assert_source_download_allowed(source, full_dataset=False, max_bytes=max_bytes)
    assets = source.get("sample_assets", [])
    if not assets:
        return DownloadResult(source_id, False, dry_run, issues=["no_sample_assets"])
    estimated = sum(asset.get("size_bytes") or 0 for asset in assets)
    if estimated and estimated > max_bytes:
        raise PermissionError("Sample asset estimates exceed byte limit.")
    destination_root = project_root / "data/downloads" / source_id
    if dry_run:
        return DownloadResult(source_id, True, True)
    if disk_free_bytes(project_root) < max_bytes:
        raise OSError("Insufficient free disk space for requested limit.")
    existing = _existing_checksums(project_root)
    files: list[DownloadedFile] = []
    issues: list[str] = []
    for asset in assets:
        url = _asset_url(source, asset)
        destination = destination_root / safe_filename(asset.get("filename") or url)
        try:
            downloaded = download_http(
                url,
                destination,
                max_bytes=max_bytes,
                expected_sha256=asset.get("sha256"),
            )
            duplicate_of = existing.get(downloaded.checksum_sha256)
            files.append(
                DownloadedFile(**{**asdict(downloaded), "duplicate_of": duplicate_of})
            )
            existing.setdefault(downloaded.checksum_sha256, downloaded.path)
        except Exception as exc:
            issues.append(f"{url}: {exc}")
            quarantine = project_root / "data/quarantine/downloads" / source_id
            quarantine.mkdir(parents=True, exist_ok=True)
            part = destination.with_suffix(destination.suffix + ".part")
            if part.exists():
                shutil.move(str(part), quarantine / part.name)
    result = DownloadResult(source_id, not issues, False, files, issues)
    if result.ok and source.get("metadata", {}).get("supports_ingestion_sample"):
        create_ingestion_manifest_for_download(source, result, project_root)
    write_download_manifest(result, project_root)
    write_license_metadata(source, project_root)
    return result


def create_ingestion_manifest_for_download(
    source: dict[str, Any], result: DownloadResult, project_root: Path
) -> Path | None:
    from clouda_data.ground_truth.checksums import sha256_text
    from clouda_data.ingestion.file_inspection import text_from_ground_truth

    xml_file = next(
        (item for item in result.files if item.path.lower().endswith(".xml")), None
    )
    if xml_file is None:
        return None
    xml_path = Path(xml_file.path)
    clean_text = text_from_ground_truth(xml_path, "page_xml")
    manifest = {
        "documents": [
            {
                "document_id": source["source_id"],
                "source_path": xml_path.name,
                "source_type": "page_xml",
                "language": "ar",
                "source_license": source["license"],
                "title": source["name"],
            }
        ],
        "pages": [
            {
                "document_id": source["source_id"],
                "page_id": f"{source['source_id']}_sample_001",
                "page_number": 1,
                "source_path": xml_path.name,
                "source_type": "page_xml",
                "language": "ar",
                "ground_truth_path": xml_path.name,
                "text_checksum": sha256_text(clean_text),
                "source_license": source["license"],
                "reading_order": [],
            }
        ],
    }
    out = xml_path.parent / "source_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def write_download_manifest(result: DownloadResult, project_root: Path) -> Path:
    out = (
        project_root / "data/manifests/download_manifests" / f"{result.source_id}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out


def write_license_metadata(source: dict[str, Any], project_root: Path) -> Path:
    out = (
        project_root / "data/downloads" / source["source_id"] / "LICENSE_METADATA.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_id": source["source_id"],
        "name": source["name"],
        "license": source["license"],
        "classification": source["classification"],
        "commercial_use_status": source["commercial_use_status"],
        "redistribution_status": source["redistribution_status"],
        "attribution_requirements": source["attribution_requirements"],
        "verification_date": source["verification_date"],
        "official_url": source["official_url"],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def verify_download(source_id: str, *, project_root: Path) -> DownloadResult:
    manifest_path = (
        project_root / "data/manifests/download_manifests" / f"{source_id}.json"
    )
    if not manifest_path.exists():
        return DownloadResult(
            source_id, False, False, issues=["missing_download_manifest"]
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    files: list[DownloadedFile] = []
    issues: list[str] = []
    for item in payload.get("files", []):
        path = Path(item["path"])
        if not path.exists():
            issues.append(f"missing_file:{path}")
            continue
        actual = sha256_file(path)
        if actual != item["checksum_sha256"]:
            issues.append(f"checksum_mismatch:{path}")
        files.append(DownloadedFile(**item))
    return DownloadResult(source_id, not issues, False, files, issues)


def result_to_dict(result: DownloadResult) -> dict[str, Any]:
    return asdict(result)

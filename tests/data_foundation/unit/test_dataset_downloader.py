from __future__ import annotations

import functools
import json
import os
import tempfile
import threading
import unittest
from unittest import mock
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from clouda_data.datasets.downloader import (
    DownloadCancelled,
    download_http,
    download_dataset_sample,
    safe_filename,
    verify_download,
)


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


class LocalServer:
    def __init__(self, root: Path) -> None:
        handler = functools.partial(QuietHandler, directory=str(root))
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host = str(self.server.server_address[0])
        port = int(self.server.server_address[1])
        return f"http://{host}:{port}"

    def __enter__(self) -> "LocalServer":
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def create_registry(path: Path, url: str, classification: str = "approved") -> None:
    payload = {
        "sources": [
            {
                "source_id": "tiny_source",
                "name": "Tiny Source",
                "classification": classification,
                "license": "Apache-2.0",
                "license_verified": True,
                "commercial_use_status": "allowed",
                "redistribution_status": "allowed",
                "attribution_requirements": "preserve notice",
                "sample_size_bytes": 1024,
                "download_method": "https",
                "requires_authentication": False,
                "requires_form": False,
                "requires_account": False,
                "sample_assets": [
                    {"filename": "sample.txt", "url": url, "size_bytes": 1024}
                ],
                "metadata": {},
                "official_url": url,
                "risk_level": "low",
                "verification_date": "2026-07-23",
            }
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class DatasetDownloaderTests(unittest.TestCase):
    def test_safe_filename(self) -> None:
        self.assertEqual(safe_filename("https://example.com/a b.txt"), "a_b.txt")

    def test_download_sample_and_verify_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server_root = root / "server"
            server_root.mkdir()
            (server_root / "sample.txt").write_text(
                "tiny Arabic OCR metadata", encoding="utf-8"
            )
            with LocalServer(server_root) as server:
                registry = root / "data/manifests/dataset_registry.json"
                create_registry(registry, f"{server.url}/sample.txt")
                with mock.patch.dict(
                    os.environ,
                    {
                        "CLOUDA_ALLOW_PRIVATE_DOWNLOADS": "true",
                        "CLOUDA_ALLOW_INSECURE_DOWNLOADS": "true",
                        "CLOUDA_ALLOW_UNPINNED_DATASET_DOWNLOADS": "true",
                    },
                ):
                    result = download_dataset_sample(
                        "tiny_source", project_root=root, max_bytes=1024 * 1024
                    )
            self.assertTrue(result.ok, result.issues)
            self.assertTrue((root / "data/downloads/tiny_source/sample.txt").exists())
            verified = verify_download("tiny_source", project_root=root)
            self.assertTrue(verified.ok, verified.issues)

    def test_blocks_research_only_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = root / "data/manifests/dataset_registry.json"
            create_registry(
                registry, "http://127.0.0.1/sample.txt", classification="research_only"
            )
            with self.assertRaises(PermissionError):
                download_dataset_sample(
                    "tiny_source", project_root=root, max_bytes=1024 * 1024
                )

    def test_blocks_over_limit_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = root / "data/manifests/dataset_registry.json"
            create_registry(registry, "http://127.0.0.1/sample.txt")
            with self.assertRaises(PermissionError):
                download_dataset_sample(
                    "tiny_source", project_root=root, max_bytes=3 * 1024 * 1024 * 1024
                )

    def test_download_reports_progress_and_cancelled_partial_can_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server_root = root / "server"
            server_root.mkdir()
            payload = b"clouda-progress" * 100_000
            (server_root / "large.bin").write_bytes(payload)
            destination = root / "downloads" / "large.bin"
            progress: list[tuple[int, int | None]] = []
            cancelled = False

            def on_progress(completed: int, total: int | None) -> None:
                nonlocal cancelled
                progress.append((completed, total))
                if completed:
                    cancelled = True

            with (
                LocalServer(server_root) as server,
                mock.patch.dict(
                    os.environ,
                    {
                        "CLOUDA_ALLOW_PRIVATE_DOWNLOADS": "true",
                        "CLOUDA_ALLOW_INSECURE_DOWNLOADS": "true",
                        "CLOUDA_ALLOW_UNPINNED_DATASET_DOWNLOADS": "true",
                    },
                ),
            ):
                with self.assertRaises(DownloadCancelled):
                    download_http(
                        f"{server.url}/large.bin",
                        destination,
                        max_bytes=len(payload) + 1,
                        progress_callback=on_progress,
                        cancellation_check=lambda: cancelled,
                    )
                partial = destination.with_suffix(".bin.part")
                self.assertTrue(partial.is_file())
                self.assertGreater(partial.stat().st_size, 0)

                resumed_progress: list[tuple[int, int | None]] = []
                result = download_http(
                    f"{server.url}/large.bin",
                    destination,
                    max_bytes=len(payload) + 1,
                    progress_callback=lambda completed, total: resumed_progress.append(
                        (completed, total)
                    ),
                )

            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(result.size_bytes, len(payload))
            self.assertTrue(progress)
            self.assertEqual(resumed_progress[-1], (len(payload), len(payload)))

    def test_sample_download_propagates_aggregate_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server_root = root / "server"
            server_root.mkdir()
            payload = b"canonical sample"
            (server_root / "sample.txt").write_bytes(payload)
            progress: list[tuple[int, int | None]] = []
            with LocalServer(server_root) as server:
                registry = root / "data/manifests/dataset_registry.json"
                create_registry(registry, f"{server.url}/sample.txt")
                with mock.patch.dict(
                    os.environ,
                    {
                        "CLOUDA_ALLOW_PRIVATE_DOWNLOADS": "true",
                        "CLOUDA_ALLOW_INSECURE_DOWNLOADS": "true",
                        "CLOUDA_ALLOW_UNPINNED_DATASET_DOWNLOADS": "true",
                    },
                ):
                    result = download_dataset_sample(
                        "tiny_source",
                        project_root=root,
                        max_bytes=1024 * 1024,
                        progress_callback=lambda completed, total: progress.append(
                            (completed, total)
                        ),
                    )

            self.assertTrue(result.ok)
            self.assertEqual(progress[-1], (len(payload), len(payload)))


if __name__ == "__main__":
    unittest.main()

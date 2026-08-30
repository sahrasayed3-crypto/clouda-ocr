from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shlex
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from clouda_contracts.storage import StorageRoots

from .engines import OCRResult, OCR_STATUS_FAILED, OCR_STATUS_SUCCEEDED

TASK_PROMPTS = {
    "plain_arabic": "استخرج النص العربي فقط كما يظهر في الصورة.",
    "reading_order": "استخرج النص مع الحفاظ على ترتيب القراءة.",
    "markdown": "استخرج المستند بصيغة Markdown مع الحفاظ على البنية.",
    "bounding_boxes": "استخرج الكلمات وإحداثيات الصناديق المحيطة بها.",
    "layout": "استخرج النص ومناطق التخطيط وترتيب القراءة.",
    "mixed_arabic_english": "Extract Arabic and English text while preserving direction and reading order.",
}


@dataclass(frozen=True)
class LocalOCRConfig:
    enabled: bool = False
    engine: str = "none"
    model_path: str = ""
    processor_path: str = ""
    device: str = "cpu"
    dtype: str = "float32"
    max_image_pixels: int = 40_000_000
    max_new_tokens: int = 2048
    timeout_seconds: int = 120
    batch_size: int = 1
    endpoint: str = "http://127.0.0.1:8001/v1"
    task: str = "markdown"

    @classmethod
    def from_env(cls) -> "LocalOCRConfig":
        env = os.environ
        return cls(
            enabled=env.get("CLOUDA_LOCAL_OCR_ENABLED", "").lower()
            in {"1", "true", "yes", "on"},
            engine=env.get("CLOUDA_LOCAL_OCR_ENGINE", "none").strip().lower(),
            model_path=env.get("CLOUDA_LOCAL_OCR_MODEL_PATH", "").strip(),
            processor_path=env.get("CLOUDA_LOCAL_OCR_PROCESSOR_PATH", "").strip(),
            device=env.get("CLOUDA_LOCAL_OCR_DEVICE", "cpu").strip(),
            dtype=env.get("CLOUDA_LOCAL_OCR_DTYPE", "float32").strip(),
            max_image_pixels=int(
                env.get("CLOUDA_LOCAL_OCR_MAX_IMAGE_PIXELS", "40000000")
            ),
            max_new_tokens=int(env.get("CLOUDA_LOCAL_OCR_MAX_NEW_TOKENS", "2048")),
            timeout_seconds=int(env.get("CLOUDA_LOCAL_OCR_TIMEOUT_SECONDS", "120")),
            batch_size=int(env.get("CLOUDA_LOCAL_OCR_BATCH_SIZE", "1")),
            endpoint=env.get(
                "CLOUDA_LOCAL_OCR_ENDPOINT", "http://127.0.0.1:8001/v1"
            ).strip(),
            task=env.get("CLOUDA_LOCAL_OCR_TASK", "markdown").strip(),
        )

    def validate(self) -> None:
        if self.engine not in {
            "none",
            "mock",
            "local_http",
            "openai_compatible",
            "command_line",
            "transformers",
            "qwen_vl",
        }:
            raise ValueError("Unsupported local OCR engine")
        if self.task not in TASK_PROMPTS:
            raise ValueError("Unsupported OCR task")
        if self.device != "cpu" and not self.model_path:
            raise ValueError("Non-CPU devices require an explicit local model path")
        if not 1 <= self.batch_size <= 16:
            raise ValueError("batch size must be between 1 and 16")
        if not 1 <= self.timeout_seconds <= 3600:
            raise ValueError("timeout must be between 1 and 3600 seconds")


def _validate_image(image_bytes: bytes, max_pixels: int) -> None:
    with Image.open(io.BytesIO(image_bytes)) as image:
        if image.width * image.height > max_pixels:
            raise ValueError("OCR image exceeds configured pixel limit")
        image.verify()


class MockOCRProvider:
    def __init__(self, text: str = "نص عربي تجريبي", confidence: float = 0.99) -> None:
        self.text = text
        self.confidence = confidence

    def available(self) -> bool:
        return os.getenv("CLOUDA_ALLOW_MOCK_OCR", "").lower() in {"1", "true", "yes"}

    def extract_page(self, *, image_bytes: bytes, page_no: int) -> OCRResult:
        del image_bytes
        return OCRResult(
            engine_name="mock_local_ocr",
            model_name="synthetic-test-only",
            status=OCR_STATUS_SUCCEEDED,
            text=self.text,
            confidence=self.confidence,
            metadata={
                "page_no": page_no,
                "model_revision": "test-fixture-v1",
                "synthetic_mock": True,
            },
        )


class LocalHTTPProvider:
    def __init__(
        self, config: LocalOCRConfig, *, openai_compatible: bool = False
    ) -> None:
        self.config = config
        self.openai_compatible = openai_compatible
        parsed = urllib.parse.urlparse(config.endpoint)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Local OCR endpoint must use HTTP(S)")
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"} and os.getenv(
            "CLOUDA_LOCAL_OCR_ALLOW_REMOTE_ENDPOINT", ""
        ).lower() not in {"1", "true", "yes"}:
            raise ValueError("Remote OCR endpoints require explicit opt-in")

    def available(self) -> bool:
        return self.config.enabled

    def extract_page(self, *, image_bytes: bytes, page_no: int) -> OCRResult:
        _validate_image(image_bytes, self.config.max_image_pixels)
        started = time.perf_counter()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        if self.openai_compatible:
            body: dict[str, Any] = {
                "model": self.config.model_path or "local-model",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": TASK_PROMPTS[self.config.task]},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encoded}"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": self.config.max_new_tokens,
            }
            url = self.config.endpoint.rstrip("/") + "/chat/completions"
        else:
            body = {
                "image_base64": encoded,
                "prompt": TASK_PROMPTS[self.config.task],
                "page_no": page_no,
            }
            url = self.config.endpoint
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as response:
                if response.length and response.length > 10 * 1024 * 1024:
                    raise ValueError("OCR response exceeds size limit")
                payload = json.loads(response.read(10 * 1024 * 1024).decode())
            text = (
                payload.get("choices", [{}])[0].get("message", {}).get("content", "")
                if self.openai_compatible
                else payload.get("text", "")
            )
            confidence = float(payload.get("confidence", 0.5))
            if not str(text).strip():
                raise ValueError("OCR endpoint returned empty text")
            return OCRResult(
                engine_name=(
                    "openai_compatible_local"
                    if self.openai_compatible
                    else "local_http"
                ),
                model_name=self.config.model_path or "local-endpoint",
                status=OCR_STATUS_SUCCEEDED,
                text=str(text),
                confidence=max(0.0, min(1.0, confidence)),
                processing_time=time.perf_counter() - started,
                metadata={"page_no": page_no},
            )
        except (urllib.error.URLError, ValueError, json.JSONDecodeError) as exc:
            return OCRResult(
                engine_name="local_http",
                status=OCR_STATUS_FAILED,
                confidence=None,
                processing_time=time.perf_counter() - started,
                error_message=f"{type(exc).__name__}: {exc}",
            )


class CommandLineOCRProvider:
    def __init__(self, config: LocalOCRConfig) -> None:
        self.config = config
        command = os.getenv("CLOUDA_LOCAL_OCR_COMMAND", "").strip()
        if command.startswith("["):
            value = json.loads(command)
            if not isinstance(value, list) or not all(
                isinstance(item, str) and item for item in value
            ):
                raise ValueError("OCR command JSON must be a string array")
            self.command = value
        else:
            self.command = (
                shlex.split(command, posix=os.name != "nt") if command else []
            )
        if self.command and not Path(self.command[0]).expanduser().is_absolute():
            raise ValueError("OCR executable path must be absolute")
        if self.command:
            executable = Path(self.command[0]).expanduser().resolve()
            allowed = {
                str(Path(item).expanduser().resolve()).casefold()
                for item in os.getenv("CLOUDA_LOCAL_OCR_ALLOWED_EXECUTABLES", "").split(
                    os.pathsep
                )
                if item.strip()
            }
            if not allowed or str(executable).casefold() not in allowed:
                raise PermissionError(
                    "OCR executable is not in CLOUDA_LOCAL_OCR_ALLOWED_EXECUTABLES"
                )
            if executable.is_symlink() or not executable.is_file():
                raise PermissionError("OCR executable must be a regular local file")
            self.command[0] = str(executable)

    def available(self) -> bool:
        return (
            self.config.enabled
            and bool(self.command)
            and Path(self.command[0]).is_file()
        )

    def extract_page(self, *, image_bytes: bytes, page_no: int) -> OCRResult:
        _validate_image(image_bytes, self.config.max_image_pixels)
        if not self.available():
            return OCRResult(
                engine_name="command_line_ocr",
                status=OCR_STATUS_FAILED,
                error_message="Command-line OCR is unavailable",
            )
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="clouda-ocr-") as directory:
            image = Path(directory) / f"page-{page_no}.png"
            image.write_bytes(image_bytes)
            safe_environment = {
                key: value
                for key, value in os.environ.items()
                if key.upper()
                in {
                    "PATH",
                    "PATHEXT",
                    "SYSTEMDRIVE",
                    "SYSTEMROOT",
                    "TEMP",
                    "TMP",
                    "WINDIR",
                }
            }
            safe_environment["PYTHONIOENCODING"] = "utf-8"
            stdout_path = Path(directory) / "stdout.txt"
            stderr_path = Path(directory) / "stderr.txt"
            with stdout_path.open("w+b") as stdout, stderr_path.open("w+b") as stderr:
                completed = subprocess.run(
                    [*self.command, str(image)],
                    stdout=stdout,
                    stderr=stderr,
                    timeout=self.config.timeout_seconds,
                    shell=False,
                    check=False,
                    cwd=directory,
                    env=safe_environment,
                    close_fds=True,
                )
            max_output = 4 * 1024 * 1024
            if stdout_path.stat().st_size > max_output:
                return OCRResult(
                    engine_name="command_line_ocr",
                    status=OCR_STATUS_FAILED,
                    processing_time=time.perf_counter() - started,
                    error_message="OCR process output exceeded the byte limit",
                )
            text = stdout_path.read_text(encoding="utf-8", errors="replace").strip()
        if completed.returncode != 0 or not text:
            return OCRResult(
                engine_name="command_line_ocr",
                status=OCR_STATUS_FAILED,
                processing_time=time.perf_counter() - started,
                error_message=f"OCR process failed with exit code {completed.returncode}",
            )
        return OCRResult(
            engine_name="command_line_ocr",
            model_name=Path(self.command[0]).name,
            status=OCR_STATUS_SUCCEEDED,
            text=text,
            confidence=0.5,
            processing_time=time.perf_counter() - started,
            metadata={"page_no": page_no},
        )


class TransformersVisionLanguageProvider:
    def __init__(self, config: LocalOCRConfig) -> None:
        self.config = config
        self.model_path = (
            Path(config.model_path).expanduser().resolve()
            if config.model_path
            else None
        )
        self.processor_path = (
            Path(config.processor_path).expanduser().resolve()
            if config.processor_path
            else self.model_path
        )
        roots = StorageRoots.from_env()
        for path in (self.model_path, self.processor_path):
            if path is None:
                continue
            try:
                path.relative_to(roots.model_root.resolve())
            except ValueError as exc:
                raise PermissionError(
                    "Local OCR model paths must stay inside model://"
                ) from exc
            if path.is_symlink():
                raise PermissionError("Symbolic-link model paths are not accepted")
        self._model: Any = None
        self._processor: Any = None

    def available(self) -> bool:
        return bool(
            self.config.enabled and self.model_path and self.model_path.is_dir()
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.available():
            raise RuntimeError("A local model directory is required")
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            str(self.processor_path), local_files_only=True, trust_remote_code=False
        )
        self._model = AutoModelForVision2Seq.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            weights_only=True,
            device_map=None if self.config.device == "cpu" else self.config.device,
        )

    def extract_page(self, *, image_bytes: bytes, page_no: int) -> OCRResult:
        _validate_image(image_bytes, self.config.max_image_pixels)
        started = time.perf_counter()
        try:
            self._load()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            inputs = self._processor(
                images=image,
                text=TASK_PROMPTS[self.config.task],
                return_tensors="pt",
            )
            output = self._model.generate(
                **inputs, max_new_tokens=self.config.max_new_tokens
            )
            text = self._processor.batch_decode(output, skip_special_tokens=True)[
                0
            ].strip()
            if not text:
                raise ValueError("Local model returned empty text")
            return OCRResult(
                engine_name="transformers_local",
                model_name=self.model_path.name if self.model_path else None,
                status=OCR_STATUS_SUCCEEDED,
                text=text,
                confidence=0.5,
                processing_time=time.perf_counter() - started,
                metadata={
                    "page_no": page_no,
                    "local_files_only": True,
                    "trust_remote_code": False,
                },
            )
        except Exception as exc:
            return OCRResult(
                engine_name="transformers_local",
                status=OCR_STATUS_FAILED,
                processing_time=time.perf_counter() - started,
                error_message=f"{type(exc).__name__}: {exc}",
            )


def provider_from_config(config: LocalOCRConfig):
    config.validate()
    if config.engine == "mock":
        return MockOCRProvider(
            os.getenv("CLOUDA_MOCK_OCR_TEXT", "نص عربي تجريبي"),
            float(os.getenv("CLOUDA_MOCK_OCR_CONFIDENCE", "0.99")),
        )
    if config.engine == "local_http":
        return LocalHTTPProvider(config)
    if config.engine == "openai_compatible":
        return LocalHTTPProvider(config, openai_compatible=True)
    if config.engine == "command_line":
        return CommandLineOCRProvider(config)
    if config.engine in {"transformers", "qwen_vl"}:
        return TransformersVisionLanguageProvider(config)
    return None


def model_revision(config: LocalOCRConfig) -> str:
    if config.engine == "mock":
        return "test-fixture-v1"
    if not config.model_path:
        return "unresolved"
    path = Path(config.model_path).expanduser()
    if not path.exists():
        return "unresolved"
    material = f"{path.resolve()}:{path.stat().st_mtime_ns}".encode()
    return hashlib.sha256(material).hexdigest()[:16]

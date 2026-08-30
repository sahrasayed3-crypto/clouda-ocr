from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
AWS_KEY = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
OPENAI_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")
GENERIC_SECRET_PATTERNS = (
    re.compile(r"""(?imx)
        ^[ \t]*[A-Z0-9_]*(?:API_KEY|SECRET|TOKEN|PASSWORD)[A-Z0-9_]*
        [ \t]*=[ \t]*([A-Za-z0-9_./+=:@-]{16,})[ \t]*$
        """),
    re.compile(r"""(?ix)
        (?:api[_-]?key|secret|token|password)
        [ \t]*[:=][ \t]*["']([A-Za-z0-9_./+=:@-]{16,})["']
        """),
)
SAFE_MARKERS = {
    "test",
    "example",
    "placeholder",
    "changeme",
    "your",
    "dummy",
    "fake",
    "redacted",
    "invalid",
}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
    ".ps1",
    ".sh",
    ".service",
    ".ini",
    ".cfg",
}
FORBIDDEN_TRACKED_SUFFIXES = {
    ".db",
    ".sqlite",
    ".sqlite3",
    ".safetensors",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".gguf",
}


def _tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [root / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def _secret_findings(path: Path, content: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for rule, pattern in (
        ("private_key", PRIVATE_KEY),
        ("aws_access_key", AWS_KEY),
        ("openai_style_key", OPENAI_KEY),
    ):
        for match in pattern.finditer(content):
            findings.append(
                {
                    "rule": rule,
                    "line": content.count("\n", 0, match.start()) + 1,
                }
            )
    for pattern in GENERIC_SECRET_PATTERNS:
        for match in pattern.finditer(content):
            if re.fullmatch(r"env:[A-Z][A-Z0-9_]{0,127}", match.group(1)):
                continue
            value = match.group(1).casefold()
            if not any(marker in value for marker in SAFE_MARKERS):
                findings.append(
                    {
                        "rule": "generic_secret_assignment",
                        "line": content.count("\n", 0, match.start()) + 1,
                    }
                )
    return findings


def scan_repository(root: str | Path, *, max_bytes: int = 5 * 1024 * 1024) -> dict:
    repository = Path(root).resolve()
    secret_findings: list[dict[str, Any]] = []
    large_files: list[dict[str, Any]] = []
    forbidden_files: list[str] = []
    forbidden_paths: list[dict[str, Any]] = []
    tracked = _tracked_files(repository)
    for path in tracked:
        relative = path.relative_to(repository).as_posix()
        size = path.stat().st_size
        if size > max_bytes:
            large_files.append({"path": relative, "size_bytes": size})
        if path.suffix.lower() in FORBIDDEN_TRACKED_SUFFIXES:
            forbidden_files.append(relative)
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
            ".env.example",
            ".gitignore",
        }:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for finding in _secret_findings(path, content):
            secret_findings.append({"path": relative, **finding})
        if relative not in {
            "MERGE_PROVENANCE.md",
            "MERGE_FILE_MAP.json",
            "MERGE_SOURCE_INVENTORY.json",
            "MERGE_SOURCE_HASHES.json",
        }:
            for needle in ("E:\\new ocr project", "E:\\project_collected"):
                if needle.casefold() in content.casefold():
                    forbidden_paths.append({"path": relative, "pattern": needle})
    return {
        "schema_version": 1,
        "tracked_file_count": len(tracked),
        "secret_findings": secret_findings,
        "large_files": large_files,
        "forbidden_tracked_files": forbidden_files,
        "forbidden_source_paths": forbidden_paths,
        "passed": not (
            secret_findings or large_files or forbidden_files or forbidden_paths
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--max-bytes", type=int, default=5 * 1024 * 1024)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    report = scan_repository(args.root, max_bytes=args.max_bytes)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.apply:
        if args.output is None:
            raise SystemExit("--apply requires --output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

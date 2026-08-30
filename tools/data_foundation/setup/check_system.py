from __future__ import annotations

import json
import platform
import shutil
import subprocess
from pathlib import Path


def run(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command, check=False, text=True, capture_output=True, timeout=15
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"
    return (completed.stdout or completed.stderr).strip()


def inspect_system() -> dict:
    tools = [
        "git",
        "docker",
        "wsl",
        "nvidia-smi",
        "tesseract",
        "pdftoppm",
        "magick",
        "gs",
    ]
    return {
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "python_executable": shutil.which("python"),
        "powershell": run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "$PSVersionTable.PSVersion.ToString()",
            ]
        ),
        "tools": {tool: shutil.which(tool) for tool in tools},
        "cuda": run(["nvidia-smi"]) if shutil.which("nvidia-smi") else "not detected",
        "rocm": "not detected on Windows by this lightweight checker",
    }


def main() -> int:
    report = inspect_system()
    output = Path("docs/LOCAL_SYSTEM_REPORT.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "# Local System Report\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

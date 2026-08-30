from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from typing import Any


def _run(command: list[str]) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if not executable:
        return {"available": False, "command": command[0], "output": ""}
    try:
        completed = subprocess.run(
            [executable, *command[1:]],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return {
            "available": True,
            "command": " ".join(command),
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {"available": True, "command": " ".join(command), "error": str(exc)}


def _torch_info() -> dict[str, Any]:
    try:
        import torch  # type: ignore
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    info: dict[str, Any] = {
        "available": True,
        "version": getattr(torch, "__version__", ""),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "hip_version": getattr(torch.version, "hip", None),
        "device_count": 0,
        "devices": [],
    }
    try:
        count = torch.cuda.device_count()
        info["device_count"] = int(count)
        for index in range(count):
            props = torch.cuda.get_device_properties(index)
            info["devices"].append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": getattr(props, "total_memory", None),
                    "major": getattr(props, "major", None),
                    "minor": getattr(props, "minor", None),
                }
            )
    except Exception as exc:
        info["device_error"] = str(exc)
    return info


def collect() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "torch": _torch_info(),
        "rocm_smi": _run(
            [
                "rocm-smi",
                "--showproductname",
                "--showdriverversion",
                "--showmeminfo",
                "vram",
            ]
        ),
        "rocminfo": _run(["rocminfo"]),
        "hipcc": _run(["hipcc", "--version"]),
        "nvidia_smi": _run(["nvidia-smi"]),
    }


def main() -> int:
    print(json.dumps(collect(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

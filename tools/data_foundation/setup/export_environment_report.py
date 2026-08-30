from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path


def main() -> int:
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        text=True,
        capture_output=True,
        check=False,
    )
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "pip_freeze": freeze.stdout.splitlines(),
    }
    path = Path("docs/ENVIRONMENT_REPORT.json")
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

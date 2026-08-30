from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv", default=".venv")
    args = parser.parse_args()
    venv_path = Path(args.venv)
    subprocess.check_call([sys.executable, "-m", "venv", str(venv_path)])
    print(f"Created virtual environment at {venv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

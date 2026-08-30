from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", default="requirements-dev.txt")
    args = parser.parse_args()
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", args.requirements]
    )
    subprocess.check_call([sys.executable, "-m", "pip", "check"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

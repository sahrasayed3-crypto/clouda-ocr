from __future__ import annotations

from pathlib import Path


def append_execution_log(
    message: str, path: str | Path = "logs/execution_history.log"
) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")

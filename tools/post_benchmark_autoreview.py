import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from pdfword.self_learning import SelfLearningEngine

WORKSPACE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()
POLL_SECONDS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
LOGS_DIR = WORKSPACE / "logs"
GENERATED_DIR = WORKSPACE / "samples" / "generated"
SKIP_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv311",
    "__pycache__",
    "backups",
    "conversions",
    "data",
    "logs",
    "outputs",
    "poppler-26.02.0",
    "tools/poppler",
    "tools/python",
}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def _list_benchmark_pids() -> list[int]:
    try:
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -match 'tools/run_full_benchmark.py' } | Select-Object -ExpandProperty ProcessId",
            ],
            text=True,
            cwd=str(WORKSPACE),
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError:
        return []
    pids = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.isdigit():
            pids.append(int(line))
    return pids


def _run_cmd(args: list[str]) -> tuple[int, str]:
    p = subprocess.run(
        args,
        cwd=str(WORKSPACE),
        text=True,
        capture_output=True,
    )
    text = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return p.returncode, text.strip()


def _relative_key(path: Path) -> str:
    try:
        return path.relative_to(WORKSPACE).as_posix()
    except ValueError:
        return path.as_posix()


def _should_skip(path: Path) -> bool:
    rel = _relative_key(path)
    parts = set(Path(rel).parts)
    return bool(parts & SKIP_PARTS) or any(
        rel == item or rel.startswith(f"{item}/") for item in SKIP_PARTS
    )


def _compile_targets() -> list[str]:
    targets = ["app.py", "pdfword", "scripts", "tests"]
    tools_dir = WORKSPACE / "tools"
    if tools_dir.exists():
        targets.extend(
            str(path.relative_to(WORKSPACE)) for path in sorted(tools_dir.glob("*.py"))
        )
    return targets


def _scan_signals() -> tuple[list[str], list[str]]:
    todo_hits: list[str] = []
    bare_except_hits: list[str] = []
    for path in WORKSPACE.rglob("*"):
        if not path.is_file():
            continue
        if _should_skip(path):
            continue
        if path.suffix.lower() not in {
            ".py",
            ".md",
            ".txt",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        rel = path.relative_to(WORKSPACE).as_posix()
        for i, line in enumerate(text.splitlines(), start=1):
            if re.search(r"\b(TODO|FIXME|XXX|HACK)\b", line):
                todo_hits.append(f"{rel}:{i}: {line.strip()}")
            if re.search(r"^\s*except\s*:\s*$", line):
                bare_except_hits.append(f"{rel}:{i}: {line.strip()}")
    return todo_hits, bare_except_hits


def main() -> int:
    stamp = _timestamp()
    watcher_log = LOGS_DIR / f"post_benchmark_autoreview_{stamp}.log"
    summary_md = LOGS_DIR / f"post_benchmark_summary_{stamp}.md"

    _append(watcher_log, f"[{_now_iso()}] watcher started")
    _append(watcher_log, f"[{_now_iso()}] waiting for benchmark processes to finish")
    while True:
        pids = _list_benchmark_pids()
        if not pids:
            break
        _append(
            watcher_log,
            f"[{_now_iso()}] running benchmark pids: {','.join(map(str, pids))}",
        )
        time.sleep(POLL_SECONDS)

    _append(watcher_log, f"[{_now_iso()}] benchmark finished; starting review")

    latest_json = None
    latest_md = None
    json_candidates = sorted(
        GENERATED_DIR.glob("benchmark_report_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    md_candidates = sorted(
        GENERATED_DIR.glob("benchmark_report_*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if json_candidates:
        latest_json = json_candidates[0]
    if md_candidates:
        latest_md = md_candidates[0]

    bench = None
    if latest_json and latest_json.exists():
        bench = json.loads(latest_json.read_text(encoding="utf-8"))

    unit_code, unit_out = _run_cmd(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"]
    )
    compile_code, compile_out = _run_cmd(
        [sys.executable, "-m", "compileall", "-q", *_compile_targets()]
    )
    todo_hits, bare_except_hits = _scan_signals()
    learner = SelfLearningEngine()
    learning_summary = learner.get_summary()
    adaptive_clear = learner.get_adaptive_profile("CLEAR")
    adaptive_complex = learner.get_adaptive_profile("COMPLEX")

    all_files = [p for p in WORKSPACE.rglob("*") if p.is_file() and not _should_skip(p)]
    py_files = [p for p in all_files if p.suffix.lower() == ".py"]

    lines: list[str] = []
    lines.append("# Post-Benchmark Autonomous Review")
    lines.append("")
    lines.append(f"- Generated at: {_now_iso()}")
    lines.append(f"- Workspace: `{WORKSPACE}`")
    lines.append("")

    if bench:
        lines.append("## Benchmark Final Result")
        lines.append("")
        lines.append(f"- Latest JSON report: `{latest_json}`")
        if latest_md:
            lines.append(f"- Latest Markdown report: `{latest_md}`")
        summary = bench.get("summary", {})
        lines.append(f"- Total cases: {summary.get('total_cases')}")
        lines.append(f"- Success: {summary.get('success_cases')}")
        lines.append(
            f"- Expected failures passed: {summary.get('expected_failures_ok')}"
        )
        lines.append(f"- Failed: {summary.get('failed_cases')}")
        lines.append(f"- Avg word accuracy: {summary.get('avg_word_accuracy')}")
        lines.append(f"- Avg char accuracy: {summary.get('avg_char_accuracy')}")
        lines.append("")

    lines.append("## Project Review Snapshot")
    lines.append("")
    lines.append(f"- Total files scanned: {len(all_files)}")
    lines.append(f"- Python files scanned: {len(py_files)}")
    lines.append(f"- Unit tests exit code: {unit_code}")
    lines.append(f"- Compile-all exit code: {compile_code}")
    lines.append(
        "- Learning memory: "
        f"errors={learning_summary.get('runtime_error_events', 0)} "
        f"keys={learning_summary.get('runtime_error_keys', 0)} "
        f"corrections={learning_summary.get('corrections', 0)}"
    )
    lines.append("")

    lines.append("### Unit Test Output")
    lines.append("")
    lines.append("```text")
    lines.append(unit_out or "(no output)")
    lines.append("```")
    lines.append("")

    lines.append("### Compile Output")
    lines.append("")
    lines.append("```text")
    lines.append(compile_out or "(no output)")
    lines.append("```")
    lines.append("")

    lines.append("### Potential Improvement Signals")
    lines.append("")
    lines.append("#### TODO/FIXME/HACK hits")
    lines.append("")
    lines.append("```text")
    lines.append("\n".join(todo_hits[:300]) if todo_hits else "No hits.")
    lines.append("```")
    lines.append("")
    lines.append("#### Bare except hits")
    lines.append("")
    lines.append("```text")
    lines.append("\n".join(bare_except_hits[:300]) if bare_except_hits else "No hits.")
    lines.append("```")
    lines.append("")
    lines.append("#### Adaptive profiles from runtime failures")
    lines.append("")
    lines.append("```text")
    lines.append(f"CLEAR  -> {adaptive_clear}")
    lines.append(f"COMPLEX-> {adaptive_complex}")
    lines.append("```")
    lines.append("")

    lines.append("## Suggested Next Improvements")
    lines.append("")
    lines.append(
        "1. Prioritize top failures from the latest benchmark report and add targeted regression tests."
    )
    lines.append(
        "2. Replace bare `except:` with explicit exception classes in critical paths."
    )
    lines.append(
        "3. Resolve high-impact TODO/FIXME markers in OCR and benchmark code paths."
    )
    lines.append(
        "4. Re-run full benchmark after each major OCR change to maintain reproducible trend tracking."
    )

    _write(summary_md, "\n".join(lines) + "\n")
    _append(watcher_log, f"[{_now_iso()}] review completed")
    _append(watcher_log, f"[{_now_iso()}] summary file: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

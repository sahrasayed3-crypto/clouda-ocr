"""Adversarial: CLI exit-code contract and resume across a process restart.

Gaps: ``test_cli.py`` only asserts ``cli is not None`` (exit-code vocabulary
never exercised) and ``test_resume.py`` never exercises the CLI ``--resume``
path across two separate OS processes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.quality.conftest import make_manifest, make_row
from clouda_data.quality import cli

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_scan_exit_code_vocabulary(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    # 0: empty manifest -> verdict PASS
    empty = make_manifest(tmp_path / "empty", [])
    assert cli.main(["scan", str(empty)]) == 0

    # 1: gate FAIL (duplicate sample_id across sources)
    fail = make_manifest(
        tmp_path / "fail",
        [
            make_row("smp_d", source_id="src1", text="نص"),
            make_row("smp_d", source_id="src2", text="نص"),
        ],
    )
    assert cli.main(["scan", str(fail)]) == 1

    # 2: config/usage error (missing manifest)
    capsys.readouterr()
    assert cli.main(["scan", str(tmp_path / "nope.jsonl")]) == 2

    # 0: verify on a report matching the default config; 2: foreign identity
    from clouda_data.quality.config import QualityGateConfig

    report = tmp_path / "pass_report.json"
    report.write_text(
        json.dumps(
            {
                "verdict": "PASS",
                "config_identity": QualityGateConfig().identity(),
            }
        ),
        encoding="utf-8",
    )
    capsys.readouterr()
    assert cli.main(["verify", "--report", str(report)]) == 0
    foreign = tmp_path / "foreign_report.json"
    foreign.write_text(
        json.dumps({"verdict": "PASS", "config_identity": "cfgA"}), encoding="utf-8"
    )
    assert cli.main(["verify", "--report", str(foreign)]) == 2


def test_resume_across_process_restart(tmp_path) -> None:  # type: ignore[no-untyped-def]
    manifest = make_manifest(
        tmp_path / "run",
        [
            make_row("smp_r1", source_id="src1", text="نص"),
            make_row("smp_r2", source_id="src1", text="نص ب"),
        ],
    )
    script = (
        "import sys\n"
        "from clouda_data.quality import cli\n"
        f"rc = cli.main(['scan', r'{manifest}', '--resume'])\n"
        "print(f'RC={rc}')\n"
    )
    outputs = []
    for _ in range(2):  # two separate OS processes
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        outputs.append((proc.stdout, proc.stderr, proc.returncode))

    first_out, first_err, first_rc = outputs[0]
    second_out, second_err, second_rc = outputs[1]
    assert first_rc == second_rc, f"exit code changed across restart: {outputs}"
    assert "resuming scan" not in first_err, f"first run claimed resume: {first_err!r}"
    assert (
        "resuming scan" in second_err
    ), f"second process did not resume persisted state: {second_err!r}"
    state = manifest.parent / ".quality" / "run_state.json"
    assert state.is_file(), "run_state.json not persisted for resume"
    payload = json.loads(state.read_text(encoding="utf-8"))
    assert payload["stage"] == "scan"
    assert (
        payload["processed_count"] == 2
    ), f"checkpoint lost row count across restart: {payload}"

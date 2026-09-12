"""Resume equivalence, determinism and optional-torch tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from clouda_training.experiments import (  # noqa: E402
    list_checkpoints,
    list_runs,
    resume_run,
    run_experiment,
    RunStatus,
)


def _load_model_state_from_checkpoint(checkpoint_path: Path):
    metadata = json.loads(
        (checkpoint_path / "metadata.json").read_text(encoding="utf-8")
    )
    digest = metadata["torch_state_sha256"]
    from clouda_training.runtime.checkpoint_torch import load_torch_state

    return load_torch_state(checkpoint_path, expected_sha256=digest)


def test_checkpoint_carries_full_execution_state(torch_config, tmp_path) -> None:
    handle = run_experiment(torch_config)
    checkpoints = list_checkpoints(handle.path)
    assert checkpoints, "no checkpoints were saved"
    payload = _load_model_state_from_checkpoint(checkpoints[-1].path)
    assert "model" in payload
    assert "optimizer" in payload
    assert "rng" in payload
    assert "python" in payload["rng"]
    assert "torch" in payload["rng"]
    assert payload["step"] == checkpoints[-1].step


def test_interrupted_resume_matches_uninterrupted_run(torch_config, tmp_path) -> None:
    """The crown jewel: resume must reproduce the uninterrupted trajectory exactly."""
    # Reference: uninterrupted run
    reference = run_experiment(torch_config)
    ref_checkpoints = list_checkpoints(reference.path)
    ref_state = _load_model_state_from_checkpoint(ref_checkpoints[-1].path)

    # Interrupt at step 9: checkpoints exist at 4 and 8 (saved at step end);
    # framework semantics raise at the START of step N, so latest checkpoint < 9.
    with pytest.raises(KeyboardInterrupt):
        run_experiment(torch_config, interrupt_at_step=9)
    interrupted = list_runs(
        torch_config.runtime.output_root, status=RunStatus.INTERRUPTED
    )[0]
    steps = [item.step for item in list_checkpoints(interrupted.path)]
    assert 8 in steps, f"expected checkpoint at step 8, got {steps}"

    resumed = resume_run(interrupted.run_id, torch_config.runtime.output_root)
    assert resumed.status is RunStatus.COMPLETED
    assert resumed.summary()["resumed_from_step"] == 8

    res_checkpoints = list_checkpoints(resumed.path)
    res_state = _load_model_state_from_checkpoint(res_checkpoints[-1].path)

    # Bitwise parameter equality between continuous and resumed runs
    for key in ref_state["model"]:
        assert torch.equal(
            ref_state["model"][key], res_state["model"][key]
        ), f"parameter {key} differs after resume"


def test_same_seed_reproduces_identical_parameters(torch_config, tmp_path) -> None:
    left = run_experiment(torch_config)
    right = run_experiment(torch_config)
    lc = list_checkpoints(left.path)[-1]
    rc = list_checkpoints(right.path)[-1]
    lstate = _load_model_state_from_checkpoint(lc.path)
    rstate = _load_model_state_from_checkpoint(rc.path)
    for key in lstate["model"]:
        assert torch.equal(
            lstate["model"][key], rstate["model"][key]
        ), "same seed must reproduce identical parameters"


def test_rng_state_restoration_is_exact(torch_config) -> None:
    from clouda_training.runtime.rng import capture_rng_state, restore_rng_state

    import random

    import numpy as np

    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)
    state = capture_rng_state()

    # consume randomness
    _ = [random.random() for _ in range(10)]
    _ = np.random.rand(10)
    _ = torch.rand(10)

    restore_rng_state(state)
    draws_py = [random.random() for _ in range(5)]
    draws_np = np.random.rand(5)
    draws_torch = torch.rand(5)

    restore_rng_state(state)
    assert [random.random() for _ in range(5)] == draws_py
    assert np.array_equal(np.random.rand(5), draws_np)
    assert torch.equal(torch.rand(5), draws_torch)


def test_core_imports_survive_missing_torch(
    torch_config, monkeypatch, tmp_path
) -> None:
    """Simulate torch-absent environment: core imports must not crash."""
    import builtins

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("No module named 'torch' (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    import importlib

    import clouda_training.runtime as runtime_pkg

    importlib.reload(runtime_pkg)
    assert runtime_pkg.torch_available() is False
    with pytest.raises(ImportError, match="PyTorch is required"):
        runtime_pkg.TorchTrainerBackend

    # run_experiment with torch adapter must fail with a helpful error
    from clouda_training.experiments import run_experiment as re_loaded

    with pytest.raises(RuntimeError, match="PyTorch is not installed"):
        re_loaded(torch_config)

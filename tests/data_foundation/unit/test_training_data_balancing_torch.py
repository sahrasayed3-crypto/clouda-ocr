"""Balancing/curriculum hooks, collation contract, and torch adapter tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from clouda_data.training_data.balancing import (
    CurriculumHook,
    SamplingSchedule,
    static_schedule,
    stratified_counts,
    stratum_of,
    weighted_stream,
)
from clouda_data.training_data.collation import (
    BatchCollator,
    LoadingTransform,
    NoopCollator,
    NoopTransform,
    SampleTransform,
    transform_batch,
)
from clouda_data.training_data.models import (
    BatchConfig,
    SampleReference,
)
from clouda_data.training_data.loader import StreamingTrainingDataLoader

from tests.data_foundation.fixtures.training_data_fixtures import (
    build_synthetic_dataset,
    shard_dataset,
)


def _ref(index: int, row_extra: dict | None = None) -> SampleReference:
    row = {"sample_id": f"s{index}", "position": index}
    if row_extra:
        row.update(row_extra)
    return SampleReference(
        sample_id=f"s{index}", shard_id="shard-x", position=index, row=row
    )


# ---------------------------------------------------------------------------
# Balancing / curriculum
# ---------------------------------------------------------------------------


class TestBalancing:
    def test_stratum_detection(self):
        assert stratum_of(_ref(0, {"difficulty_bucket": "hard"})) == (
            "difficulty_bucket=hard"
        )
        assert stratum_of(_ref(1, {"metadata": {"document_type": "invoice"}})) == (
            "document_type=invoice"
        )
        assert stratum_of(_ref(2)) is None

    def test_no_labels_no_invention(self):
        assert stratum_of(_ref(3)) is None

    def test_weighted_stream_keeps_all_without_schedule(self):
        samples = [_ref(i) for i in range(10)]
        out = list(
            weighted_stream(iter(samples), SamplingSchedule({}), global_seed=1, epoch=0)
        )
        assert len(out) == 10

    def test_weighted_stream_zero_weight_removes_stratum(self):
        samples = [
            _ref(0, {"difficulty_bucket": "easy"}),
            _ref(1, {"difficulty_bucket": "hard"}),
            _ref(2, {"difficulty_bucket": "easy"}),
        ]
        out = list(
            weighted_stream(
                iter(samples),
                SamplingSchedule({"difficulty_bucket=hard": 0.0}),
                global_seed=42,
                epoch=0,
            )
        )
        assert all(s.sample_id != "s1" for s in out)
        assert len(out) == 2

    def test_weighted_stream_deterministic(self):
        samples = [_ref(i, {"difficulty_bucket": "hard"}) for i in range(20)]
        kwargs = dict(
            schedule=SamplingSchedule({"difficulty_bucket=hard": 0.5}),
            global_seed=9,
            epoch=0,
        )
        a = [s.sample_id for s in weighted_stream(iter(samples), **kwargs)]
        b = [s.sample_id for s in weighted_stream(iter(samples), **kwargs)]
        assert a == b

    def test_invalid_weight_rejected(self):
        with pytest.raises(ValueError, match="within"):
            SamplingSchedule({"difficulty_bucket=hard": 1.5})

    def test_curriculum_hook_epoch_schedule(self):
        hook = CurriculumHook(
            lambda epoch: SamplingSchedule(
                {"difficulty_bucket=hard": 0.0 if epoch < 2 else 1.0}
            )
        )
        samples = [
            _ref(0, {"difficulty_bucket": "easy"}),
            _ref(1, {"difficulty_bucket": "hard"}),
        ]
        early = [
            s.sample_id for s in hook.stream(iter(samples), global_seed=1, epoch=0)
        ]
        late = [s.sample_id for s in hook.stream(iter(samples), global_seed=1, epoch=3)]
        assert early == ["s0"]
        assert late == ["s0", "s1"]

    def test_static_schedule(self):
        hook = static_schedule({"difficulty_bucket=hard": 0.0})
        assert hook.schedule(5).weights == {"difficulty_bucket=hard": 0.0}

    def test_stratified_counts(self):
        samples = [
            _ref(0, {"profile": "clean"}),
            _ref(1, {"profile": "clean"}),
            _ref(2, {"profile": "old_book"}),
        ]
        assert stratified_counts(samples) == {
            "profile=clean": 2,
            "profile=old_book": 1,
        }


# ---------------------------------------------------------------------------
# Collation contract
# ---------------------------------------------------------------------------


class TestCollation:
    def test_protocols_satisfied(self):
        assert isinstance(NoopTransform("."), SampleTransform)
        assert isinstance(NoopCollator(), BatchCollator)
        assert isinstance(LoadingTransform("."), SampleTransform)

    def test_noop_transform_resolves_paths(self, tmp_path: Path):

        root = tmp_path / "ds"
        (root / "pages").mkdir(parents=True)
        from tests.data_foundation.fixtures.training_data_fixtures import tiny_png

        (root / "pages" / "p.png").write_bytes(tiny_png())
        transform = NoopTransform(root)
        ref = SampleReference(
            sample_id="s", shard_id="sh", position=0, image_path="pages/p.png"
        )
        out = transform(ref)
        assert out.image_path == str((root / "pages" / "p.png").resolve())
        assert out.image is None  # not loaded

    def test_loading_transform_decodes(self, tmp_path: Path):
        root = tmp_path / "ds"
        (root / "pages").mkdir(parents=True)
        from tests.data_foundation.fixtures.training_data_fixtures import tiny_png

        (root / "pages" / "p.png").write_bytes(tiny_png())
        transform = LoadingTransform(root)
        ref = SampleReference(
            sample_id="s", shard_id="sh", position=0, image_path="pages/p.png"
        )
        out = transform(ref)
        assert out.image is not None and out.image.size == (4, 4)

    def test_transform_batch_pipeline(self, tmp_path: Path):
        root = tmp_path / "ds"
        (root / "pages").mkdir(parents=True)
        from tests.data_foundation.fixtures.training_data_fixtures import tiny_png

        (root / "pages" / "p.png").write_bytes(tiny_png())
        batch = {
            "samples": [
                SampleReference(
                    sample_id="s", shard_id="sh", position=0, image_path="pages/p.png"
                )
            ]
        }
        out = transform_batch(batch, NoopTransform(root), NoopCollator())
        assert len(out) == 1 and out[0].image_path.endswith("p.png")

    def test_loader_end_to_end_with_collation(self, tmp_path):
        manifest, root = build_synthetic_dataset(tmp_path / "ds", count=6)
        shard_dataset(manifest, tmp_path / "out", samples_per_shard=3)
        loader = StreamingTrainingDataLoader(
            shard_index_path=tmp_path / "out" / "shard_index.json",
            loader_config=__import__(
                "clouda_data.training_data.models", fromlist=["TrainingDataConfig"]
            ).TrainingDataConfig(
                dataset_id="synthetic_training_fixture",
                dataset_version="1.0.0",
                batch=BatchConfig(batch_size=3),
            ),
            manifest_path=manifest,
            dataset_root=root,
        )
        batches = list(loader.iter_batches(epoch=0))
        assert len(batches) == 2
        collated = transform_batch(batches[0], NoopTransform(root), NoopCollator())
        assert len(collated) == 3


# ---------------------------------------------------------------------------
# Torch adapter (torch not installed)
# ---------------------------------------------------------------------------


class TestTorchAdapter:
    def test_torch_absent_detected(self):
        from clouda_data.training_data import torch_adapter

        pytest.importorskip  # documentation only
        try:
            import torch  # noqa: F401

            pytest.skip("torch installed; absence tests not applicable")
        except ImportError:
            pass
        assert torch_adapter.torch_available() is False

    def test_require_torch_raises_import_error(self):
        from clouda_data.training_data import torch_adapter

        try:
            import torch  # noqa: F401

            pytest.skip("torch installed; absence tests not applicable")
        except ImportError:
            pass
        with pytest.raises(ImportError, match="PyTorch is not installed"):
            torch_adapter.require_torch()

    def test_core_package_importable_without_torch(self):
        import clouda_data.training_data  # noqa: F401
        import clouda_data.training_data.loader  # noqa: F401
        import clouda_data.training_data.torch_adapter  # noqa: F401

    def test_torch_adapter_not_imported_by_package_init(self):
        import subprocess
        import sys

        code = (
            "import sys; import clouda_data.training_data; "
            "assert 'clouda_data.training_data.torch_adapter' not in sys.modules, "
            "'torch_adapter was eagerly imported'; print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr

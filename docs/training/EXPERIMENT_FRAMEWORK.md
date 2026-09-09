# Training Experiment Framework

## Purpose and current boundary

The framework manages reproducible Clouda OCR experiments before production
training hardware is available. It validates immutable experiment
specifications, protects dataset boundaries, records run provenance, drives a
framework-neutral trainer adapter, logs metrics, manages checkpoints and
resume, and compares runs. The checked-in adapter is deliberately a
deterministic `MockTrainer`: it performs no model download, inference, or
gradient update.

An **experiment** is a named hypothesis and configuration lineage. A **run** is
one execution of a resolved experiment configuration. Repeating the same
experiment produces a new run id while retaining the same config hash.

## Configuration

[`configs/training/mock-experiment.yaml`](../../configs/training/mock-experiment.yaml)
is the complete offline example. The schema is strict: unknown sections and
fields fail, required identities cannot be blank, numeric boundaries are
validated, and relative manifest/output paths resolve from the config file.
JSON is also accepted because YAML is a superset of JSON.

After resolution the nested configuration consists only of frozen dataclasses.
Canonical JSON with sorted keys is SHA-256 hashed. Command-line overrides use
`section.field=value`; an unknown field is rejected rather than ignored.

```powershell
clouda-training validate-config configs/training/mock-experiment.yaml
clouda-training dry-run configs/training/mock-experiment.yaml
clouda-training dry-run configs/training/mock-experiment.yaml `
  --override training.seed=42 --json
```

`clouda-training run` uses the configured adapter. This build permits only
`mock`/`dry_run` adapters with `runtime.dry_run: true`; it fails closed for real
training.

## Run contract and lifecycle

```text
runs/<experiment>/<experiment>__<UTC>__<config-hash>-<nonce>/
  resolved_config.yaml
  metadata.json
  status.json
  metrics.jsonl
  summary.json
  environment.json
  checkpoints/step-00000002/{metadata.json,state.json}
  logs/
  artifacts/integrity.json
```

Creation never reuses a directory. `status.json` records an append-only logical
history through `CREATED`, `RUNNING`, and one of `COMPLETED`, `FAILED`, or
`INTERRUPTED`. Exceptions and keyboard interrupts update status before being
re-raised, and failed/interrupted directories remain available for audit and
resume. Completed core artifacts have SHA-256 entries in
`artifacts/integrity.json`.

Metadata records the config and dataset-manifest hashes, dataset/model ids and
versions, preprocessing version, selected split, source ids/licenses already
present in the manifest, seed, timestamps, host/platform/Python, Git commit and
dirty state, command line, and resume source. Environment capture records only
an allowlist of runtime facts and package versions; it never reads or records
environment-variable values, tokens, or credentials.

## Reproducibility

The centralized seed function covers Python, NumPy when installed, PyTorch when
installed, and CUDA generators when CUDA is available. Deterministic PyTorch
algorithms are requested in deterministic mode. The exact settings applied and
limitations are stored in metadata. These controls reduce variability; they do
not promise bitwise identity for every future kernel, device, or adapter.

Mock loss is derived solely from seed and step, so identical resolved configs
produce identical metric streams apart from timestamps and run ids. Changing
the seed changes the simulated result.

## Dataset identity and protected holdout

Before a run directory is created, the framework requires the manifest to
exist, computes its SHA-256, parses it with the canonical pre-training manifest
reader, and confirms that the selected split contains rows. It rejects:

- `holdout`, `protected_holdout`, `benchmark_holdout`, `private_holdout`, and
  any split name containing `holdout`;
- manifests marked `protected`, `protected_holdout`, `holdout`, `benchmark`, or
  `evaluation_only`; and
- selected rows carrying a protected marker.

A canonical manifest may still contain non-selected holdout rows; only the
explicit non-protected split is considered. The mock evaluator uses generated
strings and the existing `clouda_data.evaluation` CER/WER functions. It never
automatically evaluates the protected holdout.

## Checkpoints and resume

Checkpoint directories use zero-padded step names. Each contains step, epoch,
timestamp, metric snapshot, config hash, run id, model identity, dataset
identity/version, and a SHA-256 over its state file. A same-directory partial
checkpoint is finalized by rename. Corrupt state or missing metadata fails
validation.

Retention counts the best checkpoint inside `save_total_limit`; remaining
slots keep the newest checkpoints. Resume is explicit and allowed only for a
`FAILED` or `INTERRUPTED` run. It validates the run id, exact config hash,
experiment, model id/revision, and dataset id/version before continuing from
the latest retained checkpoint. There is intentionally no silent compatibility
override. Cross-experiment resume/lineage is not enabled until a real adapter
defines which optimizer/model state changes are safe.

```powershell
clouda-training list --runs-root runs
clouda-training show <run-id> --runs-root runs --json
clouda-training checkpoints <run-id> --runs-root runs
clouda-training resume <run-id> --runs-root runs
```

## Metrics, evaluation, registry, and comparison

Metrics are append-and-fsync JSONL records containing timestamp, step, epoch,
split, metric name, numeric value, and run id. Summaries expose final and best
metrics, duration, checkpoint count, best checkpoint, and resume step.

The lightweight registry derives experiments and runs from their directories;
there is no duplicate database. It supports status/tag filters, latest run, and
best completed run by metric.

```powershell
clouda-training compare <run-a> <run-b> --runs-root runs
clouda-training compare <run-a> <run-b> --runs-root runs --format json
clouda-training compare <run-a> <run-b> --runs-root runs --format csv
```

Comparison reports run status/duration, resolved configuration differences
(including model, dataset, and seed), and final metric differences.

## Adding a real adapter later

Implement the `Trainer` protocol in `clouda_training.experiments.trainer` and
map only adapter-owned model loading/training details into it. A future
HunyuanOCR-1.5 adapter should pin model/tokenizer revisions, honor offline and
trust-remote-code policies, consume only the already validated dataset split,
emit metrics through `MetricLogger`, and checkpoint through
`CheckpointManager`. Do not put Hunyuan-specific assumptions into run,
dataset, registry, or comparison code.

Real adapter activation must remain fail-closed until its hardware validation
and license review are complete. GPU training support has **not** been
validated by this work.

See [HARDWARE_VALIDATION_TODO.md](HARDWARE_VALIDATION_TODO.md) for the deferred
validation boundary.

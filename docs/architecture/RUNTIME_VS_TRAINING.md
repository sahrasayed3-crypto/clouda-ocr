# Runtime versus training

User conversion requests execute only the `pdf_conversion` workload. Dataset
preparation, training preparation, and model evaluation have separate queues,
timeouts, retry policies, worker capabilities, and filesystem access.

The Streamlit and FastAPI entrypoints do not import `clouda_training`. The
subsystem validates licenses, estimates local examples and bytes, creates
deterministic document-level splits, emits plans, and can execute a CPU-only
mock experiment lifecycle for engineering validation. User documents have no
route into the dataset catalog, training planner, or experiment runner.

Real adapter execution is fail-closed behind explicit opt-in (`dry_run=false`,
offline runtime, approved dataset, approved model in the training-use
approval catalog) and has been validated only with a tiny synthetic trainer
on CPU — no real OCR model has been trained. See
[`docs/training/REAL_TRAINING_RUNTIME.md`](../training/REAL_TRAINING_RUNTIME.md).

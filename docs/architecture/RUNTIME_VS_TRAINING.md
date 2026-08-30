# Runtime versus training

User conversion requests execute only the `pdf_conversion` workload. Dataset
preparation, training preparation, and model evaluation have separate queues,
timeouts, retry policies, worker capabilities, and filesystem access.

The Streamlit and FastAPI entrypoints do not import `clouda_training`.
Training execution is intentionally disabled; the subsystem validates
licenses, estimates local examples and bytes, creates deterministic
document-level splits, and emits plans. User documents have no route into the
dataset catalog or training planner.

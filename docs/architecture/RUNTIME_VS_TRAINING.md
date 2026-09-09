# Runtime versus training

User conversion requests execute only the `pdf_conversion` workload. Dataset
preparation, training preparation, and model evaluation have separate queues,
timeouts, retry policies, worker capabilities, and filesystem access.

The Streamlit and FastAPI entrypoints do not import `clouda_training`.
Production model training remains intentionally disabled. The subsystem
validates licenses, estimates local examples and bytes, creates deterministic
document-level splits, emits plans, and can execute a CPU-only mock experiment
lifecycle for engineering validation. User documents have no route into the
dataset catalog, training planner, or experiment runner. Real adapters remain
fail-closed until hardware and license validation is complete.

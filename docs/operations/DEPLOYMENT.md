# Deployment

Use Python 3.11 and install the tested constraints. Configure external roots and
secrets through one protected environment file. Windows may launch Streamlit
with `python -m streamlit run app.py`. Linux units under `deploy/linux` launch
the runtime and internal worker API.

Run dataset, training-preparation, and model-evaluation workers under separate
OS identities with only their declared roots. Keep Redis private. Terminate TLS
at a trusted reverse proxy, validate hosts, restrict forwarded-header sources,
apply request and rate limits, and protect the local single-user UI before
multi-user exposure. Rotate logs and worker keys; never put secrets on command
lines.

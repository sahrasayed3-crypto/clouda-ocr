# Security Audit — Current Session (2026-09-21)

Scope: `pdfword/` (FastAPI server, worker, Streamlit components), `clouda_lab/`
(loopback dashboard), `clouda_data/`, `clouda_training/`, `clouda_models/`,
launch scripts, packaging metadata. Method: Bandit (repo config), pip-audit,
manual review of every endpoint/upload/fetch/deserialize/subprocess surface.

## Automated tooling results

- **ruff**: clean (E4/E7/E9/F over the repo).
- **mypy**: 1 pre-existing annotation error in `tests/test_maintenance.py:55`
  (test file only; production packages clean).
- **Bandit** (source packages, repo config): 0 High / 1 Medium / 62 Low.
  - Medium B614 `clouda_training/runtime/checkpoint_torch.py:47` —
    `torch.load(weights_only=False)`. Both callers pass `expected_sha256`
    (`runtime/torch_backend.py:231,253`) and the file is local-only, not
    network-reachable. **Hardening opportunity**: prefer `weights_only=True`
    once the payload is proven tensor-only, or make the digest mandatory.
  - High B613 `_raqm/corpus.py:59` — intentional Arabic bidi control
    characters (RLM/LRM corpus constants). **False positive.**
  - B105 `worker_api.py:61` — regex constant, not a credential. **False
    positive.**
  - Low B110 (try/except-pass) sites triaged: all are degrade-with-log paths
    (status probes, optional enrichment), none swallow state mutations.
- **pip-audit** (venv): 7 advisories in 2 packages.
  - `pypdf 6.15.0` — PYSEC-2026-3910/3911/3913 (CVE-2026-84310/84311/84309,
    DoS via crafted PDF outlines/XObjects/outline writes). **Material**: pypdf
    parses user-uploaded PDFs. The repo's `constraints/py311.txt` already
    pinned `pypdf==6.16.1` but the dev venv and the minimum floors were
    stale. **Fixed this session**: floors raised to `pypdf>=6.16.1` in
    `pyproject.toml` + `requirements-base.txt`, venv upgraded, PDF-path tests
    green.
  - `setuptools 78.1.0` (dev venv only) — PYSEC-2025-49 (path traversal in
    deprecated `PackageIndex`), PYSEC-2026-3447 (MANIFEST.in NFC/NFD sdist
    bypass, macOS-relevant). **Not applicable at runtime**; builds use the
    build-system requirement `setuptools>=83,<84` (fixed line). Environment
    hygiene: upgrade the dev venv's setuptools to >=83 at convenience.
  - torch skipped (`2.14.0+cpu` not on PyPI) — expected for a local CPU wheel.

## Manual review — confirmed posture (negative checks)

1. **Archive handling**: no `extractall` in project code. Every zip passes
   `clouda_contracts/archive_security.py` (zip-slip, absolute paths, `..`,
   NUL, symlinks, special modes, encrypted members, duplicate/case-colliding
   names, per-member + total caps, >100:1 compression ratio) and is extracted
   member-by-member with containment assertions (`pdfword/backup.py:108-124`).
   Applied to worker DOCX uploads (`worker_api.py:536-547,2219-2230`) and
   backups.
2. **Uploads**: extension + streaming size cap (413) + `%PDF` magic + page
   caps (user 500/guest 5); guest 10 MiB vs user 100 MiB; storage under
   server-generated uuid names; user filename never used in output paths;
   Content-Type never trusted; download names fixed (`result.docx`)
   (`worker_api.py:1195-1247,1379-1494`, `tenant_storage.py:82-127`).
3. **SSRF**: no user-supplied URLs fetched. OpenRouter/provider URLs are
   operator constants/env; the local-OCR HTTP provider refuses non-loopback
   hosts unless explicitly env-enabled, refuses non-HTTP schemes, caps
   response size (`local_ocr_adapters.py:131-137,183-185`). Dataset downloader
   resolves and blocks private/loopback/link-local IPs, re-validates every
   redirect, disables proxies, HTTP requires opt-in
   (`clouda_data/datasets/downloader.py:91-150`). Residual: validation-time
   vs connect-time DNS re-resolution TOCTOU (HARDENING — pin resolved IP).
4. **Subprocess**: two sites, both argv-list `shell=False`, allowlisted
   executables (symlink-refusing), sanitized env, timeout
   (`local_ocr_adapters.py:288-298`), constant `git` args
   (`clouda_data/doctor/system.py:228`). No `os.system`, `eval`, `exec`.
5. **Deserialization**: `yaml.safe_load` only (9 sites); defusedxml only; no
   pickle/joblib in project code.
6. **Secrets**: no `.env`, no keys in tracked files (checked `git ls-files`,
   SBOM.json, greps for key formats). Runtime keys stored outside the repo
   (`pdfword/constants.py:14-22`). Redaction is systematic: `structured_log`
   (`operations.py:16-35`), browser-facing payloads (`clouda_lab/dashboard/
   security.py:15-80`), worker failure messages (`worker_api.py:208-210`),
   constant-time key comparison (`worker_api.py:481-492`).
7. **Auth coverage**: every mutating worker_api endpoint carries
   CSRF/session/admin/internal-key guards; `_authenticate` fails closed when
   `WORKER_API_KEY` unset; first-admin bootstrap requires per-deploy token;
   dev auth endpoints double-gate on `CLOUDA_AUTH_VERIFIER=fake` AND
   `CLOUDA_ENV in {development,test}` (`worker_api.py:341-344`).
8. **Path traversal**: strict ID regexes on job/user/task ids
   (`worker_api.py:57`, `clouda_lab/dashboard/security.py:13-26`), containment
   checks on every DB-stored path served, `validate_relative_components`
   rejects `..`/colons/reserved devices (`clouda_contracts/storage.py:27-44`).
9. **Loopback enforcement**: Lab CLI rejects non-loopback `--host`
   (`clouda_lab/cli.py:30-38`); `require_loopback` checks client IP as a
   global dependency (`dashboard/security.py:94-99`) so even a `0.0.0.0` bind
   403s remote clients; worker_api adds `TrustedHostMiddleware` defaulting to
   `127.0.0.1,localhost`; docs/redoc/openapi disabled on both apps.
10. **XSS/CORS**: no CORSMiddleware anywhere; dashboard JS builds DOM via
    `textContent` only; server-rendered HTML escapes user values including
    filenames (`app.py`, `ui_components.py:327-338`).
11. **Resource limits**: request-size middleware (Content-Length pre-check +
    streaming enforcement per upload); result-size cap; rate limiter
    (per-process, loudly acknowledged at `worker_api.py:136-140`).

## Hardening opportunities (not vulnerabilities)

| ID | Item | Location | Note |
|---|---|---|---|
| H1 | Lab `/api/lab/session` hands the action token to any loopback origin with no Host validation — DNS-rebinding from a browser on the host could read the token and call mutating endpoints (loopback + dev tool context) | `clouda_lab/dashboard/app.py:201-203` | Add TrustedHostMiddleware (as worker_api does) or require a same-origin header. Low practical risk |
| H2 | `torch.load(weights_only=False)` | `checkpoint_torch.py:47` | See Bandit section |
| H3 | Chunked (no Content-Length) bodies skip the request-size pre-check; per-file caps still enforced during streaming | `worker_api.py:149-164` | Low impact |
| H4 | PDF fully parsed before page-count cap enforced (bounded CPU DoS within size caps) | `worker_api.py:1223-1230,1409-1416` | Consider cheap header-only page count via pypdfium2 if it matters |
| H5 | Dataset download authenticity: sha256 verified only when the registry asset carries one | `downloader.py:259,334` | = CODE_AUDIT U3; pin digests per source |
| H6 | Guest result token accepted as URL query parameter | `worker_api.py:1519` | Single-use, not logged; consider header/body |
| H7 | `_validate_docx_file` reads whole ≤100 MiB file to inspect 2 bytes | `worker_api.py:533` | Use a bounded read |
| H8 | `guest_has_active_job` check-then-act across calls (concurrent guest trials can both pass) | `worker_api.py:1385-1386` | = CODE_AUDIT U8; needs unique constraint |

## Environment-dependent risks

- Rate limiter and in-process worker registries are per-process; multi-worker
  deployments (WEB_CONCURRENCY > 1) lose global limits — explicitly logged at
  startup.
- `.env.example` ships dev-friendly defaults (`CLOUDA_AUTH_VERIFIER=fake`,
  `CLOUDA_SESSION_COOKIE_SECURE=false`); all dev endpoints double-gate on env,
  documented for local use only.

## Not applicable / false positives

- Bandit B613/B105 (above); vendored `tools/` tree excluded by repo config;
  no tarfile usage; no eval/exec; no CORS to harden.

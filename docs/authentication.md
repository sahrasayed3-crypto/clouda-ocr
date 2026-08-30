# Authentication and Local Development

Clouda uses a provider-neutral identity boundary. The server verifies a Firebase/Google Identity compatible bearer token once, creates an internal user record, and returns an opaque HttpOnly application session cookie. The project database stores roles, status, limits, ownership, sessions, and audit events. It never stores user passwords.

## Modes

`AUTH_MODE` is the preferred selector:

- `local` enables the development helper only in development or test.
- `firebase_emulator` uses the Firebase Auth emulator with the normal Clouda
  session boundary.
- `firebase_staging` verifies tokens against a real, non-production Firebase
  project.

The legacy `CLOUDA_AUTH_VERIFIER` setting remains supported for compatibility.

## Local Fake Verifier

Use only for tests and local emulator work:

```powershell
$env:CLOUDA_ENV="development"
$env:AUTH_MODE="local"
$env:CLOUDA_AUTH_VERIFIER="fake"
$env:CLOUDA_FIREBASE_PROJECT_ID="local-test-project"
```

Fake token format for tests:

```text
fake:<project>:<provider>:<uid>:<email>:<verified|expired|revoked|disabled|unverified>:<display_name>
```

The fake verifier is strict and does not accept arbitrary tokens.

For local end-to-end UI/API testing with `CLOUDA_AUTH_VERIFIER=fake`, the FastAPI
development endpoints emulate provider-owned flows without storing passwords in the
project database:

- `POST /auth/dev/register`
- `POST /auth/dev/verify-email`
- `POST /auth/dev/login`
- `POST /auth/dev/google-login`
- `POST /auth/dev/password-reset/start`
- `POST /auth/dev/password-reset/complete`

These endpoints are unavailable when `CLOUDA_ENV=production`. They are not a
production identity provider; Firebase remains the production boundary.

## Firebase Emulator

The emulator uses the normal token-exchange and application-session flow
without a production service-account key:

```text
AUTH_MODE=firebase_emulator
CLOUDA_ENV=development
FIREBASE_AUTH_EMULATOR_HOST=127.0.0.1:9099
FIREBASE_PROJECT_ID=local-test-project
FIREBASE_CLIENT_API_KEY=placeholder-emulator-api-key
FIREBASE_CLIENT_AUTH_DOMAIN=localhost
FIREBASE_CLIENT_PROJECT_ID=local-test-project
```

## Firebase / Google Staging Setup

Use a dedicated staging Firebase project. Enable Email/Password and Google
providers, configure staging-only authorized domains and redirect origins, and
set email-verification and password-reset action URLs to staging origins. Do
not enable SMS/phone authentication for this configuration.

Supply Firebase Admin credentials outside Git, preferably through Application
Default Credentials already scoped to the staging project. Alternatively,
point `FIREBASE_SERVICE_ACCOUNT_JSON_PATH` to a local file outside the
repository. Never commit service-account JSON, private keys, refresh tokens,
or exported user data.

```text
AUTH_MODE=firebase_staging
CLOUDA_ENV=staging
CLOUDA_REQUIRE_VERIFIED_EMAIL=true
CLOUDA_FIREBASE_PROJECT_ID=clouda-staging-placeholder
FIREBASE_PROJECT_ID=clouda-staging-placeholder
FIREBASE_CLIENT_API_KEY=public-web-api-key-placeholder
FIREBASE_CLIENT_AUTH_DOMAIN=clouda-staging-placeholder.firebaseapp.com
FIREBASE_CLIENT_PROJECT_ID=clouda-staging-placeholder
FIREBASE_SERVICE_ACCOUNT_JSON_PATH=<outside-repository>/firebase-staging-service-account.json
```

Firebase web configuration values are not private credentials, but they must
remain environment-specific to prevent project confusion.

The optional live staging smoke tool is fail-closed and requires explicit
allowlisting:

```powershell
$env:CLOUDA_ALLOW_LIVE_FIREBASE_TESTS="1"
$env:CLOUDA_STAGING_FIREBASE_PROJECT_ALLOWLIST="clouda-staging-placeholder"
.\.venv311\Scripts\python.exe tools\firebase_staging_smoke.py
```

The tool avoids printing credentials or tokens. It does not prove email
delivery without a separate human observation in the staging mailbox.

## Sessions

Clients exchange the external identity token at `/auth/session`. The server sends only `clouda_session` as an HttpOnly cookie and returns a CSRF token for state-changing requests. A fresh login revokes any existing session cookie before issuing a new one, which prevents session fixation. Logout, logout-all, account deletion requests, privilege changes, and account disabling revoke sessions.

## Admin Bootstrap

Set `CLOUDA_FIRST_ADMIN_BOOTSTRAP_TOKEN` only for the first controlled bootstrap. Authenticate as the intended first admin, call `/admin/bootstrap` with `X-First-Admin-Token`, then remove the environment value and log in again.

## Rollback and production boundary

For local rollback, use `AUTH_MODE=local` with
`CLOUDA_ENV=development`. In staging, remove the Firebase settings and stop the
service; the backend fails closed. Production DNS, HTTPS termination, reverse
proxy policy, Redis, database migration, monitoring, managed secrets, and
incident response must be validated separately before public deployment.

# Key Router Architecture

The Key Router / Provider Account Router is a dormant foundation under
`pdfword/key_router/`. It is separate from:

- Model Router: chooses a model.
- Teacher Router: chooses teacher/escalation roles.
- Key Router: chooses an authorized provider account and environment-held API key.

Default mode is `KEY_ROUTER_MODE=disabled`. In this mode the existing OpenRouter path
continues unchanged and no account selection, reservation, or database side effect is
performed by legacy requests.

Integration point:

- `pdfword/openrouter_client.py` calls the compatibility facade only after checking mode.
- `disabled`: returns the original API key immediately.
- `shadow`: records a sanitized proposed decision, then uses the old client path.
- `active`: requires eligible account configuration and environment-held secret.

No provider network health check or paid request is executed by this implementation.

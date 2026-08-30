# Error Normalization

The normalized model covers authentication, permission, temporary rate limit, quota,
billing, invalid request/payload, unsupported capability, model availability, provider
availability/overload, connection/network, timeout-before-send, timeout-after-send,
invalid/empty/malformed response, refusal, safety rejection, and unknown errors.

Classification uses HTTP status together with sanitized provider code/message/scope and
headers. `Retry-After` accepts seconds or HTTP dates with a conservative configured
fallback. A 401 quarantines the account and requires manual restoration. A model-scoped
429 does not disable the account. Provider 5xx errors feed provider/model circuits.
Stored messages and fingerprints exclude raw responses, credentials, and payloads.

# Provider Adapters

The adapter contract implements configuration validation, request construction,
explicit transport dispatch, response parsing, error normalization, usage extraction,
Retry-After extraction, capability checks, and request redaction.

- Generic OpenAI-compatible adapter: implemented and mock-tested. Provider definitions
  exist for OpenRouter, Requesty, NVIDIA NIM, SambaNova, GitHub Models, SiliconFlow,
  DeepInfra, Fireworks, Together, Nebius, and Novita. None was live-tested; compatibility
  differences remain configuration and verification risks.
- Google Gemini native adapter: implemented and mock-tested with native request and usage
  shapes. Not live-tested.
- Alibaba DashScope, Hugging Face, Cloudflare Workers AI, and Replicate:
  configuration-ready only and intentionally excluded from active dispatch.

The production HTTP transport exists but was not executed during this task. Normal tests
inject fake transports.

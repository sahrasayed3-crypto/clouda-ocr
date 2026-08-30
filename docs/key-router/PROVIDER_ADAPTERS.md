# Provider Adapters

Configured provider registry:

- OpenRouter: implemented but untested without credentials, OpenAI-compatible.
- Google AI Studio / Gemini API: configuration-ready.
- Alibaba Model Studio / DashScope: configuration-ready.
- Requesty: implemented but untested without credentials, OpenAI-compatible.
- Hugging Face Inference Providers: configuration-ready.
- NVIDIA NIM: implemented but untested without credentials, OpenAI-compatible.
- SambaNova Cloud: implemented but untested without credentials, OpenAI-compatible.
- GitHub Models: implemented but untested without credentials, OpenAI-compatible.
- Cloudflare Workers AI: configuration-ready.
- SiliconFlow: implemented but untested without credentials, OpenAI-compatible.
- DeepInfra: implemented but untested without credentials, OpenAI-compatible.
- Fireworks AI: implemented but untested without credentials, OpenAI-compatible.
- Together AI: implemented but untested without credentials, OpenAI-compatible.
- Nebius AI Studio: implemented but untested without credentials, OpenAI-compatible.
- Replicate: configuration-ready.
- Novita AI: implemented but untested without credentials, OpenAI-compatible.

No adapter performs network calls in tests. Provider-specific API behavior remains
untested until credentials and explicit activation are supplied.

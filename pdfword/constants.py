OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_API = "https://openrouter.ai/api/v1/models"

MODEL_FAST = "openai/gpt-4o-mini"
MODEL_ACCURATE_PRIMARY = "anthropic/claude-sonnet-4.5"
ACCURATE_FALLBACKS = [
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-sonnet-4",
    "~anthropic/claude-sonnet-latest",
]

REQUEST_TIMEOUT = 180

SECRETS_DIR = __import__("os").getenv(
    "CLOUDA_STATE_DIR",
    __import__("os").path.join(
        __import__("os").path.expanduser("~"), ".clouda_pdf_word"
    ),
)
API_KEY_FILE = __import__("os").path.join(SECRETS_DIR, "openrouter_api_key.txt")
LEARNING_STORE_FILE = __import__("os").path.join(SECRETS_DIR, "ocr_learning_store.json")
LEGACY_API_KEY_FILE = __import__("os").path.join(".secrets", "openrouter_api_key.txt")

MODEL_ROUTER_DEEPSEEK_V3 = __import__("os").getenv(
    "MODEL_ROUTER_DEEPSEEK_V3", "deepseek/deepseek-chat"
)
MODEL_ROUTER_GPT5_MINI = __import__("os").getenv(
    "MODEL_ROUTER_GPT5_MINI", "openai/gpt-5-mini"
)
MODEL_ROUTER_GPT5 = __import__("os").getenv("MODEL_ROUTER_GPT5", "openai/gpt-5")
MODEL_ROUTER_GEMINI_PRO = __import__("os").getenv(
    "MODEL_ROUTER_GEMINI_PRO", "google/gemini-2.5-pro"
)
MODEL_ROUTER_CLAUDE_SONNET = __import__("os").getenv(
    "MODEL_ROUTER_CLAUDE_SONNET", "anthropic/claude-sonnet-4.5"
)
MODEL_ROUTER_STORE_FILE = __import__("os").path.join(
    SECRETS_DIR, "model_router_store.json"
)

VISION_MODEL_PRIORITY = [
    MODEL_FAST,
    MODEL_ROUTER_GPT5_MINI,
    MODEL_ROUTER_GPT5,
    MODEL_ROUTER_GEMINI_PRO,
    MODEL_ROUTER_CLAUDE_SONNET,
]

# Backward-compatible aliases for existing code/tests.
MODEL_ROUTER_DEEPSEEK = MODEL_ROUTER_DEEPSEEK_V3
MODEL_ROUTER_GPT41_MINI = MODEL_ROUTER_GPT5_MINI
MODEL_ROUTER_GEMINI_FLASH = MODEL_ROUTER_GEMINI_PRO

QUALITY_ROUTING_PROMPT = (
    "You are a strict OCR quality router. Analyze this page image and reply with exactly one token: "
    "CLEAR or COMPLEX. "
    "Return CLEAR only if the Arabic text is clearly readable and straightforward. "
    "Return COMPLEX if there is blur, low contrast, handwriting, corruption, faint text, stamps, "
    "or anything likely to reduce plain-text OCR accuracy. "
    "Do not classify a page as COMPLEX only because it contains a table or embedded image."
)

STRICT_EXTRACTION_PROMPT = (
    "You are an OCR transcription engine for Arabic documents. "
    "Transcribe visible document text into plain text in the SAME reading order. "
    "Do NOT summarize, explain, rewrite, or add any text that is not present in the image. "
    "Do NOT output introductions, conclusions, or generic filler text. "
    "You are strictly forbidden from writing placeholders like [unclear] or leaving blanks. "
    "If a word is blurred, infer it from immediate local Arabic grammatical and logical context only. "
    "Preserve numbers, punctuation, line breaks, headings, and simple list text as faithfully as possible. "
    "If a table is visible, do NOT rebuild it as a Markdown table and do NOT keep its grid layout; "
    "extract only surrounding readable text and any clearly linear cell text when useful, ignoring table structure. "
    "If an embedded picture or illustration is visible, do NOT describe it and do NOT extract its contents. "
    "Only use the page image for OCR when the page itself is scanned/image-only and the main document text is inside the image. "
    "Output only text-only content without code fences, Markdown tables, or image references."
)

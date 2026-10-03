# Configuration and provider notes

`load_config()` reads `.env` from the repository root with `python-dotenv`. Existing process environment variables take precedence over `.env` values. The default provider is `openai` with model `gpt-4o-mini`; API keys are optional so offline agents and benchmarks can load configuration without credentials.

Environment variables:

- Main model: `LLM_PROVIDER`, `LLM_MODEL`, `LLM_TEMPERATURE`
- Judge model: `JUDGE_PROVIDER`, `JUDGE_MODEL`, `JUDGE_TEMPERATURE`
- Credentials: `OPENAI_API_KEY`, `CUSTOM_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OLLAMA_API_KEY`, `OPENROUTER_API_KEY`
- Optional endpoints: `OPENAI_BASE_URL`, `CUSTOM_BASE_URL`, `OLLAMA_BASE_URL`, `OPENROUTER_BASE_URL`
- Compact memory: `COMPACT_THRESHOLD_TOKENS` (default `900`) and `COMPACT_KEEP_MESSAGES` (default `6`)

Supported canonical providers are `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`. `normalize_provider()` also accepts the aliases `anthorpic`, `google`, `google-genai`, and `open_router`. Unknown names raise `ValueError` immediately with the supported names. Provider integrations are imported only when `build_chat_model()` is called, so loading configuration does not require an API key or start a live model.

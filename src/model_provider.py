from __future__ import annotations

from dataclasses import dataclass


_PROVIDER_ALIASES = {
    "anthorpic": "anthropic",  # Common transposition typo from the lab prompt.
    "google": "gemini",
    "google-genai": "gemini",
    "open_router": "openrouter",
}
_SUPPORTED_PROVIDERS = {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}


@dataclass
class ProviderConfig:
    """Provider settings shared by the agents and judge."""

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Return a supported canonical provider name or raise a clear error."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Provider must be one of: " + ", ".join(sorted(_SUPPORTED_PROVIDERS)))
    provider = value.strip().lower()
    provider = _PROVIDER_ALIASES.get(provider, provider)
    if provider not in _SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown provider {value!r}. Supported providers: "
            + ", ".join(sorted(_SUPPORTED_PROVIDERS))
        )
    return provider


def build_chat_model(config: ProviderConfig):
    """Build the selected LangChain chat model, importing its integration lazily."""
    provider = normalize_provider(config.provider)
    common = {"model": config.model_name, "temperature": config.temperature}

    try:
        if provider in {"openai", "custom"}:
            from langchain_openai import ChatOpenAI

            if provider == "custom" and not config.base_url:
                raise ValueError("Provider 'custom' requires CUSTOM_BASE_URL.")
            options = {**common}
            if config.api_key:
                options["api_key"] = config.api_key
            if config.base_url:
                options["base_url"] = config.base_url
            return ChatOpenAI(**options)

        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            options = {"model": config.model_name, "temperature": config.temperature}
            if config.api_key:
                options["google_api_key"] = config.api_key
            return ChatGoogleGenerativeAI(**options)

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            options = {**common}
            if config.api_key:
                options["api_key"] = config.api_key
            return ChatAnthropic(**options)

        if provider == "ollama":
            from langchain_ollama import ChatOllama

            options = {**common}
            if config.base_url:
                options["base_url"] = config.base_url
            return ChatOllama(**options)

        from langchain_openrouter import ChatOpenRouter

        options = {**common}
        if config.api_key:
            options["api_key"] = config.api_key
        if config.base_url:
            options["base_url"] = config.base_url
        return ChatOpenRouter(**options)
    except ImportError as exc:
        raise ImportError(
            f"Provider {provider!r} requires its LangChain integration package. "
            "Install the dependencies listed in README.md."
        ) from exc

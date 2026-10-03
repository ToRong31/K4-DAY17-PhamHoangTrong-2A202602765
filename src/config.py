from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared paths, compact-memory settings, and model configurations."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    load_dotenv(dotenv_path=root / ".env", override=False)

    provider = normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "custom": os.getenv("CUSTOM_API_KEY"),
        "gemini": os.getenv("GEMINI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "ollama": os.getenv("OLLAMA_API_KEY"),
        "openrouter": os.getenv("OPENROUTER_API_KEY"),
    }
    base_urls = {
        "openai": os.getenv("OPENAI_BASE_URL"),
        "custom": os.getenv("CUSTOM_BASE_URL"),
        "gemini": os.getenv("GEMINI_BASE_URL"),
        "anthropic": os.getenv("ANTHROPIC_BASE_URL"),
        "ollama": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "openrouter": os.getenv("OPENROUTER_BASE_URL"),
    }

    model = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        api_key=api_keys.get(provider),
        base_url=base_urls.get(provider),
    )

    judge_provider = normalize_provider(os.getenv("JUDGE_PROVIDER", provider))
    judge_model = ProviderConfig(
        provider=judge_provider,
        model_name=os.getenv("JUDGE_MODEL", model_name).strip(),
        temperature=float(os.getenv("JUDGE_TEMPERATURE", "0")),
        api_key=api_keys.get(judge_provider) or model.api_key,
        base_url=base_urls.get(judge_provider) or model.base_url,
    )

    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=int(os.getenv("COMPACT_THRESHOLD_TOKENS", "900")),
        compact_keep_messages=int(os.getenv("COMPACT_KEEP_MESSAGES", "6")),
        model=model,
        judge_model=judge_model,
    )

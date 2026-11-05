import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODELS = {
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-4o-mini",
    "groq": "openai/gpt-oss-120b",  # free tier, no credit card
}
KEY_VARS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "groq": "GROQ_API_KEY"}


@dataclass(frozen=True)
class Settings:
    provider: str
    api_key: str
    model: str
    timeout: float
    max_history: int


def get_settings() -> Settings:
    """Read env on each call so tests / key rotation don't need a restart."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    key_var = KEY_VARS.get(provider, "OPENAI_API_KEY")
    return Settings(
        provider=provider,
        api_key=os.getenv(key_var, "").strip(),
        model=os.getenv("LLM_MODEL", DEFAULT_MODELS.get(provider, "")).strip(),
        timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "20")),
        max_history=int(os.getenv("MAX_HISTORY_TURNS", "6")),
    )

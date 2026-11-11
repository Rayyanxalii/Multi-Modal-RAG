import os
from contextvars import ContextVar, Token
from typing import Any
from dotenv import load_dotenv
from llama_index.llms.groq import Groq

load_dotenv()

# Thread-safe / async-safe context variable holding the caller's Groq API key
_user_groq_key_ctx: ContextVar[str | None] = ContextVar("user_groq_key_ctx", default=None)

# Cache initialized LLM instances by API key to avoid reconstruction overhead
_LLM_CACHE: dict[str, Groq] = {}
_VISION_CACHE: dict[str, Groq] = {}


def set_request_groq_key(key: str | None) -> Token:
    """Set the active Groq API key for the current request context."""
    clean = key.strip().strip("'\"") if key and isinstance(key, str) else None
    return _user_groq_key_ctx.set(clean)


def reset_request_groq_key(token: Token) -> None:
    """Reset the request context token back to previous state."""
    try:
        _user_groq_key_ctx.reset(token)
    except Exception:
        pass


def get_active_groq_key() -> str | None:
    """Retrieve active Groq API key (request-scoped user key first, then server env fallback)."""
    ctx_key = _user_groq_key_ctx.get()
    if ctx_key and ctx_key.strip():
        return ctx_key.strip().strip("'\"")
    env_key = os.getenv("GROQ_API")
    return env_key.strip().strip("'\"") if env_key else None


def get_llm(api_key: str | None = None) -> Groq:
    """Get standard text reasoning LLM (llama_index Groq client) with the provided or active key."""
    key = api_key or get_active_groq_key()
    if not key:
        raise ValueError(
            "Groq API key required. Please click the 🔑 API Key button in the top navigation to add your free Groq API key."
        )
    if key not in _LLM_CACHE:
        _LLM_CACHE[key] = Groq(
            model="openai/gpt-oss-20b",
            api_key=key,
            max_tokens=500,
        )
    return _LLM_CACHE[key]


def get_llm_vision(api_key: str | None = None) -> Groq:
    """Get multimodal vision LLM (llama_index Groq client) with the provided or active key."""
    key = api_key or get_active_groq_key()
    if not key:
        raise ValueError(
            "Groq API key required. Please click the 🔑 API Key button in the top navigation to add your free Groq API key."
        )
    if key not in _VISION_CACHE:
        _VISION_CACHE[key] = Groq(
            model="qwen/qwen3.8-27b",
            api_key=key,
            max_tokens=1000,
        )
    return _VISION_CACHE[key]


class DynamicLLMProxy:
    """Dynamic proxy that routes calls to a Groq client instantiated with the active request's API key.

    Enables complete thread safety and per-user BYOK (Bring Your Own Key) without modifying
    existing call sites in graph nodes or workflow orchestrators.
    """

    def __init__(self, is_vision: bool = False):
        self._is_vision = is_vision

    def _target(self) -> Groq:
        return get_llm_vision() if self._is_vision else get_llm()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target(), name)


llm = DynamicLLMProxy(is_vision=False)
llm_vision = DynamicLLMProxy(is_vision=True)

"""Provider wrapper using plain HTTPS (httpx) instead of vendor SDKs.
Maps every failure to a small set of safe, typed errors so nothing raw
(stack traces, keys, upstream bodies) can leak to the frontend."""
import httpx

from .config import get_settings


class LLMError(Exception):
    code = "llm_error"
    http_status = 502
    public_message = "The assistant is temporarily unavailable."


class LLMConfigError(LLMError):
    code = "service_not_configured"
    http_status = 503
    public_message = "The assistant is not configured correctly. Please contact the administrator."


class LLMTimeout(LLMError):
    code = "llm_timeout"
    http_status = 504
    public_message = "The assistant took too long to respond. Please try again."


class LLMRateLimited(LLMError):
    code = "llm_rate_limited"
    http_status = 429
    public_message = "The AI service is busy or out of quota. Please try again in a minute."


class LLMUpstreamError(LLMError):
    code = "llm_unavailable"
    http_status = 502
    public_message = "The assistant service is currently unavailable. Please try again shortly."


class LLMOutputError(LLMError):
    code = "invalid_model_output"
    http_status = 502
    public_message = "The assistant returned an invalid response. Please try again."


def complete(system: str, messages: list) -> str:
    s = get_settings()
    if not s.api_key:
        raise LLMConfigError()
    if s.provider == "anthropic":
        return _anthropic(s, system, messages)
    if s.provider == "openai":
        return _openai_compatible(s, system, messages, "https://api.openai.com/v1", json_mode=True)
    if s.provider == "groq":  # free tier; OpenAI-compatible API
        return _openai_compatible(s, system, messages, "https://api.groq.com/openai/v1", json_mode=False)
    raise LLMConfigError()


def _post(url: str, headers: dict, payload: dict, timeout: float) -> dict:
    try:
        r = httpx.post(url, headers=headers, json=payload, timeout=timeout)
    except httpx.TimeoutException:
        raise LLMTimeout() from None
    except httpx.HTTPError:
        raise LLMUpstreamError() from None
    if r.status_code in (401, 403):
        raise LLMConfigError()  # invalid / unauthorised key
    if r.status_code == 429:
        raise LLMRateLimited()
    if r.status_code >= 400:
        raise LLMUpstreamError()
    try:
        return r.json()
    except ValueError:
        raise LLMUpstreamError() from None


def _anthropic(s, system, messages) -> str:
    data = _post(
        "https://api.anthropic.com/v1/messages",
        {"x-api-key": s.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        {"model": s.model, "max_tokens": 400, "temperature": 0, "system": system, "messages": messages},
        s.timeout,
    )
    try:
        return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
    except (KeyError, TypeError, AttributeError):
        raise LLMUpstreamError() from None


def _openai_compatible(s, system, messages, base_url: str, json_mode: bool) -> str:
    payload = {
        "model": s.model,
        "max_completion_tokens": 2000,  # reasoning models spend part of this on thinking
        "messages": [{"role": "system", "content": system}, *messages],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    # Some reasoning models (gpt-5*, gpt-6*, o-series) reject a custom temperature.
    if not s.model.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")):
        payload["temperature"] = 0
    data = _post(
        f"{base_url}/chat/completions",
        {"Authorization": f"Bearer {s.api_key}", "Content-Type": "application/json"},
        payload,
        s.timeout,
    )
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        raise LLMUpstreamError() from None

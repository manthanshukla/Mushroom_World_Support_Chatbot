import json
import logging
import re
import time

from pydantic import ValidationError

from . import llm
from .config import get_settings
from .prompts import PROMPT_MARKER, SYSTEM_PROMPT
from .retrieval import Hit, retrieve
from .schemas import ChatRequest, ChatResponse, LLMOutput

log = logging.getLogger("support")

NOT_FOUND_TEXT = "I could not find this information in the available product documentation."
REFUSED_TEXT = "I can't help with that request. I can only answer questions about the product documentation."
GREETING_TEXT = (
    "Hello! I'm the product support assistant. I can help with password resets, "
    "two-factor authentication, and subscription cancellation. What would you like to know?"
)
_GREETING_RE = re.compile(
    r"^\s*(hi+|hello+|hey+|hola|good (morning|afternoon|evening)|what can you (do|help with))\W*$",
    re.I,
)

INJECTION_PATTERNS = [
    r"ignore (all |any |your |the |previous |prior |above )*(instructions|rules|prompt)",
    r"(disregard|forget|override) (all |any |your |the |previous |prior )*(instructions|rules|prompt)",
    r"(reveal|show|print|display|repeat|leak|give me).{0,30}(system prompt|instructions|api key|secret|configuration)",
    r"system prompt",
    r"api[ _-]?key",
    r"developer (message|mode)",
    r"jailbreak",
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.I)
_SECRET_RE = re.compile(r"sk-[A-Za-z0-9_\-]{10,}")
_TAG_RE = re.compile(r"</?\s*(user_question|documentation)\s*>", re.I)
HISTORY_CHAR_LIMIT = 500


class OutputError(Exception):
    pass


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text))


def _clean(text: str) -> str:
    return _TAG_RE.sub("", text).strip()


def build_messages(req: ChatRequest, hits: list, max_history: int) -> list:
    """Bounded history (last N turns, each truncated) + context-wrapped current question.
    Provider APIs require alternating roles starting with 'user', so normalise first."""
    turns = [
        {"role": t.role, "content": _clean(t.content)[:HISTORY_CHAR_LIMIT]}
        for t in req.conversation[-max_history:]
        if t.content.strip()
    ]
    merged: list = []
    for t in turns:
        if merged and merged[-1]["role"] == t["role"]:
            merged[-1]["content"] += "\n" + t["content"]
        else:
            merged.append(dict(t))
    while merged and merged[0]["role"] != "user":
        merged.pop(0)
    if merged and merged[-1]["role"] == "user":  # current question follows; keep alternation
        merged.pop()

    docs = "\n\n".join(f"## {h.title}\n{h.section.content}" for h in hits)
    final = f"<documentation>\n{docs}\n</documentation>\n\n<user_question>\n{_clean(req.message)}\n</user_question>"
    return [*merged, {"role": "user", "content": final}]


def parse_output(raw: str, allowed_sources: set) -> LLMOutput:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise OutputError("no JSON object found")
    try:
        out = LLMOutput.model_validate(json.loads(text[start : end + 1]))
    except (json.JSONDecodeError, ValidationError) as e:
        raise OutputError(f"schema violation: {type(e).__name__}") from None
    unknown = set(out.sources) - allowed_sources
    if unknown:
        raise OutputError("sources must be titles from <documentation>")
    return out


def _leaks(answer: str, api_key: str) -> bool:
    return (
        PROMPT_MARKER in answer
        or bool(_SECRET_RE.search(answer))
        or (bool(api_key) and api_key in answer)
    )


def _respond(answer, status, sources, confidence) -> ChatResponse:
    return ChatResponse(answer=answer, status=status, sources=sources, confidence=confidence)


def _answer(req: ChatRequest) -> ChatResponse:
    s = get_settings()

    # 0. Greetings get a friendly welcome (no model call).
    if _GREETING_RE.match(req.message):
        return _respond(GREETING_TEXT, "answered", [], 1.0)

    # 1. Cheap deterministic guard for obvious injection / secret-extraction attempts.
    if looks_like_injection(req.message):
        return _respond(REFUSED_TEXT, "refused", [], 0.0)

    # 2. Retrieval. Nothing relevant -> don't call the LLM at all (no hallucination, no cost).
    prev_user = next((t.content for t in reversed(req.conversation) if t.role == "user"), "")
    hits: list[Hit] = retrieve(req.message, prev_user)
    if not hits:
        return _respond(NOT_FOUND_TEXT, "not_found", [], 0.1)

    # 3. LLM call with one corrective retry on malformed output.
    messages = build_messages(req, hits, s.max_history)
    allowed = {h.title for h in hits}
    out = None
    for attempt in range(2):
        raw = llm.complete(SYSTEM_PROMPT, messages)
        try:
            out = parse_output(raw, allowed)
            break
        except OutputError as e:
            log.warning("invalid model output (attempt %d): %s", attempt + 1, e)
            messages = [
                *messages,
                {"role": "assistant", "content": (raw or "(empty)")[:800]},
                {"role": "user", "content": f"Invalid reply ({e}). Reply again with ONLY the JSON object."},
            ]
    if out is None:
        raise llm.LLMOutputError()

    # 4. Output-side leak check.
    if _leaks(out.answer, s.api_key):
        log.warning("blocked response containing sensitive content")
        return _respond(REFUSED_TEXT, "refused", [], 0.0)

    if out.status == "not_found":
        return _respond(NOT_FOUND_TEXT, "not_found", [], min(out.confidence, 0.3))
    return _respond(out.answer, "answered", out.sources, round(out.confidence, 2))


def answer_question(req: ChatRequest) -> ChatResponse:
    t0 = time.perf_counter()
    resp = _answer(req)
    resp.latency_ms = int((time.perf_counter() - t0) * 1000)
    return resp
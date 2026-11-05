"""Basic keyword retrieval. Deliberately simple and deterministic: no vector DB needed
for 3 sections. See README for how this would be swapped for embeddings in production."""
import re
from dataclasses import dataclass

from .kb import KNOWLEDGE_BASE, Section

STOPWORDS = {
    "a", "an", "the", "i", "me", "my", "we", "our", "you", "your", "it", "is", "are", "was", "be",
    "do", "does", "did", "how", "what", "when", "where", "why", "which", "who", "can", "could",
    "would", "should", "will", "to", "of", "in", "on", "for", "and", "or", "if", "this", "that",
    "with", "from", "about", "please", "help", "need", "want", "product", "app", "application",
    "user", "users", "get", "have", "has", "there", "any", "tell", "show",
}
MIN_SCORE = 2  # a single weak body-text match is not enough to count as relevant
TOP_K = 2


@dataclass(frozen=True)
class Hit:
    section: Section
    score: int

    @property
    def title(self) -> str:
        return self.section.title


def _stem(tok: str) -> str:
    for suffix in ("ing", "es", "ed", "s"):
        if tok.endswith(suffix) and len(tok) - len(suffix) >= 4:
            return tok[: -len(suffix)]
    return tok


def tokenize(text: str) -> set:
    return {_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS}


def _score(query: set, section: Section) -> int:
    title, kw, body = tokenize(section.title), tokenize(" ".join(section.keywords)), tokenize(section.content)
    return sum(3 * (t in title) + 2 * (t in kw) + 1 * (t in body) for t in query)


def _rank(query: set) -> list:
    scored = sorted((Hit(s, _score(query, s)) for s in KNOWLEDGE_BASE), key=lambda h: -h.score)
    if not scored or scored[0].score < MIN_SCORE:
        return []
    top = scored[0].score
    return [h for h in scored[:TOP_K] if h.score >= MIN_SCORE and h.score >= top * 0.5]


def retrieve(message: str, previous_user_message: str = "") -> list:
    """Score sections against the message. If it matches nothing on its own, retry with the
    previous user turn added so follow-ups ('how long is it valid?') still resolve.
    Returns [] when nothing is relevant."""
    query = tokenize(message)
    hits = _rank(query)
    if not hits and previous_user_message:
        hits = _rank(query | tokenize(previous_user_message))
    return hits

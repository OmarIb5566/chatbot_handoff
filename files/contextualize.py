"""
Follow-up resolution: turn "in the self execution process" into a standalone query.

Sits between the user's question and the retriever. Nothing downstream changes -
retrieval still runs on every turn, on either the raw query or a rewritten one.

    user question
        |  (Arabic? translate to English first - translate.py)
        v
    contextualize()            <- THIS MODULE
        |   1. session history: last N turns for this session
        |   2. gate: does this even look like a follow-up?   (no model call)
        |   3. rewrite: only if the gate fires               (one model call)
        v
    retriever.search(query, boost_prefixes=...)
        v
    generation (qwen3:14b)

Three deliberate properties:

  * The gate is cheap and runs first. A self-contained question costs zero extra
    model calls, so the common case pays nothing for this feature at all.

  * The rewrite model is configured independently of the generation model, even
    though both currently point at qwen3:14b. Rewriting is a short structured
    transformation and looks like it should run on a 3B model; measurement said
    otherwise (see REWRITE_MODEL). Keeping the seam means re-testing a smaller
    model is a one-line change.

  * A failed rewrite degrades to the raw query, never to an error. Every guard
    in `_clean_rewrite` falls back rather than raising: a bad rewrite would
    otherwise turn a working question into a silent retrieval miss.

ON THE SOFT BOOST (and a conflict worth knowing about)
-------------------------------------------------------
When the resolved query names a document family, that is passed to the
retriever as `boost_prefixes` - an additive nudge on the fused score, not a
filter. Nothing is excluded.

But `Retriever.search(route=True)` is ALSO active by default, and that one IS a
hard pre-filter: `_candidates()` drops every chunk outside the routed family.
So today a query can be narrowed by the router and then nudged by the boost.
The boost cannot rescue a document the router already excluded.

These are two different answers to the same question and the repo currently
implements both. Deciding between them is a design call - `route=False` makes
metadata purely a soft signal, at the cost of the routing behaviour measured in
retriever.py's docstring. This module does not make that choice; it only adds
the soft channel and leaves the router as it found it.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field

import requests

OLLAMA_HOST = "http://localhost:11434"

# The rewrite model. Kept as its own constant, separate from the generation
# model, so it can be swapped without touching anything else - that separation
# is the point, not the specific value.
#
# Currently the same model as generation, chosen on measurement rather than
# principle. llama3.2 (3B) was tried first, being the right size class for a
# short structured task, and it substituted subjects: "how many hours must the
# PM respond" came back rewritten about QA. qwen3:14b resolves the same
# fragment correctly, and costs ~6s against ~2.7s. It is also already resident
# for generation, so this adds no second model to VRAM.
#
# Phi-3.5-mini, which the brief asked for, is not pulled here. To try it:
#   ollama pull phi3.5   then   REWRITE_MODEL = "phi3.5:latest"
# and re-check the PM/QA case above - a small model failing it is the reason
# _keeps_user_terms() exists.
REWRITE_MODEL = "qwen3:14b"

MAX_TURNS = 4              # rolling window, within the 3-5 asked for
ANSWER_SUMMARY_CHARS = 200  # store a gist, never the full generation


# ------------------------------------------------------------------ history
@dataclass
class Turn:
    query: str
    answer_summary: str


@dataclass
class Session:
    turns: deque = field(default_factory=lambda: deque(maxlen=MAX_TURNS))


class SessionStore:
    """In-memory rolling history, capped per session.

    Deliberately not persistent and deliberately not a transcript. Only the
    question and a short gist of the answer are kept, because the rewrite model
    needs to know what was being discussed, not what was said about it - and a
    full generation in the prompt would cost more tokens than the rewrite.
    """

    def __init__(self, max_turns: int = MAX_TURNS):
        self.max_turns = max_turns
        self._sessions: dict[str, Session] = {}

    def history(self, session_id: str) -> list[Turn]:
        s = self._sessions.get(session_id)
        return list(s.turns) if s else []

    def add(self, session_id: str, query: str, answer: str) -> None:
        s = self._sessions.setdefault(session_id, Session(deque(maxlen=self.max_turns)))
        s.turns.append(Turn(query=query, answer_summary=summarize_answer(answer)))

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


def summarize_answer(answer: str) -> str:
    """First sentence, hard-capped. No model call - this runs on every turn."""
    text = " ".join((answer or "").split())
    if not text:
        return ""
    m = re.search(r"(?<=[.!?])\s", text)
    first = text[:m.start() + 1] if m else text
    return first[:ANSWER_SUMMARY_CHARS].rstrip()


# --------------------------------------------------------------------- gate
# Openers that are grammatically fragments - they continue a previous question
# rather than starting a new one.
_FRAGMENT_STARTS = (
    "in the", "in ", "for the", "and for", "and in", "and the", "and ",
    "what about", "how about", "what if", "which one", "which of",
    "same for", "same in", "also for", "also in", "or the", "or ",
    "but the", "but ", "then ", "ok and", "okay and",
)

# A question that stands alone almost always has one of these. Their absence is
# evidence of a fragment, not proof - hence "signals", plural, below.
_WH_WORDS = ("who", "what", "when", "where", "why", "which", "how", "whose")
_VERBS = (
    "is", "are", "was", "were", "be", "does", "do", "did", "can", "could",
    "should", "shall", "must", "will", "would", "has", "have", "had",
    "approve", "approves", "sign", "signs", "send", "sends", "use", "uses",
    "require", "requires", "need", "needs", "happen", "happens", "prepare",
    "issue", "issues", "submit", "submits", "list", "lists", "define", "defines",
)

# Words that can only be resolved against something said earlier.
_ANAPHORA = ("it", "its", "that", "this", "those", "these", "them", "they",
             "he", "she", "his", "her", "there", "one", "ones", "same")

MAX_FRAGMENT_WORDS = 7


@dataclass
class GateResult:
    is_followup: bool
    reasons: list[str]


def looks_like_followup(query: str, has_history: bool) -> GateResult:
    """Cheap heuristic. No model call, no embedding, pure string work.

    Biased toward NOT firing. A false positive costs a rewrite call and risks
    mangling a good question; a false negative just leaves the query as the user
    typed it, which is the current behaviour anyway. So a single weak signal is
    not enough - a fragment opener alone fires, otherwise two signals are needed.
    """
    reasons: list[str] = []
    if not has_history:
        # Nothing to resolve against. The first question of a session is never
        # a follow-up, however fragmentary it looks.
        return GateResult(False, ["no history"])

    q = " ".join((query or "").lower().split())
    if not q:
        return GateResult(False, ["empty query"])
    words = re.findall(r"[a-z0-9']+", q)

    starts_fragment = q.startswith(_FRAGMENT_STARTS)
    if starts_fragment:
        reasons.append("starts with a fragment opener")
    if len(words) <= MAX_FRAGMENT_WORDS:
        reasons.append(f"short ({len(words)} words)")
    if not any(w in _WH_WORDS for w in words) and not any(w in _VERBS for w in words):
        reasons.append("no question word or verb")
    if any(w in _ANAPHORA for w in words):
        reasons.append("contains a back-reference")

    # A fragment opener is decisive on its own; otherwise require corroboration.
    fired = starts_fragment or len(reasons) >= 2
    return GateResult(fired, reasons)


# ------------------------------------------------------------------ rewrite
REWRITE_PROMPT = """You rewrite follow-up questions so they can be understood on their own.

Conversation so far:
{history}

Follow-up: {query}

Rewrite the follow-up as ONE standalone question. Carry over the subject from the
conversation and resolve any pronouns. Keep the user's own wording where you can.
Do not answer it. Do not explain. Do not add anything that was not asked.
Output only the rewritten question, on a single line.

Rewritten question:"""


def _format_history(turns: list[Turn]) -> str:
    if not turns:
        return "(none)"
    out = []
    for t in turns:
        out.append(f"Q: {t.query}")
        if t.answer_summary:
            out.append(f"A: {t.answer_summary}")
    return "\n".join(out)


_PREAMBLE_RE = re.compile(
    r"^\s*(?:the\s+)?(?:rewritten\s+question|standalone\s+question|question|answer|"
    r"rewrite|output)\s*[:\-–]\s*", re.I)


# Words a rewrite may legitimately drop: the fragment scaffolding itself.
_DROPPABLE = {
    "and", "or", "but", "then", "also", "same", "what", "about", "how", "which",
    "one", "of", "for", "in", "the", "a", "an", "to", "is", "are", "it", "that",
    "this", "those", "these", "them", "they", "ok", "okay", "so", "well", "please",
}


def _keeps_user_terms(rewrite: str, original: str) -> bool:
    """Did the rewrite keep every content word the user actually typed?

    Guards against the rewrite model *substituting* a subject rather than
    supplying a missing one. Observed with llama3.2: "and within how many hours
    must the PM respond" came back as "Within how many hours must QA respond
    ...", QA having been the subject of the previous answer. The user said PM.
    The corpus answers PM. The rewrite turned a good question into a miss, and
    nothing downstream could tell.

    Resolving a fragment means ADDING the omitted context, never overwriting
    what was stated. Anything the user typed that is not fragment scaffolding
    has to survive.
    """
    orig_terms = {w for w in re.findall(r"[a-z0-9']+", original.lower())
                  if len(w) > 1 and w not in _DROPPABLE}
    have = set(re.findall(r"[a-z0-9']+", rewrite.lower()))
    return orig_terms <= have


def _clean_rewrite(raw: str, fallback: str) -> str:
    """Coerce the model's output into a single query, or give up cleanly.

    Every branch here falls back to the original query rather than raising. A
    rewrite that goes wrong must not be able to turn a working turn into a
    retrieval miss - the worst acceptable outcome is "no rewrite happened".
    """
    if not raw:
        return fallback
    line = next((l.strip() for l in raw.strip().splitlines() if l.strip()), "")
    line = _PREAMBLE_RE.sub("", line).strip()
    if len(line) >= 2 and line[0] in "\"'“«" and line[-1] in "\"'”»":
        line = line[1:-1].strip()
    # A rewrite should be a question of roughly human length. Anything outside
    # that is the model explaining itself or answering instead of rewriting.
    if not line or len(line) > 300 or len(line.split()) < 3:
        return fallback
    if not _keeps_user_terms(line, fallback):
        # The rewrite dropped or replaced something the user said. Retrieval on
        # the raw fragment is worse than a good rewrite but far better than a
        # confident rewrite about the wrong subject.
        return fallback
    return line


def rewrite_query(query: str, turns: list[Turn], model: str = REWRITE_MODEL,
                  host: str = OLLAMA_HOST, timeout: int = 120) -> str:
    """Fragment + history -> standalone query. Falls back to `query` on failure."""
    prompt = REWRITE_PROMPT.format(history=_format_history(turns), query=query)
    payload = {
        "model": model, "prompt": prompt, "stream": False, "think": False,
        # Deterministic and short: this is a transformation, not generation.
        "options": {"temperature": 0, "seed": 0, "num_predict": 80, "num_ctx": 2048},
    }
    try:
        r = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
        if r.status_code == 400:
            payload.pop("think")
            r = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
        r.raise_for_status()
        return _clean_rewrite(r.json()["response"], query)
    except Exception:
        # Ollama down, model not pulled, timeout: retrieval still has to run.
        return query


# ------------------------------------------------------------- orchestration
@dataclass
class Contextualized:
    query: str                 # what retrieval should use
    original: str
    was_rewritten: bool
    gate: GateResult
    boost_prefixes: list[str]
    rewrite_model: str | None = None


def family_signal(query: str) -> list[str]:
    """Doc-code families the query names, as a SOFT signal for the retriever.

    Uses retriever.soft_prefixes, NOT route_prefixes. The latter returns [] as
    soon as two families match, which is right for a hard router and wrong
    here: a rewritten follow-up mentions two topics by construction ("...the
    Customer Satisfaction Survey in the self-execution process"), so the
    conservative version handed back an empty signal for exactly the queries
    this module produces.

    See the module docstring - the hard router is still separately live, and
    reconciling the two is a design decision this module does not make.
    """
    from retriever import soft_prefixes

    return soft_prefixes(query)


class Contextualizer:
    """Gate + rewrite, with its own model and its own session store."""

    def __init__(self, store: SessionStore | None = None,
                 rewrite_model: str = REWRITE_MODEL, host: str = OLLAMA_HOST,
                 enabled: bool = True):
        self.store = store or SessionStore()
        self.rewrite_model = rewrite_model
        self.host = host
        self.enabled = enabled

    def resolve(self, query: str, session_id: str | None) -> Contextualized:
        """Decide whether to rewrite, and produce the query retrieval will use."""
        turns = self.store.history(session_id) if session_id else []
        if not self.enabled or not session_id:
            # No session means no contextualisation AND no boost. Existing
            # callers - the notebook, eval_arabic.py, any single-shot ask() -
            # therefore retrieve exactly as they did before this module
            # existed. Opting in is what turns both on.
            gate = GateResult(False, ["contextualization off" if not self.enabled
                                      else "no session id"])
            return Contextualized(query, query, False, gate, [])

        gate = looks_like_followup(query, has_history=bool(turns))
        if not gate.is_followup:
            # The cheap path: straight to retrieval, no second model touched.
            return Contextualized(query, query, False, gate, family_signal(query))

        rewritten = rewrite_query(query, turns, model=self.rewrite_model, host=self.host)
        changed = rewritten.strip().lower() != query.strip().lower()
        return Contextualized(
            query=rewritten, original=query, was_rewritten=changed, gate=gate,
            boost_prefixes=family_signal(rewritten),
            rewrite_model=self.rewrite_model if changed else None,
        )

    def record(self, session_id: str | None, query: str, answer: str) -> None:
        """Append the turn. Stores the ORIGINAL question, not the rewrite, so the
        history reads the way the user actually spoke."""
        if session_id and self.enabled:
            self.store.add(session_id, query, answer)


if __name__ == "__main__":
    from translate import enable_utf8_stdout

    enable_utf8_stdout()

    print("--- gate (no model calls) ---")
    probes = [
        ("What form is used for the Customer Satisfaction Survey?", True),
        ("in the self execution process", True),
        ("what about procurement", True),
        ("and for steel?", True),
        ("which one applies to subcontractors", True),
        ("Who approves the RFQ and within how many days?", True),
        ("in the self execution process", False),   # no history -> never a follow-up
    ]
    for q, hist in probes:
        g = looks_like_followup(q, has_history=hist)
        print(f"  followup={str(g.is_followup):<5} hist={str(hist):<5} {q[:52]:<52} {g.reasons}")

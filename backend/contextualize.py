"""
Follow-up resolution: turn "in the self execution process" into a standalone query.

Sits between the user's question and the retriever.

    user question
        |  (Arabic? translate to English first - translate.py)
        v
    contextualize()            <- THIS MODULE
        |   1. session history: last N turns for this session
        |   2. gate: does this even look like a follow-up?   (no model call)
        |   3. classify: follow-up or continuation?          (one model call)
        |   4. rewrite: FOLLOWUP only                        (one model call)
        v
    FOLLOWUP / new question         CONTINUATION
    retriever.search(...)           reuse the last turn's chunks - no retrieval
        v                               v
    generation (qwen3:14b)          re-prompt from the SAME chunks

THE THIRD OUTCOME
-----------------
The gate firing used to mean one thing: rewrite, then retrieve. But two very
different messages get past it, and they need opposite handling:

  * "what about procurement" is a NEW question missing its subject. Rewriting it
    into a standalone query is exactly right, and retrieval on that query is
    where the answer comes from.

  * "you missed a few, list the rest" is not a question at all. It says the
    previous answer was incomplete. There is nothing to rewrite into a
    standalone query - "list the rest" has no subject to restore - and
    retrieval is not the failing step: the chunks were already there, the
    generator just did not exhaust them. Retrieving on this text would search
    the corpus for the words "list the rest" and quietly swap the context out
    from under the very answer the user is asking to be completed.

So the gate outcome is three-way. The heuristic still runs first and unchanged
(cheap, biased toward not firing); only when it fires does a second small model
call split follow-up from continuation. A continuation replays the stored chunks
from the previous turn and re-prompts the generator against them.

This is not a general licence to skip retrieval. It is relaxed for exactly the
case where the retrieval that matters has already happened and its result is
being deliberately reused.

THE GATE IS AN EXEMPTION, NOT A DETECTOR
-----------------------------------------
The gate is the only way to reach the classifier, so anything it misses is a
continuation served as a fresh retrieval. Tuned as a fragment detector it missed
half of them: "there were more than three, finish the list" is a grammatical,
verb-bearing sentence of ordinary length, which is exactly what a
fragment-hunting heuristic reads as "self-contained question".

Hunting harder does not fix that - it just moves the hole one phrasing further
out, which is the same objection that rules out a list of continuation phrases.
So the question is asked the other way round: fire by default, and exempt only
a message that visibly does not need the history. See `looks_like_followup`.

The cost this pays is latency on questions that could have skipped the check,
and the reason it is affordable is that the classifier now sits behind the gate:
firing is no longer the same as rewriting. A false positive costs two short
model calls and changes no answer. The zero-cost path still exists, for an
ordinary question of four words or more opening with a wh-word or an auxiliary.

Measured over the 136 standalone questions in eval_set.json and eval_set_v2.json
- none of which should fire, since all of them stand alone:

    fire by default, membership tests only          35/136   26%
    + fronted adverbials, determiner this/that,
      existential there, acronyms, local antecedents  2/136    1%

Recall over 20 held-out phrasings not used to tune any of that: 20/20. The two
remaining false positives are "five or more years of service" and "additional
information", both caught by _RESIDUAL. "more" earns its place - "there were
more than three, finish the list" is the same two words in the same order - so
they stay, at a cost of one classifier call each.

Every one of those five context rules was a false positive before it was a
rule; they are in the probes at the bottom of this file to keep them fixed.

Four deliberate properties:

  * The gate is cheap and runs first: string work, no model, no embedding. A
    question that plainly stands on its own still reaches the retriever without
    touching a second model.

  * The classifier is a prompt, not a phrase list. "you missed some", "that's
    not all of them", "keep going", "there were more than that" share no
    vocabulary; anything matching on literal strings gets the ones the author
    thought of and nothing else. The cost of a prompt instead is one short
    model call, paid only when the gate has already fired.

  * The rewrite model is configured independently of the generation model, even
    though both currently point at qwen3:14b. Rewriting is a short structured
    transformation and looks like it should run on a 3B model; measurement said
    otherwise (see REWRITE_MODEL). Keeping the seam means re-testing a smaller
    model is a one-line change.

  * A failed rewrite degrades to the raw query, never to an error. Every guard
    in `_clean_rewrite` falls back rather than raising: a bad rewrite would
    otherwise turn a working question into a silent retrieval miss. The
    classifier degrades the same way, one step earlier: anything other than a
    clean CONTINUATION - empty output, junk, timeout, Ollama down, no stored
    chunks to replay - is read as FOLLOWUP, which is what this module did
    before the third outcome existed.

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
# and re-check the PM/QA case above - the rewrite probes in __main__ cover it,
# and a small model failing it is what they are there to catch.
REWRITE_MODEL = "qwen3:14b"

MAX_TURNS = 4              # rolling window, within the 3-5 asked for
ANSWER_SUMMARY_CHARS = 200  # store a gist, never the full generation
CLASSIFY_ANSWER_CHARS = 1200  # how much of the last answer the classifier sees


# ------------------------------------------------------------------ history
@dataclass
class Turn:
    """One exchange.

    `answer_summary` is what goes into a rewrite prompt - the rewrite model
    needs to know what was being discussed, not what was said about it.

    `answer` and `hits` exist only to serve a continuation, and are the reason
    the store keeps more than a gist. Completing an incomplete answer needs the
    answer verbatim (to know which items are already listed) and the chunks it
    was built from (to find the ones that are not). Both default empty, so a
    Turn constructed the old way still works and simply cannot be continued.
    """

    query: str
    answer_summary: str
    answer: str = ""
    hits: list[dict] = field(default_factory=list)


@dataclass
class Session:
    turns: deque = field(default_factory=lambda: deque(maxlen=MAX_TURNS))


class SessionStore:
    """In-memory rolling history, capped per session.

    Deliberately not persistent and deliberately not a transcript. Rewriting
    reads only the question and a short gist of the answer, because a full
    generation in the prompt would cost more tokens than the rewrite.

    The full answer and the retrieved hits are kept alongside that gist, unused
    by the rewrite path, because a continuation has to replay the previous turn
    rather than describe it. Bounded by the same rolling window: at MAX_TURNS
    per session this is a handful of chunks, not a transcript.
    """

    def __init__(self, max_turns: int = MAX_TURNS):
        self.max_turns = max_turns
        self._sessions: dict[str, Session] = {}

    def history(self, session_id: str) -> list[Turn]:
        s = self._sessions.get(session_id)
        return list(s.turns) if s else []

    def add(self, session_id: str, query: str, answer: str,
            hits: list[dict] | None = None) -> None:
        s = self._sessions.setdefault(session_id, Session(deque(maxlen=self.max_turns)))
        s.turns.append(Turn(query=query, answer_summary=summarize_answer(answer),
                            answer=answer or "", hits=list(hits or [])))

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

# A standalone question opens with one of these two classes: a wh-word, or an
# auxiliary in subject-verb inversion ("does the PM approve...").
_WH_WORDS = frozenset((
    "who", "what", "when", "where", "why", "which", "how", "whose", "whom",
))
_AUX = frozenset((
    "is", "are", "was", "were", "be", "am", "does", "do", "did", "can", "could",
    "should", "shall", "must", "will", "would", "has", "have", "had", "may",
    "might",
))

# Base-form verbs that open an IMPERATIVE, not a question. "list the rest",
# "sign the form". An imperative is a perfectly good standalone request, but it
# is also the shape a continuation takes, so it does not earn the exemption.
_IMPERATIVES = frozenset((
    "approve", "sign", "send", "use", "require", "need", "prepare", "issue",
    "submit", "list", "define", "give", "show", "tell", "name", "add",
    "continue", "finish", "complete", "keep", "repeat", "redo", "carry",
    "provide", "include", "expand", "elaborate", "check", "go", "state",
))

# Words that can only be resolved against something said earlier.
_ANAPHORA = frozenset((
    "it", "its", "that", "this", "those", "these", "them", "they",
    "he", "she", "his", "her", "there", "one", "ones", "same",
))

# Residual quantifiers. These are anaphora over SETS: "the rest" and "what
# else" are meaningless without a previously established set to be the rest of.
# Grammatically the same move as "it", which is why they belong here and not in
# a list of continuation phrases.
#
# "next" and "beyond" are deliberately absent - "what is the next step in the
# RFQ process" is an ordinary standalone question about this corpus.
_RESIDUAL = frozenset((
    "rest", "remainder", "remaining", "else", "other", "others", "another",
    "more", "further", "additional", "extra", "leftover", "besides",
))

# Words that point at the conversation rather than at the documents. A question
# about the corpus has no reason to mention an answer, a reply, or you.
#
# "before" and "above" are deliberately absent, both measured rather than
# guessed: "what must be signed before award" and "above what advance payment
# amount" are threshold questions about the process, not references to the
# previous turn. Between them they were most of this set's false positives.
_DISCOURSE = frozenset((
    "you", "your", "yours", "youre", "youve", "u", "ur",
    "answer", "answers", "answered", "response", "reply", "replied",
    "said", "says", "told", "listed", "mentioned", "gave", "wrote", "output",
    "earlier", "previously", "already",
    "missed", "missing", "incomplete", "forgot", "forgotten", "skipped",
    "omitted", "truncated", "halfway", "wrong",
))

# "this"/"that" are pronouns only some of the time. "in this process" is a
# determiner pointing INTO the document; "that's not all" is a pronoun pointing
# back at the conversation. Same for existential "there's a gap between", which
# points at nothing at all. Distinguished below without a POS tagger.
_DETERMINERS = frozenset(("this", "that", "these", "those"))
_EXISTENTIAL_FOLLOWERS = frozenset((
    "is", "are", "was", "were", "has", "have", "had", "will", "would", "s", "re",
))
_NEGATIONS = frozenset(("not", "n't", "never", "nt"))

# How far into the query to look for the wh-word or auxiliary. Not just the
# first token: "Within how many days must the PM respond" is a fronted
# adverbial on a perfectly standalone question, and this corpus asks a lot of
# them.
STANDALONE_HEAD_WORDS = 6
# A pronoun this far into the query has a local antecedent to bind to. Tuned
# against the eval sets: below it, "how many days does it take" still counts as
# a back-reference; above it, "...the Payment Plan before it goes to Finance"
# no longer does.
LOCAL_ANTECEDENT_WORDS = 6
# At or below this length, nothing is self-contained enough to skip the check.
# A bare "why?" or "who signs?" after an answer is a follow-up every time, and
# both open like standalone questions.
MIN_STANDALONE_WORDS = 3


@dataclass
class GateResult:
    is_followup: bool
    reasons: list[str]


def _vocab(q: str) -> tuple[list[str], set[str]]:
    """Tokens, plus the stem of each contraction.

    "that's" and "you've" are one token to the tokeniser and would otherwise
    never match "that" or "you". Splitting on the apostrophe as well costs
    nothing and is pure recall.
    """
    words = re.findall(r"[a-z0-9']+", q)
    vocab = set(words)
    vocab.update(w.split("'")[0] for w in words if "'" in w)
    return words, vocab


def _anaphora_hits(words: list[str], acronyms: frozenset[str] = frozenset()) -> list[str]:
    """Back-references that actually point BACKWARDS, out of this query.

    Membership alone over-fires badly on this corpus - it was most of the false
    positives when this was first measured. Four cases are checked in context
    instead, each one found in the eval sets rather than guessed:

      * determiner "this"/"that" before a noun - "in this process", "that
        document" - points into the document, not at the conversation. The
        pronoun reading ("that's not all", "this is wrong") is what counts, and
        it shows up as a contraction or as the subject of a verb.
      * existential "there is/was/'s" - "when there's a gap between customer
        perception and..." - points at nothing earlier.
      * a pronoun with a LOCAL antecedent - "who approves the Monthly Payment
        Plan before it goes to Finance" - is resolved by its own sentence and
        needs no history. Approximated by how much sentence precedes it: a
        pronoun in the tail of a long question has something to bind to, a
        pronoun near the front of "how many days does it take" does not.
      * "IT" the department, which lowercases into the pronoun. Any token
        written in capitals in the original is an acronym, not a back-reference.
    """
    hits: list[str] = []
    # Only the head of the query is examined. Past that there is enough
    # sentence behind a pronoun for it to be bound locally, whichever kind it
    # is - "which form number is that", "the reward that can be released",
    # "before it goes to Finance" are all resolved without any history.
    for i, w in enumerate(words[:LOCAL_ANTECEDENT_WORDS]):
        stem, contracted = w.split("'")[0], "'" in w
        nxt = words[i + 1].split("'")[0] if i + 1 < len(words) else ""
        if stem in acronyms:
            continue
        if stem == "there":
            if not contracted and nxt not in _EXISTENTIAL_FOLLOWERS:
                hits.append(stem)
        elif stem in _DETERMINERS:
            # Determiner ("in this process") or pronoun ("that's not all")?
            # Decided by what sits either side, since there is no POS tagger:
            # a contraction, a following verb, or an auxiliary in front with
            # something after it ("does that take") is the pronoun. Bare at the
            # end of the sentence ("which form number is that?") is a pronoun
            # too, but one this query answers itself, so it does not count.
            prev = words[i - 1].split("'")[0] if i else ""
            if contracted or nxt in _AUX or nxt in _NEGATIONS or (prev in _AUX and nxt):
                hits.append(stem)
        elif stem in _ANAPHORA:
            hits.append(stem)
    return sorted(set(hits))


def looks_like_followup(query: str, has_history: bool) -> GateResult:
    """Cheap heuristic. No model call, no embedding, pure string work.

    This asks the inverse of what it used to. The old version hunted for
    evidence that a message WAS a fragment and needed two signals to fire, which
    meant every phrasing nobody had thought of was missed by default - and
    continuation complaints ("there were more than three, finish the list") are
    grammatical, verb-bearing sentences of ordinary length, so they sailed
    straight through. A blacklist of continuation phrases would have had the
    same hole one phrasing further out.

    So the test is now an exemption, and the default is to fire. A message is
    let through only if it opens like a standalone question - wh-word or
    inverted auxiliary - AND names nothing it cannot resolve on its own: no
    fragment opener, no back-reference, no residual quantifier, no reference to
    the conversation. Everything else goes to the classifier. A word missing
    from those sets no longer causes a miss on its own; it only fails to revoke
    an exemption the message had to earn some other way first.

    What this costs, and why it is affordable now: firing is no longer the same
    as rewriting. The classifier runs first and sends anything that is not a
    complaint down the ordinary path, and a rewrite that resolves an already
    self-contained question mostly returns it unchanged. So a false positive
    here costs two short model calls and rarely changes an answer. That was not true when this function was
    written - back then firing meant rewriting, and a bad rewrite meant a silent
    retrieval miss, which is why it was tuned to stay quiet.

    The zero-cost path survives for the shape it was meant for: an ordinary
    question of four words or more, opening with a wh-word or an auxiliary,
    still reaches the retriever without touching a model.
    """
    if not has_history:
        # Nothing to resolve against. The first question of a session is never
        # a follow-up, however fragmentary it looks.
        return GateResult(False, ["no history"])

    q = " ".join((query or "").lower().split())
    if not q:
        return GateResult(False, ["empty query"])
    words, vocab = _vocab(q)

    starts_fragment = q.startswith(_FRAGMENT_STARTS)
    # Capitals in the ORIGINAL, before lowercasing folded "IT" into "it".
    acronyms = frozenset(w.lower() for w in re.findall(r"\b[A-Z]{2,}\b", query or ""))
    anaphora = _anaphora_hits(words, acronyms)
    residual = sorted(vocab & _RESIDUAL)
    discourse = sorted(vocab & _DISCOURSE)
    # A wh-word or auxiliary in the head of the query, or opening any later
    # clause - "For projects >= 50M USD, how many days before submission" is one
    # question with the interrogative in its second half, and "Many processes
    # reference a MOM form. Which form number is it?" is two sentences.
    heads = [words[:STANDALONE_HEAD_WORDS]]
    heads += [re.findall(r"[a-z0-9']+", c)[:1] for c in re.split(r"[,;.]", q)[1:]]
    opens_standalone = any(w in _WH_WORDS or w in _AUX for h in heads for w in h)
    long_enough = len(words) > MIN_STANDALONE_WORDS

    reasons: list[str] = []
    if starts_fragment:
        reasons.append("starts with a fragment opener")
    if anaphora:
        reasons.append(f"back-reference ({', '.join(anaphora)})")
    if residual:
        reasons.append(f"refers to a remainder ({', '.join(residual)})")
    if discourse:
        reasons.append(f"refers to the conversation ({', '.join(discourse)})")
    if not opens_standalone:
        opener = words[0] if words else ""
        reasons.append(f"imperative opener ({opener})" if opener in _IMPERATIVES
                       else "does not open like a standalone question")
    if not long_enough:
        reasons.append(f"very short ({len(words)} words)")

    exempt = (opens_standalone and long_enough and not starts_fragment
              and not anaphora and not residual and not discourse)
    if exempt:
        return GateResult(False, ["opens like a standalone question, "
                                  "nothing to resolve against history"])
    return GateResult(True, reasons)


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


def _clean_rewrite(raw: str, fallback: str) -> str:
    """Coerce the model's output into a single query, or give up cleanly.

    Every branch here falls back to the original query rather than raising. A
    rewrite that goes wrong must not be able to turn a working turn into a
    retrieval miss - the worst acceptable outcome is "no rewrite happened".

    WHAT IS CHECKED, AND WHAT DELIBERATELY IS NOT
    ---------------------------------------------
    These are FORMAT checks: a preamble, a wrapping quote, a length outside
    human range. All of them catch the model explaining itself or answering
    instead of rewriting, and none of them needs to know a single English word.

    There used to be a semantic one too - `_keeps_user_terms`, which required
    every content word the user typed to survive into the rewrite, with a
    hand-written `_DROPPABLE` set naming the scaffolding a rewrite was allowed
    to shed. It was written against llama3.2 substituting a subject (see
    REWRITE_MODEL) and it is gone, for two measured reasons:

      * It could not tell scaffolding from content without that list, and the
        list only covered wh-fragments. Every imperative follow-up - "give me
        the details", "tell me more about it", "list everything it covers" -
        had its perfectly good rewrite rejected because `give` and `me` are not
        in the set, and retrieval then ran on a topic-free raw fragment. "give
        me the details provided in these documents" answered out of the Quality
        Manual after a turn entirely about the Internship Policy.

      * The list cannot be replaced by corpus statistics, which is the obvious
        fix and does not work: over this corpus `give` has idf 4.78 and `pm` -
        the exact term the guard existed to protect - has 1.83. Any threshold
        that drops the scaffolding drops the subject first.

    On qwen3:14b the guard also protected nothing. Removing it leaves the
    PM/QA case, "what about procurement", "who signs it" and "in the self
    execution process" byte-identical: the model preserves a stated subject
    unaided. It fired only on imperatives, and only ever wrongly.

    The seam it was guarding is real for a smaller model, so it is now held by
    the rewrite probes in __main__ rather than by a word list - swapping
    REWRITE_MODEL re-measures substitution instead of guessing at it.
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


# ----------------------------------------------------------------- classify
# Kept as its own prompt and its own call rather than folded into the rewrite,
# for the same reason REWRITE_MODEL is its own constant: the two are separate
# decisions and should be separately swappable. A combined "classify and, if
# follow-up, also rewrite" prompt would make either one impossible to change,
# re-measure or re-point at a different model without disturbing the other.
CLASSIFY_PROMPT = """You classify a follow-up message in a conversation as one of two types.

Previous question: {prev_query}
Previous answer: {prev_answer}
New message: {query}

FOLLOWUP: the new message asks something new that depends on earlier context to
understand (a missing subject, a pronoun, an implied topic).

CONTINUATION: the new message is telling you the previous answer was incomplete,
wrong, or cut off, and is asking you to add to, fix, or finish it — not asking
a new question.

Output exactly one word: FOLLOWUP or CONTINUATION. Nothing else.

Label:"""

_LABELS = {"FOLLOWUP", "CONTINUATION"}


def _clean_label(raw: str) -> str:
    """Coerce the model's output into a label, or give up cleanly.

    Same posture as `_clean_rewrite`: every branch falls back rather than
    raising, and the fallback is FOLLOWUP - the behaviour this module had
    before continuations existed. Misreading a continuation as a follow-up
    costs a rewrite and a retrieval; misreading a follow-up as a continuation
    answers a new question out of last turn's chunks. The defaults are not
    symmetric, so neither is the guard.
    """
    if not raw:
        return "FOLLOWUP"
    # Tokenised rather than whole-line, so a stray preamble ("Label: X") still
    # reads - the same leniency _clean_rewrite gives via _PREAMBLE_RE. A line
    # naming both labels is the model reasoning aloud rather than answering, so
    # it is ambiguous and takes the default.
    for line in raw.strip().splitlines():
        found = {w for w in re.findall(r"[A-Za-z]+", line.upper()) if w in _LABELS}
        if len(found) == 1:
            return found.pop()
        if found:
            return "FOLLOWUP"
    return "FOLLOWUP"


def classify_followup(query: str, turns: list[Turn], model: str = REWRITE_MODEL,
                      host: str = OLLAMA_HOST, timeout: int = 60) -> str:
    """FOLLOWUP or CONTINUATION. Never raises; degrades to FOLLOWUP."""
    if not turns:
        return "FOLLOWUP"
    prev = turns[-1]
    # The verbatim answer, capped: the classifier needs to see whether the new
    # message is complaining about THIS text, and the gist is too lossy for
    # that. The cap keeps the prompt inside num_ctx.
    prev_answer = " ".join((prev.answer or prev.answer_summary or "").split())
    prev_answer = prev_answer[:CLASSIFY_ANSWER_CHARS]
    prompt = CLASSIFY_PROMPT.format(prev_query=prev.query, prev_answer=prev_answer,
                                    query=query)
    payload = {
        "model": model, "prompt": prompt, "stream": False, "think": False,
        # One word out. Same determinism as the rewrite, a much smaller budget.
        "options": {"temperature": 0, "seed": 0, "num_predict": 8, "num_ctx": 2048},
    }
    try:
        r = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
        if r.status_code == 400:
            payload.pop("think")
            r = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
        r.raise_for_status()
        return _clean_label(r.json()["response"])
    except Exception:
        # Ollama down, model not pulled, timeout: the turn still has to be
        # answered, and the answerable path is the old one.
        return "FOLLOWUP"


# -------------------------------------------------------------- continuation
# Not a retrieval query - a generation prompt. The chunks are already chosen;
# what is being asked for is a second pass over them that excludes what the
# first pass already said.
CONTINUATION_PROMPT = """Source context:
{chunks}

Your previous answer was:
{prev_answer}

That answer was incomplete. Using only the source context above, list the
remaining items that were not included in the previous answer. Do not repeat
items already listed. Always cite the doc filename. If the source context
contains no further items, say so plainly rather than inventing any."""


# ------------------------------------------------------------- orchestration
@dataclass
class Contextualized:
    """What the caller should do with this turn.

    `query` is None for a continuation and only for a continuation - there is
    no query to run, and None makes that unmissable at the call site rather
    than something a caller has to remember to check. `needs_retrieval` is the
    flag to branch on; the None is the tripwire if anyone forgets.
    """

    query: str | None          # what retrieval should use; None = do not retrieve
    original: str
    was_rewritten: bool
    gate: GateResult
    boost_prefixes: list[str]
    rewrite_model: str | None = None
    mode: str = "new"          # "new" | "followup" | "continuation"
    reuse_hits: list[dict] = field(default_factory=list)
    prev_answer: str = ""

    @property
    def needs_retrieval(self) -> bool:
        return self.mode != "continuation"

    @property
    def is_continuation(self) -> bool:
        return self.mode == "continuation"


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

        # The gate fired. Only now is it worth a model call to find out which
        # kind of follow-up this is.
        label = classify_followup(query, turns, model=self.rewrite_model, host=self.host)
        prev = turns[-1]
        if label == "CONTINUATION" and prev.hits:
            # Replay, not retrieval. `prev.hits` empty means the previous turn
            # predates this feature or stored nothing to replay - there is
            # nothing to continue from, so fall through and treat it as a
            # follow-up, which at worst retrieves as it always did.
            gate = GateResult(True, gate.reasons + ["classified CONTINUATION"])
            return Contextualized(
                query=None, original=query, was_rewritten=False, gate=gate,
                boost_prefixes=[], mode="continuation",
                reuse_hits=list(prev.hits),
                prev_answer=prev.answer or prev.answer_summary,
            )

        rewritten = rewrite_query(query, turns, model=self.rewrite_model, host=self.host)
        changed = rewritten.strip().lower() != query.strip().lower()
        return Contextualized(
            query=rewritten, original=query, was_rewritten=changed, gate=gate,
            boost_prefixes=family_signal(rewritten),
            rewrite_model=self.rewrite_model if changed else None,
            mode="followup",
        )

    def record(self, session_id: str | None, query: str, answer: str,
               hits: list[dict] | None = None) -> None:
        """Append the turn. Stores the ORIGINAL question, not the rewrite, so the
        history reads the way the user actually spoke.

        `hits` are stored so a continuation of THIS turn can replay them. On a
        continuation the caller passes the reused hits straight back, so a
        second "and the rest?" continues from the same chunks rather than
        losing them."""
        if session_id and self.enabled:
            self.store.add(session_id, query, answer, hits=hits)


if __name__ == "__main__":
    from translate import enable_utf8_stdout

    enable_utf8_stdout()

    print("--- gate (no model calls) ---")
    # `want` is what the gate should do, not what the message is: True means
    # "spend a model call on this", which covers follow-ups and continuations
    # alike. False means "self-contained, go straight to retrieval".
    probes = [
        # must NOT fire - the zero-cost path
        ("What form is used for the Customer Satisfaction Survey?", True, False),
        ("Who approves the RFQ and within how many days?", True, False),
        ("Does the PM sign the subcontract?", True, False),
        ("Who approves the RFQ?", True, False),
        ("in the self execution process", False, False),  # no history -> never
        # the four context rules, each one a real false positive first
        ("Within how many hours must the PM respond?", True, False),   # fronted
        ("How often is the IT price list updated?", True, False),      # acronym
        ("What does 'Brown Field' mean in this process?", True, False),  # determiner
        ("Who approves the Payment Plan before it goes to Finance?", True, False),
        ("Several processes name a form. Which form number is that?", True, False),
        # fragments
        ("in the self execution process", True, True),
        ("what about procurement", True, True),
        ("and for steel?", True, True),
        ("which one applies to subcontractors", True, True),
        ("why?", True, True),
        ("who signs it", True, True),
        ("how long does that take", True, True),   # pronoun, not determiner
        # continuations - the phrasings the old two-signal rule missed
        ("you missed a few, list the rest", True, True),
        ("that's not all of them", True, True),
        ("keep going", True, True),
        ("there were more than three, finish the list", True, True),
        ("your answer got cut off halfway", True, True),
        ("incomplete — what else does the document name?", True, True),
    ]
    wrong = 0
    for q, hist, want in probes:
        g = looks_like_followup(q, has_history=hist)
        ok = g.is_followup == want
        wrong += not ok
        print(f"  {'ok  ' if ok else 'MISS'} fires={str(g.is_followup):<5} "
              f"hist={str(hist):<5} {q[:46]:<46} {g.reasons}")
    print(f"  {len(probes) - wrong}/{len(probes)} as expected")

    # ------------------------------------------------------------- classifier
    # One model call each, so this half only runs when Ollama is up.
    from chatbot import ollama_up

    if not ollama_up():
        print("\n--- classifier: skipped, ollama not reachable ---")
        raise SystemExit(0)

    print(f"\n--- classifier ({REWRITE_MODEL}, one call each) ---")
    _prev = Turn(
        query="Which documents are required to open a subcontract?",
        answer_summary="Three documents are required.",
        answer=("Three documents are required to open a subcontract: the "
                "Subcontract Agreement (F-PRO-01), the Bid Analysis Sheet "
                "(F-PRO-04) and the Insurance Certificate (F-PRO-09). "
                "[Subcontracts - Subcontract Preparation (UAE) V1.pdf]"),
        hits=[{"filename": "stub.pdf", "section": "stub", "text": "stub", "score": 1.0}],
    )
    # The continuation probes deliberately share no vocabulary with each other:
    # if a phrase list were doing the work here rather than the model, most of
    # these would miss. "missed", "rest" and "continue" appear at most once.
    classify_probes = [
        ("you missed a few, list the rest", "CONTINUATION"),
        ("that's not all of them", "CONTINUATION"),
        ("keep going", "CONTINUATION"),
        ("there were more than three, finish the list", "CONTINUATION"),
        ("your answer got cut off halfway", "CONTINUATION"),
        ("incomplete — what else does the document name?", "CONTINUATION"),
        ("what about procurement", "FOLLOWUP"),
        ("and for steel?", "FOLLOWUP"),
        ("who signs it", "FOLLOWUP"),
        ("in the self execution process", "FOLLOWUP"),
    ]
    # Two columns, because the classifier is not reached on its own. The gate
    # runs first in production and a message it rejects is never classified -
    # so `gate` here is as much a result as `label` is.
    print(f"  {'gate':<5} {'label':<13} {'want':<13} message")
    correct = blocked = wrong = 0
    for q, want in classify_probes:
        fired = looks_like_followup(q, has_history=True).is_followup
        label = classify_followup(q, [_prev]) if fired else "(not reached)"
        if not fired and want == "CONTINUATION":
            mark, blocked = "gate", blocked + 1
        elif label == want or (not fired and want == "FOLLOWUP"):
            mark, correct = "ok", correct + 1
        else:
            mark, wrong = "MISS", wrong + 1
        print(f"  {str(fired):<5} {label:<13} {want:<13} {q}   [{mark}]")
    print(f"\n  {correct}/{len(classify_probes)} routed correctly, {wrong} misclassified, "
          f"{blocked} never reached the classifier.")
    if blocked:
        print("  [gate] = the heuristic did not fire, so this continuation was\n"
              "  handled as it is today: rewritten and retrieved. See the gate\n"
              "  note in the module docstring - widening it is a separate call.")

    # ---------------------------------------------------------------- rewrite
    # What `_keeps_user_terms` used to assert, measured instead of assumed.
    #
    # Two properties, and they pull in opposite directions - which is exactly
    # why a single string rule could not hold both:
    #
    #   resolves    the rewrite must SUPPLY the subject the fragment omitted,
    #               or retrieval runs on a topic-free string. Every imperative
    #               row here failed this while the guard was in place.
    #   keeps       the rewrite must not REPLACE a subject the user stated.
    #               `pm_probe` is the llama3.2 failure that motivated the
    #               guard; on qwen3:14b the model holds it unaided.
    #
    # `want_in` is checked on the resolved query, lowercased. Re-pointing
    # REWRITE_MODEL at a smaller model and re-running this is the whole point:
    # a substitution shows up as a `keeps` failure, not as a silent miss.
    print(f"\n--- rewrite ({REWRITE_MODEL}, one call each) ---")
    _interns = Turn(
        query="what is the compensation for interns",
        answer_summary=('The compensation for interns is referred to as a "Grant," '
                        "defined as financial compensation based on attendance "
                        "and work hours."),
        answer="x",
        hits=[{"filename": "MHR19_Internship_Policy.pdf", "section": "s",
               "text": "t", "score": 1.0}],
    )
    _qa = Turn(
        query="What are the QA department's responsibilities in the self execution process?",
        answer_summary="The QA department reviews and approves the inspection requests.",
        answer="x",
        hits=[{"filename": "stub.pdf", "section": "s", "text": "t", "score": 1.0}],
    )
    rewrite_probes = [
        # (history, message, terms the resolved query must contain, must it change?)
        # imperatives - all three were rejected by the old guard on `give`/`me`
        ([_interns], "give me the details provided in these documents", ["intern"], True),
        ([_interns], "tell me more about it", ["intern"], True),
        ([_interns], "list everything it covers", ["intern"], True),
        # substitution - the user said PM, the history said QA
        ([_qa], "and within how many hours must the PM respond", ["pm"], True),
        # ordinary fragments, unchanged by the removal
        ([_qa], "what about procurement", ["procurement"], True),
        ([_qa], "in the self execution process", ["self execution"], True),
    ]
    good = bad = 0
    for turns, q, want_in, must_change in rewrite_probes:
        out = rewrite_query(q, turns)
        changed = out.strip().lower() != q.strip().lower()
        missing = [w for w in want_in if w not in out.lower()]
        ok = not missing and (changed or not must_change)
        good, bad = good + ok, bad + (not ok)
        why = "" if ok else ("  <- unresolved" if not changed else f"  <- lost {missing}")
        print(f"  {'ok  ' if ok else 'MISS'} {q[:44]:<44} -> {out[:72]}{why}")
    print(f"  {good}/{len(rewrite_probes)} resolved without dropping a stated term")

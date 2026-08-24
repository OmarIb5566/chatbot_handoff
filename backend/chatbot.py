"""
The question-answering pipeline, in one place.

This used to live inside a cell of demo_chatbot.ipynb. It moved here when the
Streamlit app arrived, because the alternative was two copies of the prompt
text, the reasoning-strip regex and the validation order - and those are
exactly the things that drift apart silently. The notebook and app.py now
import the same `Chatbot` and differ only in how they render the result.

The seam is deliberate: `Chatbot.ask()` returns a plain dict and prints
nothing. Presentation belongs to the caller, because a notebook wants a
latency breakdown and a web app wants a spinner.

Pipeline, for one question:

    is_arabic?  -> translate the question to English   (translate.py)
    follow-up?  -> rewrite into a standalone query     (contextualize.py)
    route + retrieve top-k                             (retriever.py)
    build_prompt, asking for an Arabic answer if the question was Arabic
    generate                                           (Ollama)
    strip reasoning traces
    validate cited form numbers against the corpus     (validator.py)

Retrieval always runs on English; see §5.5 of the notebook, or translate.py,
for why. The answer is generated in Arabic rather than translated into it.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

import validator as V
from contextualize import CONTINUATION_PROMPT, Contextualizer
from retriever import Retriever, route_prefixes
from translate import ANSWER_IN_ARABIC, REFUSAL_AR, is_arabic, to_arabic, to_english

from paths import CHUNKS_JSON, WORKFLOW_CHUNKS_JSON  # noqa: E402  (see paths.py)


def load_corpus(include_workflows: bool = True) -> list[dict]:
    """The indexed corpus: process chunks, plus workflow diagrams.

    Three kinds of chunk live here and they are not equally trustworthy.

      pdf_text         verbatim PDF text (adaptive_chunker.py)
      pdf_vector       labels read verbatim from a vector flowchart's own text
                       layer, with the topology derived geometrically from the
                       shapes (workflow_vector.py). No model involved.
      vlm_description  a vision model's reading of a genuinely scanned page -
                       now only the two image-only signature matrices, which
                       have no text layer to read (workflow_extractor.py).

    All workflow chunks stay `review: "machine"` until a person checks them:
    the pdf_vector tokens are verbatim, but which arrow points where is still
    inferred, and `coverage` plus `audit_warnings` on each chunk say how much
    of the page the extractor could account for.

    They share an index because a user does not know which kind of document
    answers their question, and "who approves procurement over 500 K" is
    answered by the signature matrix while "what form is the RFQ" is answered
    by the process text.
    """
    chunks = json.load(open(CHUNKS_JSON, encoding="utf-8"))
    if include_workflows and WORKFLOW_CHUNKS_JSON.exists():
        chunks += json.load(open(WORKFLOW_CHUNKS_JSON, encoding="utf-8"))
    return chunks


def ground_truth_chunks(chunks: list[dict]) -> list[dict]:
    """Only chunks whose TOKENS came off the page rather than out of a model.

    The form-number whitelist MUST be built from these alone. It is what makes
    a hallucinated citation detectable, and it works by asserting that a cited
    form appears somewhere in the corpus. Let model-generated text into the
    corpus and a form number the vision model invented would whitelist itself -
    the validator would then confirm it as genuine, which is worse than having
    no validator at all.

    The test is on the tokens, not on the sentences. `pdf_vector` chunks
    qualify: every label in them is read from the flowchart's own text layer,
    so a form number in a workflow cannot have been invented, even though the
    prose around it is rendered from the graph rather than quoted. Only
    `vlm_description` - a model looking at a picture - is excluded, and since
    workflow_vector.py took over the vector sheets that is just the two
    image-only signature matrices.
    """
    return [c for c in chunks if c.get("source_type") != "vlm_description"]


OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:14b"

THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)

# Arabic-Indic and Extended Arabic-Indic digits. A form number rendered in
# these is invisible to validator.FORM_RE, so it reads as "cited nothing"
# rather than as a citation - a silent failure, hence the explicit check.
AR_DIGITS_RE = re.compile(r"[٠-٩۰-۹]")

REFUSAL_EN = "Not specified in these process documents."


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> traces before validation."""
    out = THINK_RE.sub("", text or "")
    if "<think>" in out.lower():          # unterminated trace: drop the dangling tail
        out = re.split(r"<think>", out, flags=re.I)[0]
    return out.strip()


def format_context(retrieved: list[dict]) -> str:
    """Chunks as the model sees them. Shared, so a continuation prompt shows the
    same text in the same shape as the answer it is continuing."""
    return "\n\n".join(
        f"[{c['filename']} | {c['section']}]\n{c['text']}" for c in retrieved
    )


def build_prompt(question: str, retrieved: list[dict], answer_in_arabic: bool = False) -> str:
    """Context-only prompt. The Arabic directive is the only thing that varies."""
    context = format_context(retrieved)
    refusal = REFUSAL_AR if answer_in_arabic else REFUSAL_EN
    instruction = (
        "Answer using ONLY the context below. Dont just state whats in the context, explain it before stating to refer to the document. If the answer isn't in the context, "
        f"say '{refusal}' Always cite the doc filename."
    )
    if answer_in_arabic:
        instruction += ANSWER_IN_ARABIC
    return f"{instruction}\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\nANSWER:"


def build_continuation_prompt(retrieved: list[dict], prev_answer: str,
                              answer_in_arabic: bool = False) -> str:
    """Prompt for "you missed some" - same chunks, second pass.

    No QUESTION line, because there is no new question: the instruction is to
    finish the previous answer out of the context it was already built from.
    """
    prompt = CONTINUATION_PROMPT.format(chunks=format_context(retrieved),
                                        prev_answer=prev_answer)
    if answer_in_arabic:
        prompt += ANSWER_IN_ARABIC
    return prompt + "\n\nANSWER:"


# ------------------------------------------------------------------- ollama
def ollama_up(host: str = OLLAMA_HOST) -> bool:
    try:
        return requests.get(f"{host}/api/tags", timeout=3).status_code == 200
    except Exception:
        return False


def ollama_models(host: str = OLLAMA_HOST) -> list[str]:
    r = requests.get(f"{host}/api/tags", timeout=10)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]


def ollama_placement(host: str = OLLAMA_HOST) -> str:
    """Report whether a resident model is on GPU or CPU.

    Every latency number is uninterpretable without this, and it cannot be
    read from `torch`: the torch in this venv is a CPU-only build, while
    Ollama ships its own CUDA runtime and will happily use a GPU that torch
    reports as absent. /api/ps is the only honest source, and it lists
    nothing until a model has actually been loaded.
    """
    try:
        models = requests.get(f"{host}/api/ps", timeout=5).json().get("models", [])
    except Exception as e:
        return f"placement unknown ({type(e).__name__})"
    if not models:
        return "no model resident yet"
    out = []
    for m in models:
        total, vram = m.get("size", 0), m.get("size_vram", 0)
        if vram >= total * 0.99:
            where = "fully on GPU"
        elif vram == 0:
            where = "CPU only"
        else:
            where = f"partly on GPU ({vram / total:.0%} of weights)"
        out.append(f"{m['name']}: {where}, {total / 1e9:.1f} GB")
    return " | ".join(out)


def generate(prompt: str, model: str = DEFAULT_MODEL, host: str = OLLAMA_HOST,
             timeout: int = 1800) -> str:
    """Deterministic generation. temperature=0 and a fixed seed, so a repeated
    question gives a repeated answer.

    `num_predict: -1` because a capped budget truncates long explanatory
    answers mid-sentence with nothing in the output saying so. `num_ctx` is
    pinned at 8192 so Ollama's 4096 default cannot quietly reintroduce a
    ceiling that shrinks as top_k grows.
    """
    payload = {
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "num_predict": -1, "num_ctx": 8192, "seed": 0},
        "think": False,
    }
    resp = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
    if resp.status_code == 400:           # older Ollama, or the model rejects `think`
        payload.pop("think")
        resp = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["response"].strip()


# ------------------------------------------------------------------ chatbot
class Chatbot:
    """Loaded pipeline. Build once, ask many times.

    Construction is the expensive part - it loads MiniLM and builds the FAISS
    index - so callers should hold onto the instance (`@st.cache_resource` in
    the app, a module global in the notebook).
    """

    def __init__(self, chunks: list[dict] | None = None, model: str = DEFAULT_MODEL,
                 host: str = OLLAMA_HOST, policy: str = "flag",
                 retriever: Retriever | None = None,
                 contextualizer: "Contextualizer | None" = None):
        if chunks is None:
            chunks = load_corpus()
        self.chunks = chunks
        self.model = model
        self.host = host
        self.policy = policy
        # Reuse a retriever if the caller already built one (the notebook has).
        self.retriever = retriever or Retriever(chunks)
        # Whitelists from verbatim source text only - never from VLM output.
        truth = ground_truth_chunks(chunks)
        self.form_whitelist = V.build_form_whitelist(truth)
        self.doc_whitelist = V.build_doc_code_whitelist(truth)
        self.n_workflow_chunks = len(chunks) - len(truth)
        # Its own model, its own store. Passing no session_id to ask() makes it
        # a no-op, so existing callers - the notebook, eval_arabic.py - keep
        # their exact current behaviour.
        self.contextualizer = contextualizer or Contextualizer(host=host)

    def _gen(self, prompt: str) -> str:
        """Generator handed to translate.py, so translations get the same
        determinism and the same trace stripping as answers."""
        return strip_reasoning(generate(prompt, model=self.model, host=self.host))

    def ask(self, question: str, top_k: int = 3, session_id: str | None = None) -> dict:
        """Answer one question. Returns a record; prints nothing.

        Arabic in, Arabic out, with a single translation hop on the way in.
        The answer is generated in formal Arabic directly from the English
        context rather than written in English and translated back - one
        fewer chance to corrupt a form number, and one fewer model call.
        """
        src_ar = is_arabic(question)

        t0 = time.perf_counter()
        q_en = to_english(question, self._gen) if src_ar else question
        t_in = time.perf_counter() - t0

        # Follow-up resolution. AFTER translation, so the gate, the history and
        # the rewrite model all only ever see English - the same reason
        # retrieval does. Skipped entirely without a session_id.
        t0 = time.perf_counter()
        ctx = self.contextualizer.resolve(q_en, session_id)
        t_ctx = time.perf_counter() - t0

        if ctx.needs_retrieval:
            # Retrieval runs on the resolved query. Previous chunks are not
            # reused - the boost is a ranking signal, nothing is excluded.
            # Spelling repair up front, so retrieval AND generation see the
            # same words. Doing it only inside search() was not enough: the
            # chunks came back right and the generator still read the typo in
            # the question, which was the difference between "160 kilometers"
            # and "Not specified in these process documents" off identical
            # context. Idempotent, so search() repeating it is free.
            #
            # Surfaced in the record for the same reason the Arabic
            # translation is: if it guessed wrong, that is invisible in the
            # answer.
            q_search, repairs = self.retriever.repair_query(ctx.query)
            t0 = time.perf_counter()
            route = route_prefixes(q_search)
            hits = self.retriever.search(q_search, top_k=top_k,
                                         boost_prefixes=ctx.boost_prefixes)
            t_ret = time.perf_counter() - t0
            prompt = build_prompt(q_search, hits, answer_in_arabic=src_ar)
        else:
            # Continuation: the user said the last answer was incomplete. The
            # chunks are already the right ones - retrieving on "list the rest"
            # would search the corpus for those words and replace the very
            # context being asked about. Replay them and re-prompt instead.
            q_search = ctx.original
            route, t_ret, repairs = None, 0.0, []
            hits = ctx.reuse_hits
            prompt = build_continuation_prompt(hits, ctx.prev_answer,
                                               answer_in_arabic=src_ar)

        t0 = time.perf_counter()
        raw = generate(prompt, model=self.model, host=self.host)
        t_gen = time.perf_counter() - t0

        answer = strip_reasoning(raw)
        # `retrieved=hits` turns on the groundedness check: a form that exists
        # but was not in the chunks this answer was built from. The whitelist
        # cannot see that class of error - see validate_answer's docstring.
        v = V.validate_answer(answer, self.form_whitelist, self.doc_whitelist,
                              policy=self.policy, retrieved=hits)

        # A code in Arabic-Indic digits is unlookuppable AND invisible to the
        # validator, so it would otherwise pass as "cited nothing, clean".
        ar_digits = sorted(set(AR_DIGITS_RE.findall(answer)))

        # Only the block-policy refusal still needs translating: it is English
        # text produced by validator.py, not by the model, so no prompt can
        # pre-empt it.
        t0 = time.perf_counter()
        display = v["display_answer"]
        if src_ar and v["verdict"] == "blocked":
            display = to_arabic(display, self._gen)
        t_out = time.perf_counter() - t0

        # Any retrieved chunk that is a machine reading of a scanned diagram
        # rather than source text. Surfaced because the validator cannot vouch
        # for these: its whitelist is built from ground truth only, so a wrong
        # threshold here is invisible to every automatic check.
        vlm_sources = [
            {"filename": h["filename"], "section": h["section"], "page": h.get("page"),
             "review": h.get("review"), "warnings": h.get("audit_warnings") or []}
            for h in hits if h.get("source_type") == "vlm_description"
        ]

        rec = {
            "question": question, "question_en": q_en,
            "query_used": q_search,
            "was_rewritten": ctx.was_rewritten,
            "mode": ctx.mode,
            "reused_sources": ctx.is_continuation,
            "spelling_repairs": repairs,
            "gate_reasons": ctx.gate.reasons,
            "rewrite_model": ctx.rewrite_model,
            "boost_prefixes": ctx.boost_prefixes,
            "contextualize_s": t_ctx,
            "lang": "ar" if src_ar else "en", "model": self.model,
            "vlm_sources": vlm_sources,
            "answer": answer, "display": display, "raw": raw,
            "hits": hits, "route": route, "validator": v,
            "had_reasoning": raw != answer, "arabic_digits": ar_digits,
            "translate_in_s": t_in, "retrieval_s": t_ret,
            "generation_s": t_gen, "translate_out_s": t_out,
            "total_s": t_ctx + t_in + t_ret + t_gen + t_out,
        }
        # Record the ORIGINAL question, so history reads as the user spoke it.
        # The hits go in too, so the next turn can be a continuation of this
        # one - including a continuation of a continuation, which stores the
        # same replayed chunks again rather than losing them.
        #
        # A continuation records what has been said ACROSS both turns, not just
        # the items it added. "Do not repeat items already listed" is only true
        # if the prompt can see everything already listed; storing the delta
        # alone would let a third turn re-offer what the first one answered.
        recorded = f"{ctx.prev_answer}\n\n{answer}".strip() if ctx.is_continuation else answer
        self.contextualizer.record(session_id, question, recorded, hits=hits)
        return rec


if __name__ == "__main__":
    import sys

    from translate import enable_utf8_stdout

    enable_utf8_stdout()
    q = " ".join(sys.argv[1:]) or "ايه هي الاستمارة اللي بتستخدم في استبيان رضا العملاء؟"
    bot = Chatbot()
    rec = bot.ask(q)
    print(f"Q ({rec['lang']}): {rec['question']}")
    if rec["lang"] == "ar":
        print(f"EN     : {rec['question_en']}")
    print(f"\n{rec['display']}\n")
    for h in rec["hits"]:
        print(f"  [{h['score']:.3f}] {h['filename']} | {h['section']}")
    print(f"\ncited: {rec['validator']['cited'] or 'none'}  "
          f"verdict: {rec['validator']['verdict']}  {rec['total_s']:.1f}s")

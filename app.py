"""
Streamlit front end for the RME process assistant.

    streamlit run app.py

Ask in English or Arabic; the language is detected, not selected. This is a
thin rendering layer - the pipeline lives in backend/chatbot.py, which the
notebook imports too, so there is exactly one copy of the prompt text and the
validation order.

WHAT CHANGED, AND WHY THE DIAGNOSTICS DID NOT GO AWAY
-----------------------------------------------------
The previous version of this file put retrieval scores, latency breakdowns,
route decisions, spelling repairs and raw chunk text on screen for every
answer. That was the right call while the thing being debugged was the
retriever. It is the wrong default for ~1000 colleagues asking about the
accommodation policy, because it presents an internal confidence number as if
it were part of the answer, and a person who cannot interpret `0.412` will
either ignore it or over-trust it. Neither is useful.

So everything is still here, behind `Developer view` in the sidebar. Nothing
was deleted - a wrong answer is still diagnosable in one click, which is the
property the old UI existed to protect.

What stays visible to everyone, because it is not debugging output:

  * The documents an answer came from. An unsourced answer about a compliance
    process is worse than no answer; the user has to be able to open the PDF.
  * Any warning that the answer may be unreliable - an unverifiable form
    number, or a fact read off a machine-interpreted diagram. These are
    phrased as what to do about it rather than as validator verdicts.

The one thing deliberately NOT surfaced in the normal view is the retrieval
score. It is a within-query ranking signal, not a probability that the answer
is correct, and showing it invites exactly that misreading.
"""
from __future__ import annotations

import re
import sys
import uuid
from pathlib import Path

# backend/ holds the pipeline; its modules import each other by bare name.
# See paths.add_backend_to_path for why that is worth keeping.
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

import streamlit as st

import config
import logs
from chatbot import (DEFAULT_MODEL, Chatbot, load_corpus, ollama_models,
                     ollama_placement, ollama_up)
from contextualize import REWRITE_MODEL, Contextualizer
from errors import PipelineError
from retriever import WORKFLOW_SOURCE_TYPES, Retriever

# Streamlit re-runs this whole module on every interaction; logs.setup() is
# idempotent, so this attaches handlers once per process rather than once per
# keystroke.
log = logs.setup()

st.set_page_config(page_title="RME Process Assistant", page_icon="📘",
                   layout="centered", initial_sidebar_state="collapsed")

# Arabic needs right-to-left or it is genuinely hard to read. Scoped to the
# answer body only - source names stay LTR because they are filenames and
# form codes.
st.markdown("""
<style>
  .rtl { direction: rtl; text-align: right; font-size: 1.05rem; line-height: 1.9; }
  .ltr { direction: ltr; text-align: left; }
  .src-card {
      border: 1px solid rgba(128,128,128,.25); border-radius: 8px;
      padding: .55rem .7rem; margin-bottom: .4rem; font-size: .88rem;
      line-height: 1.35;
  }
  .src-doc  { font-weight: 600; }
  .src-sec  { opacity: .72; }
  .src-tag  {
      display: inline-block; font-size: .72rem; padding: .05rem .4rem;
      border-radius: 4px; border: 1px solid rgba(128,128,128,.35);
      opacity: .8; margin-left: .35rem; vertical-align: middle;
  }
  .example-hint { opacity: .65; font-size: .9rem; margin-bottom: .3rem; }
  div[data-testid="stChatMessage"] { padding-top: .35rem; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------- presentation
def pretty_doc(filename: str) -> str:
    """'PCN01_Subcontract_Agreement_Process.pdf' -> 'P-CN-01 Subcontract Agreement Process'.

    Chunk filenames are the sanitised ones the extractor wrote, not the ones on
    the shared drive, so they read as identifiers rather than as documents. The
    doc code is restored to its hyphenated form because that is how these are
    referred to in conversation and in the documents themselves - someone
    looking for "P-CN-01" will not recognise "PCN01".
    """
    name = re.sub(r"\.pdf$", "", filename, flags=re.I)
    name = name.replace("_", " ")
    # PCN01 -> P-CN-01, MHR08 -> M-HR-08. Only at the start, only when it looks
    # like a doc code, so workflow filenames ("Bonds Request - Bonds Request
    # WF") are left exactly as they are.
    name = re.sub(r"^([PM])([A-Z]{2,3})(\d{2})\b", r"\1-\2-\3", name)
    return name.strip()


def source_cards(hits: list[dict], dev: bool) -> None:
    """One card per source document, deduplicated.

    Deduplicated because three chunks of the same policy are one source to a
    reader and three rows to the retriever, and a list that says the same
    filename three times looks like a bug. The developer view keeps them split,
    since which chunk matched is the whole question there.
    """
    seen: dict[str, dict] = {}
    for h in hits:
        key = h["filename"] if not dev else f"{h['filename']}|{h.get('section')}|{h.get('page')}"
        if key not in seen:
            seen[key] = h

    for h in seen.values():
        wf = h.get("source_type") in WORKFLOW_SOURCE_TYPES
        vlm = h.get("source_type") == "vlm_description"
        tags = ""
        if vlm:
            tags += '<span class="src-tag">machine-read diagram</span>'
        elif wf:
            tags += '<span class="src-tag">workflow diagram</span>'
        if dev:
            tags += f'<span class="src-tag">{h["score"]:.3f}</span>'
        page = f" · p{h['page']}" if h.get("page") else ""
        st.markdown(
            f'<div class="src-card"><span class="src-doc">{pretty_doc(h["filename"])}</span>'
            f'{tags}<br><span class="src-sec">{h.get("section") or "—"}{page}</span></div>',
            unsafe_allow_html=True)
        if dev:
            for w in h.get("audit_warnings") or []:
                st.caption(f"extraction warning: {w}")
            with st.expander("chunk text"):
                st.text(h["text"])


def reliability_notes(rec: dict) -> None:
    """User-facing caveats. Phrased as what to do, not as a validator verdict.

    Each of these used to render as `validator flagged` or similar. That is the
    correct word internally and the wrong one on screen: it tells a user that
    something is wrong without telling them whether to act on the answer. The
    rewrite keeps the same trigger conditions and changes only the sentence.
    """
    v = rec["validator"]
    if v["verdict"] == "blocked":
        st.error("This answer was withheld because it referenced a form number "
                 "that does not appear in any indexed document. Please check with "
                 "the document owner.")
    elif v["verdict"] == "flagged":
        st.warning(f"Could not verify form number(s): {', '.join(v['unknown'])}. "
                   "They may belong to a document that is not indexed yet — "
                   "confirm before using them.")
    elif v["verdict"] == "ungrounded":
        st.info(f"{', '.join(v['ungrounded'])} is a real form, but it does not "
                "appear in the sources below. Check it against the document "
                "before relying on it.")

    if rec.get("vlm_sources"):
        unreviewed = [s for s in rec["vlm_sources"] if s.get("review") != "human"]
        msg = ("Part of this answer was read from a scanned diagram by a model, "
               "so it could not be checked automatically.")
        if unreviewed:
            msg += " It has not been verified by a person yet."
        msg += " Confirm anything critical against the original document."
        st.info(msg)

    if rec.get("variant_conflicts"):
        # The reader is the only one who knows which version they meant. The
        # answer names the one it used; this names the ones it did not, because
        # "there are four of these and you got the Damietta one" is not
        # recoverable from a source list that shows four similar filenames.
        for con in rec["variant_conflicts"]:
            others = ", ".join(f"**{o['variant']}**" for o in con["others"])
            st.warning(
                f"**{con['family']}** exists in more than one version. This answer "
                f"is built from **{con['chosen']['variant']}**. Also retrieved: "
                f"{others}. If you meant one of those, ask for it by name — the "
                "versions differ in their steps and approvers.")

    if rec.get("incomplete_workflows"):
        # The chunk knows it is incomplete and the answer cannot tell you, so
        # this says it instead. Named per document, because "some diagram was
        # incomplete" is not actionable when six sources are listed below.
        detail = "\n".join(
            f"- **{pretty_doc(w['filename'])}**"
            + (f" · {w['section']}" if w.get("section") else "")
            + ": " + "; ".join(w["gaps"])
            for w in rec["incomplete_workflows"])
        st.warning(
            "Part of a workflow diagram behind this answer could not be read "
            "in full, so steps or routes may be missing:\n\n"
            + detail
            + "\n\nCheck the original diagram before relying on the sequence.")

    if rec["arabic_digits"]:
        st.warning("This answer contains Arabic-Indic numerals. Any form number "
                   "written in them could not be checked — verify it manually.")


def render(rec: dict, dev: bool) -> None:
    """One answer."""
    css = "rtl" if rec["lang"] == "ar" else "ltr"
    st.markdown(f'<div class="{css}">{rec["display"]}</div>', unsafe_allow_html=True)

    reliability_notes(rec)

    n = len({h["filename"] for h in rec["hits"]})
    label = ("Sources from the previous answer" if rec.get("reused_sources")
             else f"Sources ({n} document{'s' if n != 1 else ''})")
    with st.expander(label):
        source_cards(rec["hits"], dev)

    if dev:
        with st.expander("Developer detail", expanded=False):
            lat = ("retrieval skipped" if rec.get("reused_sources")
                   else f"retrieval {rec['retrieval_s'] * 1000:.0f} ms")
            lat += f" · generation {rec['generation_s']:.1f} s"
            if rec["lang"] == "ar":
                lat = f"translate {rec['translate_in_s']:.1f} s · " + lat
            st.caption(f"{lat} · **{rec['total_s']:.1f} s** total · {rec['model']}"
                       + ("  ·  reasoning trace stripped" if rec["had_reasoning"] else ""))
            st.caption(f"validator: {rec['validator']['verdict']} · "
                       f"cited {', '.join(rec['validator']['cited']) or 'nothing'}")
            if rec["lang"] == "ar":
                st.markdown("**Translated for retrieval:**")
                st.code(rec["question_en"], language=None)
            if rec.get("reused_sources"):
                st.info("Read as “that answer was incomplete”, so nothing was "
                        "retrieved. The sources above are the previous turn's.")
                st.caption(f"gate: {', '.join(rec['gate_reasons'])}")
            elif rec.get("was_rewritten"):
                st.markdown("**Resolved from your follow-up:**")
                st.code(rec["query_used"], language=None)
                st.caption(f"gate: {', '.join(rec['gate_reasons'])} · "
                           f"rewritten by {rec.get('rewrite_model')}")
            if rec.get("spelling_repairs"):
                fixed = ", ".join(f"*{a}* → **{b}**" for a, b in rec["spelling_repairs"])
                st.caption(f"spelling repaired against corpus vocabulary: {fixed}")
            if not rec.get("reused_sources"):
                st.caption(f"route: {rec['route'] or 'none — searched everything'}"
                           + (f" · family boost: {', '.join(rec['boost_prefixes'])}"
                              if rec.get("boost_prefixes") else ""))
                # The workflow floor is invisible in the results themselves: a
                # diagram in the list looks the same whether it won on score or
                # was backfilled. Saying which happened is the only way to tell
                # a genuine match from a reserved slot.
                st.caption(
                    f"workflow intent: {rec.get('workflow_intent')} · "
                    f"{rec.get('n_workflow_hits', 0)} of {len(rec['hits'])} "
                    "sources are diagrams")


# ------------------------------------------------------------------- pipeline
@st.cache_resource(show_spinner="Loading the document index…")
def load_index():
    """Chunks and retriever, built once per session.

    A failure here is fatal for the whole app rather than for one answer -
    there is nothing to search - so it stops the script with the sentence that
    says what to do, instead of letting a traceback render where the answers
    go.
    """
    try:
        chunks = load_corpus()  # process text + workflow diagrams, one index
        return chunks, Retriever(chunks)
    except PipelineError as e:
        log.error("index unavailable: %s", e)
        st.error(e.user_message)
        st.stop()
    except Exception:
        log.exception("index failed to load")
        st.error("The document index could not be loaded. The details are in "
                 "the application log.")
        st.stop()


@st.cache_resource(show_spinner=False)
def get_contextualizer(rewrite_model: str) -> Contextualizer:
    """Cached, because it owns the session history.

    Streamlit re-runs this whole script on every interaction, so a
    Contextualizer built inside get_bot() would be discarded along with its
    SessionStore after each turn - follow-up resolution would never see a
    previous turn and the feature would silently do nothing.
    """
    return Contextualizer(rewrite_model=rewrite_model)


def get_bot(model: str, policy: str, rewrite_model: str, follow_ups: bool) -> Chatbot:
    chunks, retriever = load_index()
    ctx = get_contextualizer(rewrite_model)
    ctx.enabled = follow_ups
    return Chatbot(chunks, model=model, policy=policy, retriever=retriever,
                   contextualizer=ctx)


# -------------------------------------------------------------------- sidebar
up = ollama_up()

st.sidebar.title("RME Process Assistant")
if st.sidebar.button("New conversation", use_container_width=True):
    get_contextualizer(REWRITE_MODEL).store.clear(st.session_state.get("sid", ""))
    st.session_state.history = []
    st.rerun()

st.sidebar.divider()
dev = st.sidebar.toggle(
    "Developer view", value=False,
    help="Retrieval scores, latency, routing decisions and the exact text each "
         "answer was built from.")

# Defaults for everyone. Only the developer view can change them, because these
# are the settings that were tuned against the eval sets - a user who lowers
# top_k to 1 to make answers faster is silently trading away recall.
model, policy, follow_ups, top_k = DEFAULT_MODEL, "flag", True, 6

if dev:
    st.sidebar.caption("These override tuned defaults. See evals/.")
    if up:
        try:
            available = ollama_models()
        except Exception:
            available = [DEFAULT_MODEL]
        idx = available.index(DEFAULT_MODEL) if DEFAULT_MODEL in available else 0
        model = st.sidebar.selectbox("Model", available, index=idx)
    top_k = st.sidebar.slider(
        "Chunks retrieved", 1, 12, 6,
        help="Ceiling, not a quota: weak matches are trimmed, so a narrow "
             "question returns fewer than this.")
    strict = st.sidebar.checkbox(
        "Block unverifiable citations", value=False,
        help="Withhold any answer citing a form number absent from the corpus. Off "
             "by default: the whitelist is much better at 71 documents but still "
             "incomplete, so 'unknown' means unverifiable rather than fabricated.")
    policy = "block" if strict else "flag"
    follow_ups = st.sidebar.checkbox("Resolve follow-up questions", value=True)
    st.sidebar.divider()
    st.sidebar.caption(f"hardware: {ollama_placement()}" if up else "ollama: down")
    st.sidebar.caption(f"rewrite model: {REWRITE_MODEL}")

# ----------------------------------------------------------------------- main
st.title("RME Process Assistant")
st.caption("Answers about RME's processes, policies and approval workflows — "
           "drawn only from the company's own documents. "
           "اسأل بالعربي أو بالإنجليزي.")

if not up:
    st.error("The assistant is not available right now — the local model server "
             "is not responding. If you are running this yourself, start it with "
             "`ollama serve` and reload.")
    st.stop()

st.session_state.setdefault("history", [])
# One id per browser session. Kept in session_state so it survives Streamlit's
# re-run-on-every-interaction, and so two open tabs get separate histories.
st.session_state.setdefault("sid", uuid.uuid4().hex)
st.session_state.setdefault("pending", None)

# Empty state. Examples are not decoration: the corpus spans HR policy, finance
# process and approval diagrams, and a user who does not know that asks one
# narrow question, gets one answer, and never discovers the rest. One example
# per kind, including a workflow one, because route questions are the case the
# retriever was just changed for and the least obvious thing to think of asking.
EXAMPLES = [
    "How many days of annual leave do employees get?",
    "Who approves a purchase order above 10M EGP?",
    "What is the approval workflow for a subcontract amendment?",
    "Which form is used for the customer satisfaction survey?",
]

if not st.session_state.history:
    chunks, _ = load_index()
    n_docs = len({c["filename"] for c in chunks})
    n_wf = len({c["filename"] for c in chunks
                if c.get("source_type") in WORKFLOW_SOURCE_TYPES})
    st.caption(f"Indexed: {n_docs - n_wf} process documents and {n_wf} workflow diagrams.")
    st.markdown('<div class="example-hint">Try one of these:</div>',
                unsafe_allow_html=True)
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

for rec in st.session_state.history:
    with st.chat_message("user"):
        css = "rtl" if rec["lang"] == "ar" else "ltr"
        st.markdown(f'<div class="{css}">{rec["question"]}</div>', unsafe_allow_html=True)
    with st.chat_message("assistant"):
        render(rec, dev)

q = st.chat_input("Ask about a process, policy or approval workflow…")
if st.session_state.pending:
    q, st.session_state.pending = st.session_state.pending, None

if q:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        # Long explanatory answers legitimately run for minutes on CPU, so the
        # spinner has to stay up rather than imply the app has hung.
        with st.spinner("Searching the documents…"):
            try:
                rec = get_bot(model, policy, REWRITE_MODEL, follow_ups).ask(
                    q, top_k=top_k, session_id=st.session_state.sid)
            except PipelineError as e:
                # An expected failure with a known cause: say which one. "The
                # model is not pulled" and "the server is down" send the reader
                # to different places, and one generic sentence sent them to
                # neither.
                log.warning("%s: %s", type(e).__name__, e)
                st.error(e.user_message)
                if dev:
                    st.exception(e)
                st.stop()
            except Exception as e:
                # Not one of ours: a bug. The traceback goes to the log where
                # it can be found later, not only to whoever had the tab open.
                log.exception("unhandled error answering a question")
                st.error("Something went wrong answering that. Please try again.")
                if dev:
                    st.exception(e)
                st.stop()
        render(rec, dev)
    st.session_state.history.append(rec)
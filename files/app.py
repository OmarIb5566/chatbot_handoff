"""
Streamlit front end for the RME process chatbot.

    streamlit run app.py

Ask in English or Arabic; the language is detected, not selected. This is a
thin rendering layer - the pipeline lives in chatbot.py, which the notebook
imports too, so there is exactly one copy of the prompt text and the
validation order.

What the UI is actually for, beyond convenience: an answer on its own cannot
tell you whether a wrong result was a retrieval failure or a generation
failure. So every answer ships with the chunks it was built from and the
validator's verdict, in an expander rather than buried in a log. A refusal in
particular looks like the system being careful; the only way to know is to
read the sources under it.
"""
from __future__ import annotations

import uuid

import streamlit as st

from chatbot import (DEFAULT_MODEL, Chatbot, load_corpus, ollama_models,
                     ollama_placement, ollama_up)
from contextualize import REWRITE_MODEL, Contextualizer
from retriever import Retriever

st.set_page_config(page_title="RME Process Chatbot", page_icon="📄", layout="centered")

# Arabic needs right-to-left or it is genuinely hard to read. Scoped to the
# answer body only - the metadata below it stays LTR because it is filenames,
# form codes and numbers.
st.markdown("""
<style>
  .rtl { direction: rtl; text-align: right; font-size: 1.05rem; line-height: 1.9; }
  .ltr { direction: ltr; text-align: left; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading index (MiniLM + FAISS)…")
def load_index():
    """Chunks and retriever, built once per session.

    This is the expensive part - it loads MiniLM and builds the FAISS index -
    so it is cached separately from the Chatbot, which is cheap to rebuild
    when the model or policy changes in the sidebar.
    """
    chunks = load_corpus()      # process text + scanned workflow diagrams
    return chunks, Retriever(chunks)


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


def render(rec: dict, show_context: bool) -> None:
    """One answer, with everything needed to judge whether to trust it."""
    v = rec["validator"]
    css = "rtl" if rec["lang"] == "ar" else "ltr"
    st.markdown(f'<div class="{css}">{rec["display"]}</div>', unsafe_allow_html=True)

    if v["verdict"] == "flagged":
        st.warning(f"Unverifiable form number(s): {', '.join(v['unknown'])}. "
                   "Not necessarily wrong — the whitelist covers the 71 indexed process "
                   "documents, so this may be a real form from one not here yet.")
    elif v["verdict"] == "blocked":
        st.error(f"Answer withheld — cited {', '.join(v['unknown'])}, not in the corpus.")
    elif v["verdict"] == "ungrounded":
        # Weaker than "flagged": the form is real. What is wrong is that it was
        # not in the chunks below, and the prompt tells the model to answer from
        # those only. Shown as info rather than a warning because the likeliest
        # cause is a retrieval window that was too narrow, not a fabrication.
        st.info(f"Cited {', '.join(v['ungrounded'])}, which exists in the corpus but "
                "is not in the sources below. Check it against the document before "
                "relying on it.")

    # Provenance. The validator's whitelist is built from verbatim source text
    # only, so it cannot vouch for anything read off a scanned diagram - a
    # wrong approval threshold here passes every automatic check. Say so where
    # the answer is, not in a log.
    if rec.get("vlm_sources"):
        srcs = ", ".join(f"{s['section']} (p{s['page']})" for s in rec["vlm_sources"])
        unreviewed = [s for s in rec["vlm_sources"] if s.get("review") != "human"]
        flagged = [s for s in rec["vlm_sources"] if s.get("warnings")]
        msg = (f"Built partly from a **machine reading of a scanned diagram** — {srcs}. "
               "The validator cannot check these: it only knows form numbers that "
               "appear in verbatim document text.")
        if unreviewed:
            msg += " Not yet verified by a person."
        if flagged:
            msg += (f" {len(flagged)} of these carry extraction warnings — "
                    "check the sources below against the PDF.")
        st.info(msg)

    if rec["arabic_digits"]:
        st.warning(
            f"Answer contains Arabic-Indic digits ({''.join(rec['arabic_digits'])}). "
            "A form number written in those is unlookuppable and invisible to the "
            "validator, so treat any citation here as unchecked.")

    lat = f"retrieval {rec['retrieval_s']*1000:.0f} ms · generation {rec['generation_s']:.1f} s"
    if rec["lang"] == "ar":
        lat = f"translate {rec['translate_in_s']:.1f} s · " + lat
    st.caption(f"{lat} · **{rec['total_s']:.1f} s** total · {rec['model']}"
               + ("  ·  reasoning trace stripped" if rec["had_reasoning"] else ""))

    with st.expander(f"Sources ({len(rec['hits'])}) · "
                     f"cited {', '.join(v['cited']) if v['cited'] else 'nothing'} · "
                     f"validator {v['verdict']}"):
        if rec["lang"] == "ar":
            # The single most useful debugging line for an Arabic question: a
            # mistranslated domain term is a retrieval miss with no other signal.
            st.markdown("**Translated for retrieval:**")
            st.code(rec["question_en"], language=None)
        if rec.get("was_rewritten"):
            # Same reason the Arabic translation is shown: if the answer looks
            # wrong, the rewrite is the first thing to check.
            st.markdown("**Resolved from your follow-up:**")
            st.code(rec["query_used"], language=None)
            st.caption(f"gate: {', '.join(rec['gate_reasons'])} · "
                       f"rewritten by {rec.get('rewrite_model')}")
        st.caption(f"route: {rec['route'] or 'none — searched everything'}"
                   + (f" · boost: {', '.join(rec['boost_prefixes'])}"
                      if rec.get("boost_prefixes") else ""))
        for h in rec["hits"]:
            vlm = h.get("source_type") == "vlm_description"
            badge = f"  ·  ⚠ machine-read diagram, p{h.get('page')}" if vlm else ""
            st.markdown(f"**`{h['score']:.3f}`  {h['filename']}** — {h['section']}{badge}")
            for w in h.get("audit_warnings") or []:
                st.caption(f"extraction warning: {w}")
            if show_context:
                st.text(h["text"])
        if not show_context:
            st.caption("Turn on *Show retrieved text* in the sidebar to see the "
                       "exact chunks the model was given.")


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Settings")

up = ollama_up()
if up:
    try:
        available = ollama_models()
    except Exception:
        available = [DEFAULT_MODEL]
    idx = available.index(DEFAULT_MODEL) if DEFAULT_MODEL in available else 0
    model = st.sidebar.selectbox("Model", available, index=idx)
else:
    model = DEFAULT_MODEL

top_k = st.sidebar.slider("Chunks retrieved", 1, 8, 3,
                          help="How many chunks go into the prompt as context.")
show_context = st.sidebar.checkbox("Show retrieved text", value=False)
strict = st.sidebar.checkbox(
    "Block unverifiable citations", value=False,
    help="Withhold any answer citing a form number absent from the corpus. Off by "
         "default: at 71 documents the whitelist (366 forms) is much better but still "
         "incomplete, so 'unknown' means unverifiable rather than fabricated. It also "
         "would not have caught the one citation error in the 100-question run, which "
         "cited a real form that was simply the wrong one.")
policy = "block" if strict else "flag"

follow_ups = st.sidebar.checkbox(
    "Resolve follow-up questions", value=True,
    help="Rewrite fragments like 'in the self execution process' into standalone "
         "queries using the last few turns. A cheap heuristic decides first, so a "
         "self-contained question costs no extra model call.")

st.sidebar.divider()
st.sidebar.caption(f"hardware: {ollama_placement()}" if up else "ollama: down")
st.sidebar.caption(f"rewrite model: {REWRITE_MODEL}")
if st.sidebar.button("Clear conversation"):
    # Clear the rolling history too, or the next question would be resolved
    # against turns the user can no longer see.
    get_contextualizer(REWRITE_MODEL).store.clear(st.session_state.get("sid", ""))
    st.session_state.history = []
    st.rerun()

# --------------------------------------------------------------------- main
st.title("RME Process Chatbot")
st.caption("Ask about RME's process documents and approval workflows, in English or "
           "Arabic — اسأل بالعربي أو بالإنجليزي. Answers come only from the documents, "
           "with sources cited.")

if not up:
    st.error("Ollama is not reachable at `localhost:11434`. Start it with "
             "`ollama serve`, then reload this page.")
    st.stop()

st.session_state.setdefault("history", [])
# One id per browser session. Kept in session_state so it survives Streamlit's
# re-run-on-every-interaction, and so two open tabs get separate histories.
st.session_state.setdefault("sid", uuid.uuid4().hex)

for rec in st.session_state.history:
    with st.chat_message("user"):
        css = "rtl" if rec["lang"] == "ar" else "ltr"
        st.markdown(f'<div class="{css}">{rec["question"]}</div>', unsafe_allow_html=True)
    with st.chat_message("assistant"):
        render(rec, show_context)

if q := st.chat_input("Ask a question…"):
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        # Long explanatory answers legitimately run for minutes on CPU, so the
        # spinner has to stay up rather than imply the app has hung.
        with st.spinner("Retrieving and generating…"):
            try:
                rec = get_bot(model, policy, REWRITE_MODEL, follow_ups).ask(
                    q, top_k=top_k, session_id=st.session_state.sid)
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
                st.stop()
        render(rec, show_context)
    st.session_state.history.append(rec)

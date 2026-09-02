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

import config
import logs
import validator as V
from errors import (CorpusMissing, GenerationFailed, ModelNotFound,
                    ModelTimeout, ModelUnavailable)
from contextualize import CONTINUATION_PROMPT, Contextualizer
from retriever import (Retriever, route_prefixes, workflow_intent,
                       WORKFLOW_SOURCE_TYPES)
from translate import ANSWER_IN_ARABIC, REFUSAL_AR, is_arabic, to_arabic, to_english

from paths import (CHUNKS_JSON, WORKFLOW_CHUNKS_JSON,  # noqa: E402  (see paths.py)
                   WORKFLOW_CHUNKS_ACTIVE)


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
    if not CHUNKS_JSON.exists():
        # Distinguishable from a bug: the code is fine, the corpus has not been
        # built on this checkout. data/ is regenerable and therefore not in the
        # repo, so this is the FIRST thing a new clone hits, and a bare
        # FileNotFoundError halfway up a Streamlit traceback does not say what
        # to run.
        raise CorpusMissing(f"{CHUNKS_JSON} does not exist")
    chunks = json.load(open(CHUNKS_JSON, encoding="utf-8"))
    if include_workflows:
        # WORKFLOW_CHUNKS_ACTIVE is the repaired corpus; fall back to the raw
        # extraction if it has not been built on this checkout.
        wf = (WORKFLOW_CHUNKS_ACTIVE if WORKFLOW_CHUNKS_ACTIVE.exists()
              else WORKFLOW_CHUNKS_JSON)
        if wf.exists():
            chunks += drop_unreadable(json.load(open(wf, encoding="utf-8")))
    return chunks


def drop_unreadable(workflow_chunks: list[dict]) -> list[dict]:
    """Drop workflow chunks that carry no actual step names.

    ROUND 2 CORRECTION: the first version of this matched the "carry no
    label" string in `audit_warnings`, which Cowork's verification run showed
    is not scoped to the chunk it's attached to. `workflow_vector.py` writes
    that warning once per SOURCE PAGE and copies it onto every chunk split
    from that page - so "Subcontract Amendments V1(USD in Damietta
    Project)"'s three approval-flow chunks all carry the identical warning
    text, even though checking `graph.nodes` directly shows flow 1 is 0 of 17
    blank, flow 3 is 1 of 21, and only flow 2 (2 of 2) is actually empty. The
    string match dropped all three chunks to remove the one that deserved it,
    which cost eval_set_workflow a real question (WF-14) whose answer lived in
    the un-degenerate flow 3.

    This version checks the chunk's own `graph.nodes` instead: a chunk is
    dropped only if it has a graph AND every node in THAT chunk's own graph is
    blank. `audit_warnings` is no longer used for this decision - it remains a
    real signal, but at the wrong granularity for this purpose.

    Chunks without a `graph` field at all (source_type == "vlm_description":
    prose read off a genuinely scanned page, not a vector graph) are never
    touched here - a different chunk kind with a different quality signal,
    outside what this function is checking.

    Verified against the current corpus: drops exactly 1 of 137 (the one
    empty flow), keeps the two Cowork found were wrongly removed.
    """
    kept = []
    for c in workflow_chunks:
        graph = c.get("graph")
        nodes = graph.get("nodes") if graph else None
        if nodes and not any((n.get("label") or "").strip() for n in nodes):
            continue
        kept.append(c)
    return kept


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


# Kept as module-level names because every caller, the notebook and three
# eval harnesses import them from here. They are now READ from config rather
# than defined here, so there is one source of truth and it answers to the
# environment - see backend/config.py.
OLLAMA_HOST = config.OLLAMA_HOST
DEFAULT_MODEL = config.MODEL

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


# Character budget for the retrieved context, at roughly 4 chars/token against
# the 8192-token num_ctx in generate(). 18000 chars is ~4.5k tokens, leaving
# room for the instruction, the question and a long answer.
#
# This exists because top_k went from 3 to 6 and the two chunk populations are
# not the same size. Process chunks have a median of 209 characters; workflow
# chunks have a median of 1030 and a p90 of 3525, because each one renders a
# whole approval graph. Six workflow chunks is therefore a plausible 20k+
# characters, and Ollama does not error on an over-long prompt - it silently
# drops tokens, which on this prompt shape means dropping the INSTRUCTION at
# the front. The failure mode is an answer that ignores "use only the context"
# with nothing anywhere saying why.
CONTEXT_BUDGET_CHARS = config.CONTEXT_BUDGET_CHARS


def fit_context(retrieved: list[dict], budget: int = CONTEXT_BUDGET_CHARS) -> list[dict]:
    """Drop the lowest-ranked chunks until the context fits the budget.

    Order is preserved and the top hit is always kept, even if it alone busts
    the budget - a single 15k-character chunk is a chunking problem, and
    silently returning nothing would hide it.
    """
    kept, used = [], 0
    for i, c in enumerate(retrieved):
        size = len(c.get("text") or "") + len(c.get("filename") or "") + 16
        if i and used + size > budget:
            break
        kept.append(c)
        used += size
    return kept


UNCLASSIFIED_RE = re.compile(r"^(\d+) of (\d+) text spans unclassified")


def workflow_gaps(chunk: dict) -> list[str]:
    """What a workflow chunk's own metadata says is missing from its graph.

    Every pdf_vector chunk already carries the evidence that it is incomplete -
    `coverage` and `audit_warnings` are written by workflow_vector.py at
    extraction time - and until now nothing read them at answer time. The
    chunk renders exactly the same way whether the extractor accounted for
    every arrow on the page or for two thirds of them, so an answer built from
    a half-recovered diagram is indistinguishable from one built from a
    complete diagram. That is the failure this function exists to make visible:
    not to fix the extraction, but to stop it being silent.

    Two independent signals, because neither one subsumes the other:

      disconnected steps  a node in THIS chunk's own graph that no edge
                          touches. The step name was read off the page but the
                          arrow into it was not recovered, so the rendered text
                          lists it among the steps with nothing connecting it.
                          19 of 137 chunks in the current corpus are like this
                          - `Drawing List (E1 Log)` renders `Auto-Creation` and
                          `End` as steps that lead nowhere - and EVERY ONE of
                          them has coverage 1.0, because coverage counts text
                          spans classified, not arrows recovered. Coverage
                          alone would report these as perfect.

      unplaced text       `coverage < 1.0`: text on the page the extractor
                          could not assign to any node or edge. 33 of 137
                          chunks. Note this one is PAGE-scoped, not chunk-
                          scoped: workflow_vector.py copies the page's warning
                          onto every chunk split from that page, the same
                          granularity mismatch that made drop_unreadable's
                          first version delete good chunks. That is why the
                          wording says "on this page" - claiming the missing
                          spans belong to this particular flow would be a
                          stronger statement than the data supports.

    Returns plain phrases, ready to show a user. Empty list means the chunk's
    own metadata makes no complaint - which is a statement about the
    extraction, not a guarantee about the diagram.
    """
    graph = chunk.get("graph")
    if not graph:
        return []                       # vlm_description: a different signal,
                                        # already surfaced as vlm_sources
    gaps = []

    touched = set()
    for e in graph.get("edges") or []:
        touched.add(e.get("from"))
        touched.add(e.get("to"))
    orphans = [n.get("label") or n.get("id") for n in graph.get("nodes") or []
               if n.get("id") not in touched]
    if orphans:
        named = ", ".join(f"“{o}”" for o in orphans[:4])
        more = f" and {len(orphans) - 4} more" if len(orphans) > 4 else ""
        gaps.append(f"{len(orphans)} step(s) with no route in or out: {named}{more}")

    if (chunk.get("coverage") or 1.0) < 1.0:
        # Prefer the extractor's own count over the ratio: "2 of 16 spans" is
        # actionable, "coverage 0.875" is not.
        detail = next((m.group(0) for w in chunk.get("audit_warnings") or []
                       for m in [UNCLASSIFIED_RE.match(w)] if m), None)
        pct = f"{100 * chunk['coverage']:.0f}%"
        gaps.append(f"{detail} on this page" if detail
                    else f"only {pct} of the page's text could be placed")
    return gaps


def incomplete_workflows(retrieved: list[dict]) -> list[dict]:
    """The gap report for one answer's chunks, in retrieval order.

    Deduplicated per (file, page). The unplaced-text warning is written once
    per SOURCE PAGE and copied onto every chunk split from it, so a page that
    yielded three flows reported "5 of 449 text spans unclassified" three times
    in a row - a caveat that repeats reads as a bug and gets skipped, which
    defeats the point of showing it. Merging by page also lets a chunk's own
    disconnected-step gap sit alongside its page's unplaced-text gap in one
    entry, which is how a reader would think of it: one diagram, one list of
    what could not be read.
    """
    out: dict[tuple, dict] = {}
    for c in retrieved:
        gaps = workflow_gaps(c)
        if not gaps:
            continue
        key = (c["filename"], c.get("page"))
        rec = out.setdefault(key, {"filename": c["filename"],
                                   "section": c.get("section"),
                                   "page": c.get("page"),
                                   "coverage": c.get("coverage"), "gaps": []})
        # Section is only meaningful while one page contributed one flow.
        if rec["section"] != c.get("section"):
            rec["section"] = None
        for g in gaps:
            if g not in rec["gaps"]:
                rec["gaps"].append(g)
    return list(out.values())


def variant_conflicts(retrieved: list[dict]) -> list[dict]:
    """Families with more than one DOCUMENT among the retrieved chunks.

    `Subcontracts` ships five files, `Document Submittal` seven, `Request for
    Quotation` eleven. Ask "what are the subcontract preparation approval
    steps" and the retriever returns V1 (USD in Damietta) at 1.036 and V3 (USD
    in Egypt shell) at 0.960, interleaved, with nothing in the context saying
    they are different documents. The model reads one pile of steps and writes
    one workflow. Nothing downstream can catch it: every step is verbatim, the
    validator sees real form numbers, and the answer cites a real file. It is
    the one failure mode here that produces a confident, fully-grounded, wrong
    process.

    Two DIFFERENT relationships wear the same shape, and this deliberately does
    not try to tell them apart:

      versions      `Subcontract Preparation V1 / V3 / V4` - the same process,
                    superseded. Merging them invents a process that was never
                    approved.
      scopes        `IIR - Manholes / Pipes / Valve Chambers`, `Go or No Go -
                    Competitive Bidding / Direct Award` - different subjects
                    that share a family name. Merging them attributes one
                    scope's steps to another.

    Distinguishing the two needs knowledge the filenames do not carry, and both
    call for the same handling anyway: answer from ONE document and say which.
    So this reports the collision and leaves the choice to the prompt and the
    reader.

    Nothing is dropped. Collapsing to the top-scoring variant is the obvious
    fix and the wrong one here: workflow@1 is 6/18, so the top-ranked variant
    is the right document a third of the time, and a hard collapse would turn a
    visible ambiguity into a silent wrong pick. Retrieval keeps its recall; the
    context gets labelled instead.

    Returned in retrieval order, so `chosen` is the highest-ranked document of
    each family - a default to name, not a decision to enforce.
    """
    order: dict[str, list[str]] = {}
    seen: dict[str, str] = {}
    for c in retrieved:
        if c.get("source_type") not in WORKFLOW_SOURCE_TYPES:
            continue
        fam, fn = c.get("family"), c["filename"]
        if not fam:
            continue
        if fn not in seen:
            seen[fn] = c.get("variant") or fn
            order.setdefault(fam, []).append(fn)
    out = []
    for fam, files in order.items():
        if len(files) > 1:
            out.append({"family": fam,
                        "chosen": {"filename": files[0], "variant": seen[files[0]]},
                        "others": [{"filename": f, "variant": seen[f]}
                                   for f in files[1:]]})
    return out


def variant_labels(retrieved: list[dict]) -> dict[str, str]:
    """filename -> the line that goes above that chunk in the context.

    Only for files whose family has a sibling in this same context. A chunk
    with no sibling present gets no line: telling the model a document has
    versions it cannot see would invite a caveat about something not in front
    of it.
    """
    labels = {}
    for con in variant_conflicts(retrieved):
        n = len(con["others"]) + 1
        for v in [con["chosen"], *con["others"]]:
            labels[v["filename"]] = (
                f'(This is version "{v["variant"]}" of "{con["family"]}". '
                f'{n} different versions of this process appear in the context; '
                f'they are separate documents and their steps are NOT interchangeable.)')
    return labels


def format_context(retrieved: list[dict]) -> str:
    """Chunks as the model sees them. Shared, so a continuation prompt shows the
    same text in the same shape as the answer it is continuing - which is also
    why the variant labels are applied HERE rather than in build_prompt: a
    continuation replays these chunks without re-running retrieval, and a rule
    about not merging versions is worthless if the second pass cannot see which
    chunk is which version.
    """
    labels = variant_labels(retrieved)
    blocks = []
    for c in retrieved:
        head = f"[{c['filename']} | {c['section']}]"
        lab = labels.get(c["filename"])
        if lab:
            head += "\n" + lab
        blocks.append(f"{head}\n{c['text']}")
    return "\n\n".join(blocks)


# Layout rules for answers built from an approval diagram.
#
# Two failures observed in the running app, both on the Rental Equipment
# Variation Order flow, and both caused by the SHAPE of a workflow chunk rather
# than by anything missing from it:
#
#   1. workflow_vector.py renders an ordered chain only as far as the first
#      value branch ("...the route then branches; who signs next depends on the
#      value"). Everything after that branch - the send-backs, the loops, the
#      close - exists only in the unordered `Routing:` block further down. A
#      model that summarises the readable ordered part and stops looks complete
#      and is not: the first answer named 6 of the 12 routes in the chunk it
#      cited. Naming the `Routing:` block explicitly is what recovers them.
#   2. A route is a sequence, and prose flattens it. The steps, the value
#      branches and the send-backs are three different kinds of fact and are
#      unreadable interleaved in a paragraph.
WORKFLOW_FORMAT = (
    "\n\nThe context includes at least one approval-workflow diagram. If you answer "
    "from one, do not write it as a paragraph and do not use section headings. "
    "Follow this shape exactly:\n\n"
    "The workflow begins with [starting action].\n"
    "1. [Actor] performs [action].\n"
    "2. The process moves to [next actor/action].\n"
    "3. A decision is made:\n"
    "   - If [condition], [outcome].\n"
    "   - If [condition], [outcome].\n"
    "   - If [condition], [outcome].\n"
    "4. If further action is required, [actor] performs [action].\n"
    "5. The workflow ends when [final conditions].\n\n"
    "Filling it in:\n"
    "- Number the steps along the route in order, starting from the creation "
    "step. Use as many or as few steps as the diagram actually has.\n"
    "- Where a step branches on a value or a condition, write that step as "
    "'A decision is made:' with one sub-bullet per branch, each giving the "
    "condition and where it goes.\n"
    "- A route that returns to an earlier step is a sub-bullet on the step it "
    "returns FROM, written as: If [label on that route], it returns to [step].\n"
    "- The final step says which route or routes close the workflow.\n"
    "- Work from the diagram's own `Routing:` block. EVERY route listed there "
    "must appear exactly once in your answer, including the ones that simply "
    "approve and move forward. The numbered chain near the top of a diagram "
    "stops at the first branch, so summarising that chain alone leaves routes "
    "out.\n"
    "- A step's sub-bullets may use ONLY the conditions written on that step's "
    "own outgoing routes in the `Routing:` block. Never carry a condition over "
    "from a different step, and never invent one. If a step has a single "
    "outgoing route, write it as a plain step, not as a decision.\n"
    "- Before finishing, check every route in `Routing:` against what you "
    "wrote. Send-backs and loops are the ones most often dropped: each must be "
    "a sub-bullet on the step it leaves FROM.\n"
    "- After the final step, add one more line reading exactly 'Returns and "
    "loops:' and under it one bullet per route that goes back to an earlier "
    "step, each written as: [step] -> [step] (label on the route). If the "
    "diagram has none, write 'Returns and loops: none.'"
)


# Rule for the case where the context holds several versions of one process.
#
# Retrieval does not separate them and cannot be asked to: the chunks are
# near-identical by construction, and on "subcontract preparation approval
# steps" V1 and V3 come back 0.08 apart. Without this rule the model reads
# both piles of steps as one diagram and writes a single merged workflow -
# every step verbatim, every form number real, the citation a real file, and
# the process itself never approved by anyone. The instruction is what keeps
# the two apart, and naming the chosen version is what makes a wrong pick
# correctable by the reader instead of invisible.
VARIANT_RULE = (
    "\n\nThe context contains more than one VERSION of the same process, marked "
    "with a line beginning '(This is version'. They are different documents. "
    "Do NOT combine them into one sequence.\n"
    "- Answer from ONE version only: the one the question names, or else the "
    "first one in the context.\n"
    "- Begin the answer by saying which version you are describing, as: "
    "This describes the [version] version of [process].\n"
    "- Never take a step, an approver or a threshold from one version and put "
    "it in an answer about another.\n"
    "- End with one line reading exactly 'Other versions in the sources:' "
    "followed by the names of the other versions, so the reader can check "
    "whether they wanted a different one."
)


def build_prompt(question: str, retrieved: list[dict], answer_in_arabic: bool = False) -> str:
    """Context-only prompt. The Arabic directive is the only thing that varies."""
    context = format_context(retrieved)
    refusal = REFUSAL_AR if answer_in_arabic else REFUSAL_EN
    instruction = (
        "Answer using ONLY the context below. Dont just state whats in the context, explain it before stating to refer to the document. If the answer isn't in the context, "
        f"say '{refusal}' Cite the source document ONCE, on its own line at the "
        "very end of the answer, as: Source: <filename>. Never repeat a filename "
        "after individual sentences or bullet points."
    )
    if any(c.get("source_type") in WORKFLOW_SOURCE_TYPES for c in retrieved):
        instruction += WORKFLOW_FORMAT
    if variant_conflicts(retrieved):
        instruction += VARIANT_RULE
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
    # The second pass sees the same chunks, so it can make the same merge. The
    # rule travels with them.
    if variant_conflicts(retrieved):
        prompt += VARIANT_RULE
    if answer_in_arabic:
        prompt += ANSWER_IN_ARABIC
    return prompt + "\n\nANSWER:"


# ------------------------------------------------------------------- ollama
def ollama_up(host: str = OLLAMA_HOST) -> bool:
    try:
        return requests.get(f"{host}/api/tags",
                            timeout=config.PROBE_TIMEOUT).status_code == 200
    except Exception:
        return False


def ollama_models(host: str = OLLAMA_HOST) -> list[str]:
    r = requests.get(f"{host}/api/tags", timeout=config.TAGS_TIMEOUT)
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
        models = requests.get(f"{host}/api/ps",
                              timeout=config.PROBE_TIMEOUT).json().get("models", [])
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
             timeout: int | None = None) -> str:
    """Deterministic generation. temperature=0 and a fixed seed, so a repeated
    question gives a repeated answer.

    `num_predict: -1` because a capped budget truncates long explanatory
    answers mid-sentence with nothing in the output saying so. `num_ctx` is
    pinned at 8192 so Ollama's 4096 default cannot quietly reintroduce a
    ceiling that shrinks as top_k grows.

    Every failure leaves here as a PipelineError carrying a sentence the UI can
    show. See backend/errors.py for why the distinctions are worth keeping.
    """
    timeout = config.GEN_TIMEOUT if timeout is None else timeout
    payload = {
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "num_predict": -1,
                    "num_ctx": config.NUM_CTX, "seed": 0},
        "think": False,
    }
    try:
        resp = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
        if resp.status_code == 400:       # older Ollama, or the model rejects `think`
            payload.pop("think")
            resp = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
    except requests.Timeout as e:
        # Distinct from "server down": the model IS working, just not inside
        # the budget. The two want opposite responses - wait/raise the budget
        # versus go and start Ollama - so they must not share a message.
        raise ModelTimeout(f"no response from {model} within {timeout}s") from e
    except requests.RequestException as e:
        raise ModelUnavailable(f"cannot reach the model server at {host}") from e

    if resp.status_code == 404:
        # Ollama answers 404 for a model it does not have. This is the single
        # most likely failure on a fresh machine and it used to surface as a
        # generic "something went wrong", which sends the reader looking in
        # the wrong place entirely.
        raise ModelNotFound(f"model {model!r} is not pulled on {host}")
    if not resp.ok:
        raise GenerationFailed(f"{resp.status_code} from {host}: {resp.text[:200]}")
    try:
        return resp.json()["response"].strip()
    except (ValueError, KeyError) as e:
        # A 200 whose body is not the expected shape - a proxy or a captive
        # portal in front of the host, most often. Silently returning "" here
        # would produce an empty answer with no explanation anywhere.
        raise GenerationFailed(f"unexpected response body from {host}") from e


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
        self.log = logs.get("chatbot")

    def _gen(self, prompt: str) -> str:
        """Generator handed to translate.py, so translations get the same
        determinism and the same trace stripping as answers."""
        return strip_reasoning(generate(prompt, model=self.model, host=self.host))

    def ask(self, question: str, top_k: int = 6, session_id: str | None = None) -> dict:
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
            # Computed here rather than left to search() so the record can say
            # whether the workflow floor was in play - otherwise "why did a
            # diagram show up / not show up" is unanswerable from the record.
            wf = workflow_intent(q_search)
            hits = self.retriever.search(q_search, top_k=top_k,
                                         boost_prefixes=ctx.boost_prefixes,
                                         wf_intent=wf)
            t_ret = time.perf_counter() - t0
            hits = fit_context(hits)
            prompt = build_prompt(q_search, hits, answer_in_arabic=src_ar)
        else:
            # Continuation: the user said the last answer was incomplete. The
            # chunks are already the right ones - retrieving on "list the rest"
            # would search the corpus for those words and replace the very
            # context being asked about. Replay them and re-prompt instead.
            q_search = ctx.original
            route, t_ret, repairs = None, 0.0, []
            wf = False          # nothing was retrieved, so no floor was applied
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

        # The same class of caveat as vlm_sources, one layer down: not "a model
        # read this picture" but "the extractor did not recover all of this
        # diagram". Deterministic, computed from the chunks - deliberately not
        # asked of the generator, which cannot see chunk metadata and would put
        # the caveat in scope of the same 78% route coverage it is warning about.
        incomplete = incomplete_workflows(hits)

        # Which process versions were in front of the model, and which one it
        # was told to answer from. Recorded whether or not the answer obeyed:
        # this is the list the reader needs to notice they asked about V4 and
        # got V1, and the answer's own text cannot be trusted to carry it.
        variants = variant_conflicts(hits)

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
            "workflow_intent": wf,
            "n_workflow_hits": sum(
                1 for h in hits if h.get("source_type") in WORKFLOW_SOURCE_TYPES
            ),
            "contextualize_s": t_ctx,
            "lang": "ar" if src_ar else "en", "model": self.model,
            "vlm_sources": vlm_sources,
            "incomplete_workflows": incomplete,
            "variant_conflicts": variants,
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
        # One line on disk per answered question. The developer view shows all
        # of this and more, but only to whoever is looking at that moment; a
        # question about an answer given an hour ago needs it to have been
        # written down. Content is excluded unless RME_LOG_CONTENT says
        # otherwise - see logs.log_answer.
        logs.log_answer(rec, self.log)
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
    for con in rec["variant_conflicts"]:
        others = ", ".join(o["variant"] for o in con["others"])
        print(f"  VERSIONS of {con['family']}: answered from "
              f"{con['chosen']['variant']}; also retrieved: {others}")
    for w in rec["incomplete_workflows"]:
        print(f"  INCOMPLETE: {w['filename']} | {w['section']} - "
              + "; ".join(w["gaps"]))
    print(f"\ncited: {rec['validator']['cited'] or 'none'}  "
          f"verdict: {rec['validator']['verdict']}  {rec['total_s']:.1f}s")
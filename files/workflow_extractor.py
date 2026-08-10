"""
Scanned approval-flowcharts -> retrievable chunks, via a vision model.

WHY THIS EXISTS, AND WHY IT IS NOT THE NORMAL PIPELINE
------------------------------------------------------
`adaptive_chunker.py` reads PDF layout. A scanned flowchart has no layout and
no text - `CS Signature Matrix.pdf` is 3 pages of pure image, 0 characters -
so the chunker sees nothing and produces nothing. Silently: before this module
existed, dropping that file into processes_pdf/ moved the `pages with no text`
counter from 9 to 12 and changed nothing else.

OCR alone does not rescue it either, and that is the important part. The
meaning of these diagrams is in the *edges*: which arrow points where, which
decision diamond loops back on "Not Approved", and the red monetary labels
("Over 500 K", "500 K to 3M", "Over 3M") that decide who signs. OCR returns a
bag of role names with every relationship destroyed. So a vision model reads
the topology, not just the glyphs.

    classify()             0 extractable chars on every page -> "workflow"
    render()               pages -> PNG at 140 dpi
    describe()             one VLM call per page -> structured JSON graph
    resolve_shared_lanes() fold a shared approver column into the flows it serves
    audit()                deterministic checks against a second model's OCR
    to_chunks()            one chunk per LANE, prose rendered from the JSON

STRUCTURED JSON, NOT FREE PROSE
-------------------------------
The VLM emits a graph, and the prose that gets embedded is generated from that
graph by `render_prose()` - deterministic, no second model call. Free prose
would read better and be unauditable: a paraphrased threshold is invisible,
and thresholds are the thing users actually ask about.

THIS IS THE FIRST PART OF THE CORPUS THAT IS NOT GROUND TRUTH
--------------------------------------------------------------
Every other chunk is verbatim PDF text. These are a model's reading of an
image, and it is measurably fallible. Observed on `CS Signature Matrix.pdf`,
page 1, across three runs of qwen3.6:27b at temperature 0:

  run 1  dropped the "Over 500 K" label entirely
  run 2  invented "COO -> Operation Manager [Over 500 K]", an arrow that does
         not exist, and hung the real threshold on it
  run 3  duplicated "Operation Manager -> COO" and hung it on the copy

The topology and every other threshold were right all three times. It is one
stubborn label on one crowded junction - which is the point: the failure is
narrow, plausible, and completely invisible in fluent prose.

So `audit()` is deterministic and adversarial rather than a formality:

  * monetary tokens found by a SECOND model (glm-ocr) must all appear in the
    graph - catches omission, run 1;
  * an edge endpoint must be a node in its own lane - catches lanes left
    dangling by a mis-split;
  * A->B and B->A cannot both exist unless one is a rejection loop - catches
    the invented reverse arrow, run 2;
  * the same A->B cannot appear twice with different conditions - catches the
    duplicate, run 3.

None of these can catch a clean misreading: "Over 3M -> VP" instead of CEO
would pass everything. That is why chunks carry `review: "machine"` until a
person has checked them, and why the graph is JSON - six lanes can be read
against three pages in about twenty minutes, which free prose would not allow.

Two consequences enforced elsewhere:
  * chunks carry `source_type: "vlm_description"` so they can be told apart;
  * they must NEVER be fed to validator.build_form_whitelist(), or a
    hallucinated form number would whitelist itself and become unfalsifiable.
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
DEFAULT_SRC = HERE.parent / "processes_pdf"
DEFAULT_OUT = HERE / "workflow_chunks.json"

OLLAMA_HOST = "http://localhost:11434"
# Needs a model with the `vision` capability. qwen3:14b - the model that
# answers questions - does NOT have one; check with /api/tags before swapping.
VISION_MODEL = "qwen3.6:27b"

RENDER_DPI = 140          # legible for 8pt diagram labels without huge payloads
MIN_CHARS_PER_PAGE = 20   # below this a page counts as having no real text

# Monetary thresholds are the highest-value and most-missed content. Matches
# "500 K", "3M", "1,000,000", "Over 500K".
MONEY_RE = re.compile(r"\b\d[\d,\.]*\s*(?:K|M|EGP|LE)\b", re.I)


# --------------------------------------------------------------- classify
# Routing moved to document_router.py. It used to live here as "no page has
# extractable text", which was right while the only workflow in the corpus was
# a 3-page scan - and wrong the moment Workflows/ arrived, where 136 of 139
# diagrams carry a text layer. Re-exported so `from workflow_extractor import
# classify` (adaptive_chunker.chunk_corpus) keeps working.
from document_router import classify, page_text_lengths  # noqa: E402,F401


def page_native_text(pdf: Path) -> list[str]:
    """The PDF's own text layer, per page. Empty string where there is none."""
    import pymupdf

    with pymupdf.open(pdf) as doc:
        return [p.get_text() for p in doc]


def render(pdf: Path, dpi: int = RENDER_DPI) -> list[bytes]:
    import pymupdf

    out = []
    with pymupdf.open(pdf) as doc:
        for page in doc:
            out.append(page.get_pixmap(dpi=dpi).tobytes("png"))
    return out


# ---------------------------------------------------------------- describe
PROMPT = """This image is an approval-routing flowchart from a construction company's process manual.

Transcribe it as JSON with exactly these keys:
  "title": the heading text at the top, verbatim
  "revision": the revision/date footer text, verbatim, or null
  "lanes": a list; each lane is one distinct flow with keys "name", "nodes", "edges"
      "name": the lane's heading box, or the diagram title if there is only one flow
      "nodes": list of {"id": short slug, "label": box text verbatim, "kind": "step"|"decision"|"approver"}
      "edges": list of {"from": node id, "to": node id, "condition": text on the arrow verbatim, or null}
  "escalation": list of {"condition": verbatim threshold text, "approver": role} for EVERY monetary threshold

Rules:
- Transcribe labels EXACTLY as printed, including "(H.O)", "(Site)", "K", "M".
- Follow the ARROW DIRECTIONS. An arrow's head determines "to".
- Dashed arrows are still edges; put any label in the condition.
- A diamond is a decision: give it outgoing edges for each labelled branch
  ("Approved", "Not Approved"), including branches that loop backwards.
- Every red or coloured monetary label is a threshold. List each one in
  "escalation" AND as the condition on its edge. Do not omit any.
- A monetary label on a line is the condition for proceeding ALONG that line to
  the HIGHER approver. Attach it to the edge pointing TOWARD that approver.
  Never attach it to an edge pointing back down to an earlier step.
- Output an edge in ONE direction only. Do not emit both A->B and B->A unless
  the page really draws two separate arrowheads, which is rare and normally
  only happens for a "Not Approved" loop.
- Lines cross on these pages. Where two lines' paths overlap or pass near the
  same box, trace EACH line individually from its start to its own arrowhead
  before transcribing its label - do not let a label or endpoint from one line
  bleed onto the other. A box with a short loop back to itself AND a separate
  line continuing on to a different box are two edges, not one; give each its
  own "to" and its own condition.
- Every arrow either has a label or does not. If a label near an arrow is
  small, faint, or partly illegible, transcribe your best reading rather than
  outputting condition: null - null is only for arrows that are genuinely
  unlabelled on the page, never a fallback for "hard to read".
- A page may contain SEVERAL parallel flows side by side. Each is its own lane.
- A column of senior approvers (for example COO -> VP -> CFO) drawn between two
  flows is SHARED by them. It is not a lane of its own. Repeat those nodes and
  their edges inside EVERY lane that routes into them, and never emit a lane
  whose name is just the page title.
- Do not invent nodes or thresholds. If an arrow is unlabelled, condition is null.
Output ONLY the JSON."""


def describe(png: bytes, model: str = VISION_MODEL, host: str = OLLAMA_HOST) -> dict:
    """One page -> the graph the VLM reads off it. Deterministic settings."""
    payload = {
        "model": model,
        "prompt": PROMPT,
        "images": [base64.b64encode(png).decode()],
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": -1},
    }
    r = requests.post(f"{host}/api/generate", json=payload, timeout=1800)
    if r.status_code == 400:
        payload.pop("think")
        r = requests.post(f"{host}/api/generate", json=payload, timeout=1800)
    r.raise_for_status()
    return json.loads(r.json()["response"])


# ------------------------------------------------------------------ repair
def resolve_shared_lanes(graph: dict) -> list[str]:
    """Fold a shared approver column back into the lanes that route into it.

    Mutates `graph`. Returns a list of human-readable notes about what moved.

    The problem: page 2 draws one `COO -> VP -> CFO` column between the
    Engineering Material and Steel flows, serving both. The model emitted it as
    a third lane named after the page title, leaving each real lane with an
    edge pointing at a `coo` node it does not contain. Chunked as-is that gives
    you two lanes whose escalation dead-ends, plus a nameless third chunk that
    is not a workflow at all.

    The fix is deterministic rather than prompt-dependent, because the prompt
    can only ever make this rarer. Any edge pointing at a node that lives in a
    sibling lane pulls that node in, then keeps pulling along the chain, so the
    whole escalation path arrives in the lane that needs it. A donor lane whose
    every node has been absorbed elsewhere is then dropped: its content is not
    lost, it now appears in each flow it actually serves.
    """
    lanes = graph.get("lanes") or []
    if len(lanes) < 2:
        return []

    index = {}                       # node id -> (node, owning lane)
    for lane in lanes:
        for n in lane.get("nodes") or []:
            index.setdefault(n.get("id"), (n, lane))

    notes, absorbed = [], {id(l): set() for l in lanes}
    for lane in lanes:
        own = {n.get("id") for n in lane.get("nodes") or []}
        frontier = [e.get(k) for e in lane.get("edges") or []
                    for k in ("from", "to") if e.get(k) not in own]
        pulled = set()
        while frontier:
            nid = frontier.pop()
            if nid in own or nid not in index:
                continue
            node, donor = index[nid]
            if donor is lane:
                continue
            lane.setdefault("nodes", []).append(dict(node))
            own.add(nid)
            pulled.add(nid)
            absorbed[id(donor)].add(nid)
            # Follow the chain onward (COO -> VP -> CFO), bringing its edges.
            for e in donor.get("edges") or []:
                if e.get("from") == nid or e.get("to") == nid:
                    if e not in (lane.get("edges") or []):
                        lane.setdefault("edges", []).append(dict(e))
                    for end in (e.get("from"), e.get("to")):
                        if end not in own:
                            frontier.append(end)
        if pulled:
            notes.append(f"lane {lane.get('name')!r}: pulled in shared approver(s) "
                         f"{', '.join(sorted(pulled))}")

    title = (graph.get("title") or "").strip().lower()
    keep = []
    for lane in lanes:
        ids = {n.get("id") for n in lane.get("nodes") or []}
        fully_absorbed = ids and ids <= absorbed[id(lane)]
        looks_like_title = (lane.get("name") or "").strip().lower() == title
        if fully_absorbed and looks_like_title:
            notes.append(f"dropped pseudo-lane {lane.get('name')!r} - it was the shared "
                         "approver column, now folded into the flows that use it")
            continue
        keep.append(lane)
    graph["lanes"] = keep
    return notes


# ------------------------------------------------------------------- audit
OCR_MODEL = "glm-ocr:latest"   # vision, 2.2 GB - a reader, not a reasoner

OCR_PROMPT = ("Transcribe every piece of text visible in this image, including "
              "small and coloured labels on the arrows. Output the text only, "
              "one item per line, no commentary.")


def ocr_page_text(png: bytes, model: str = OCR_MODEL, host: str = OLLAMA_HOST) -> str:
    """Raw text off the page from a SECOND model, independent of the VLM.

    Only ever used to check the first model's work, never as corpus content.

    This deliberately does not use tesseract. The tesseract binary is not
    installed here (see README) and pytesseract is not a dependency, so that
    path was a silent no-op - the audit reported zero warnings because it was
    doing nothing. `glm-ocr` is already pulled, is an OCR specialist rather
    than a diagram reasoner, and disagreeing with `qwen3.6:27b` is exactly
    what makes it useful as a second opinion.

    Returns '' on any failure, which degrades the audit to its structural
    checks rather than failing the extraction.
    """
    payload = {
        "model": model, "prompt": OCR_PROMPT,
        "images": [base64.b64encode(png).decode()],
        "stream": False, "think": False,
        "options": {"temperature": 0, "seed": 0, "num_predict": -1},
    }
    try:
        r = requests.post(f"{host}/api/generate", json=payload, timeout=1800)
        if r.status_code == 400:
            payload.pop("think")
            r = requests.post(f"{host}/api/generate", json=payload, timeout=1800)
        r.raise_for_status()
        return r.json()["response"]
    except Exception:
        return ""


def reference_texts(pdf: Path, pages: list[bytes], host: str = OLLAMA_HOST) -> tuple[list[str], str]:
    """The independent text channel `audit()` checks the graph against.

    Returns (per-page text, which channel produced it).

    Prefers the PDF's OWN text layer, and that is a genuine upgrade rather than
    a shortcut. The audit's job is to prove the vision model did not drop a
    label; glm-ocr can only do that approximately, because it is a second model
    guessing at the same pixels and its own misreads show up as false warnings.
    The text layer is what the document actually says - exact, free, and
    available on 136 of the 139 diagrams in Workflows/.

    It is emphatically NOT a substitute for the vision pass. The text layer
    carries labels with every relationship destroyed ("Commercial Dept. / End /
    QAM No / COO / Submit"), which is the whole reason this module exists. It
    can say a label is missing from the graph; it cannot say where an arrow
    points.

    Falls back to glm-ocr for genuinely scanned pages, so the original
    CS Signature Matrix path is unchanged.
    """
    native = page_native_text(pdf)
    if all(len(t.strip()) >= MIN_CHARS_PER_PAGE for t in native):
        return native, "native_text"
    if any(len(t.strip()) >= MIN_CHARS_PER_PAGE for t in native):
        # Mixed: keep native where it exists, OCR only the pages missing it.
        out = [t if len(t.strip()) >= MIN_CHARS_PER_PAGE
               else ocr_page_text(png, host=host) for t, png in zip(native, pages)]
        return out, "mixed_native_ocr"
    return [ocr_page_text(png, host=host) for png in pages], "ocr_model"


# Text in the reference channel that is page furniture rather than a diagram
# label, so its absence from the graph means nothing.
_FURNITURE_RE = re.compile(
    r"^(?:page\s*\d+|\d+\s*/\s*\d+|rev(?:ision)?\.?\s*\S*|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|[\W\d_]*)$",
    re.I)


def missing_labels(graph: dict, reference: str, exact: bool) -> list[str]:
    """Reference-channel lines that appear nowhere in the graph.

    Only run when the reference is the PDF's own text layer (`exact`). Against
    OCR this check is noise - the reader's own errors become phantom "missing"
    labels - which is why it never existed while glm-ocr was the only channel.

    Deliberately reported as one warning listing everything, not one per line:
    a diagram whose graph missed six boxes has one problem, not six.
    """
    if not exact:
        return []
    blob = re.sub(r"\s+", " ", json.dumps(graph, ensure_ascii=False)).lower()
    missing = []
    for raw in (reference or "").splitlines():
        line = " ".join(raw.split())
        if len(line) < 3 or _FURNITURE_RE.match(line):
            continue
        if line.lower() not in blob:
            missing.append(line)
    # Dedupe, keep order: a label printed twice on the page is one label.
    seen, out = set(), []
    for m in missing:
        if m.lower() not in seen:
            seen.add(m.lower())
            out.append(m)
    return out


def audit(graph: dict, ocr_text: str, exact_reference: bool = False) -> list[str]:
    """Deterministic completeness check on one page's graph.

    The failure this is built for: the VLM reads most of a diagram correctly
    and drops one threshold. That is invisible downstream - the answer is
    fluent, cites the right document, and routes the approval to the wrong
    person. Comparing against independently OCR'd monetary tokens catches it.
    """
    warnings = []
    blob = json.dumps(graph, ensure_ascii=False)

    seen = {m.group(0).replace(" ", "").upper() for m in MONEY_RE.finditer(blob)}
    channel = "The page text" if exact_reference else "OCR"
    for m in MONEY_RE.finditer(ocr_text or ""):
        tok = m.group(0).replace(" ", "").upper()
        if tok not in seen:
            warnings.append(
                f"{channel} has the amount {m.group(0)!r} but it is absent from the graph")

    # Only possible when the reference channel is the PDF's own text layer.
    # Against a scanned page this stays silent rather than guessing.
    gone = missing_labels(graph, ocr_text, exact_reference)
    if gone:
        shown = "; ".join(f"{g!r}" for g in gone[:8])
        warnings.append(
            f"{len(gone)} label(s) printed on the page are absent from the graph: {shown}"
            + (" ..." if len(gone) > 8 else ""))

    lanes = graph.get("lanes") or []
    if not lanes:
        warnings.append("no lanes extracted")
    for lane in lanes:
        ids = {n.get("id") for n in lane.get("nodes") or []}
        for e in lane.get("edges") or []:
            for end in ("from", "to"):
                if e.get(end) not in ids:
                    warnings.append(
                        f"lane {lane.get('name')!r}: edge {end}={e.get(end)!r} is not a node")
        if not lane.get("edges"):
            warnings.append(f"lane {lane.get('name')!r}: no edges - topology was not read")

        # Reciprocal edges: A->B when B->A already exists. A rejection loop is
        # a legitimate backwards edge, so those are exempt - but an unlabelled
        # or threshold-labelled reciprocal pair means the model turned one
        # arrow into two and had to invent a direction for the second.
        #
        # This is not hypothetical. On page 1 of CS Signature Matrix the model
        # produced both "Operation Manager -> COO" (real) and
        # "COO -> Operation Manager [Over 500 K]" (invented), attaching a real
        # threshold to an arrow that does not exist. Presence-checking the
        # amount against OCR cannot catch that - the token IS present, just
        # bound to the wrong edge - so this check is the one that would.
        # Duplicate parallel edges: the same from->to emitted twice. A decision
        # box fanning out several distinctly-labelled branches to the same
        # target (several ways to reach "End", say) is normal and not this bug
        # - what the second run on page 1 actually did was produce both
        # "Operation Manager -> COO" and "Operation Manager -> COO [Over 500 K]",
        # attaching a real threshold to a bare COPY of an existing arrow. That
        # signature is either a blank condition sitting alongside a labelled
        # one, or the exact same condition text repeated - not merely several
        # distinct labels sharing a target.
        grouped: dict[tuple, list] = {}
        for e in lane.get("edges") or []:
            grouped.setdefault((e.get("from"), e.get("to")), []).append(e.get("condition"))
        for (a, b), conds in grouped.items():
            if len(conds) > 1:
                norm = [str(c).strip().lower() if c else "" for c in conds]
                has_blank = "" in norm
                has_repeat = len(set(norm)) < len(norm)
                if has_blank or has_repeat:
                    warnings.append(
                        f"lane {lane.get('name')!r}: {a}->{b} appears {len(conds)} times "
                        f"with conditions {conds} - the threshold is probably on a "
                        "different arrow")

        # Reciprocal edges: A->B and B->A both present. A real two-way
        # relationship (a revision loop, "resend" paired with "send back") is
        # drawn with a distinct label on each direction and is not a bug.
        # What run 2 actually did was invent a reverse arrow to hang a real
        # threshold on - so the genuine side of that pair had NO label. Flag
        # reciprocal edges only when one direction is unlabelled (the invented
        # arrow's real partner had nothing to say) or both sides carry the
        # identical condition text (the label got copied onto a fabricated
        # copy rather than describing its own arrow).
        pairs = {}
        for e in lane.get("edges") or []:
            pairs[(e.get("from"), e.get("to"))] = e.get("condition")
        for (a, b), cond in pairs.items():
            if (b, a) in pairs and str(a) < str(b):
                back = pairs[(b, a)]
                if not _is_loop(cond) and not _is_loop(back):
                    blank = not cond or not back
                    same = bool(cond) and bool(back) and str(cond).strip().lower() == str(back).strip().lower()
                    if blank or same:
                        warnings.append(
                            f"lane {lane.get('name')!r}: contradictory edges "
                            f"{a}->{b} [{cond}] and {b}->{a} [{back}] - one direction "
                            "is probably invented")
    return warnings


_LOOP_WORDS = ("not approved", "rejected", "reject", "return", "revise", "مرفوض")


def _is_loop(condition) -> bool:
    """Is this edge a legitimate rejection loop back to an earlier step?"""
    return bool(condition) and any(w in str(condition).lower() for w in _LOOP_WORDS)


# ------------------------------------------------------------------- prose
def main_sequence(lane: dict) -> tuple[list[str], bool]:
    """The lane's forward approval path in order, and whether it was truncated.

    Returns (node_ids, branched). `branched` means the walk stopped at a real
    fork rather than at the end of the flow.

    STOPPING AT FORKS IS THE WHOLE POINT. A first version of this walked
    through branches by taking the first edge, and on page 1 - where COO goes
    to VP for "500 K to 3M" and to CEO for "Over 3M" - it silently dropped the
    CEO and stated "VP signs last". The chunk then contradicted its own
    thresholds section, and "who approves over 3M", which had answered
    correctly before, started refusing.

    A linear order cannot be told about a branching graph. Where the flow
    forks, the ordered list stops and the reader is sent to the arrow list and
    the thresholds, which represent branching honestly.

    Rejection loops are excluded so the walk cannot cycle; they are backwards
    edges, not forks.
    """
    edges = [e for e in lane.get("edges") or [] if not _is_loop(e.get("condition"))]
    ids = [n.get("id") for n in lane.get("nodes") or []]
    if not edges or not ids:
        return [], False

    targets = {e.get("to") for e in edges}
    starts = [i for i in ids if i not in targets]
    if not starts:
        return [], False

    order, seen, cur = [], set(), starts[0]
    while cur and cur not in seen:
        order.append(cur)
        seen.add(cur)
        out = [e for e in edges if e.get("from") == cur and e.get("to") not in seen]
        # Distinct destinations - two edges to the same node are a duplicate,
        # already reported by audit(), not a fork.
        dests = {e.get("to") for e in out}
        if len(dests) > 1:
            return order, True
        if not out:
            break
        cur = out[0].get("to")
    return order, False


def render_prose(graph: dict, lane: dict) -> str:
    """Deterministic prose for one lane. This is what gets embedded.

    Written for retrieval rather than for reading: role names and threshold
    text appear verbatim and often, because BM25 matches literal tokens and
    those are what a question about approvals contains.
    """
    labels = {n["id"]: n.get("label", n["id"]) for n in lane.get("nodes") or []}
    name = lane.get("name") or graph.get("title") or "Workflow"

    lines = [f"{graph.get('title', '')} - {name}".strip(" -"),
             f"Approval workflow: {name}.", ""]

    steps = [n for n in lane.get("nodes") or [] if n.get("kind") != "approver"]
    if steps:
        lines.append("Steps: " + "; ".join(n.get("label", "") for n in steps) + ".")

    # Order, spelled out. The arrow list below carries the same information,
    # but only as direction that has to be interpreted - and asked "who signs
    # before and after the VP", the generator read `COO -> VP -> CFO` and
    # answered that the CFO comes *before* the VP. A confidently inverted
    # answer, on a chunk retrieved at 0.998. Numbering the path and naming
    # each neighbour removes the inference instead of hoping it goes well.
    seq, branched = main_sequence(lane)
    if len(seq) > 1:
        header = ("Order of approval so far, first to last (each one signs after the "
                  "one before it); the route then branches:" if branched else
                  "Order of approval, first to last "
                  "(each one signs after the one before it):")
        lines.append("")
        lines.append(header)
        for i, nid in enumerate(seq, 1):
            lines.append(f"  {i}. {labels.get(nid, nid)}")
        if branched:
            # Never imply a last signer on a flow that forks - who signs last
            # depends on the branch, and the thresholds below decide that.
            lines.append(f"  After {labels.get(seq[-1], seq[-1])} the route branches; "
                         "who signs next depends on the value, see Routing and "
                         "Approval thresholds below.")
        # Deliberately NOT followed by a per-node "X signs after Y and before Z"
        # sentence for every step. That version was tried and measured worse:
        # it roughly doubled the chunk, and on "who approves a comparison sheet
        # over 3M" - which had been answering correctly - it pushed the useful
        # content far enough down that the model started refusing. The numbered
        # list carries the same ordering in a fraction of the tokens.

    lines.append("")
    lines.append("Routing:")
    for e in lane.get("edges") or []:
        src, dst = labels.get(e.get("from"), e.get("from")), labels.get(e.get("to"), e.get("to"))
        cond = e.get("condition")
        lines.append(f"  - {src} -> {dst}" + (f" [{cond}]" if cond else ""))

    # `escalation` is extracted per PAGE, but a page holds several lanes and a
    # threshold belongs to exactly one of them. Attaching all of them to every
    # lane would undo the reason for splitting by lane in the first place: the
    # Wood-Formwork chunk would claim the "Over 500 K -> COO" rule that
    # actually governs the Consumable Material lane beside it.
    #
    # A threshold belongs to this lane if its condition text labels one of this
    # lane's edges. Anything unmatched is reported at page level rather than
    # silently dropped, since an unattributable threshold is worth seeing.
    conds = {str(e.get("condition")).strip().lower()
             for e in lane.get("edges") or [] if e.get("condition")}
    mine, unplaced = [], []
    for e in graph.get("escalation") or []:
        c = str(e.get("condition", "")).strip().lower()
        (mine if c in conds else unplaced).append(e)

    if mine:
        # Phrased as a full sentence pairing the word "value" with the
        # threshold text, rather than a terse "Any Value: COO" bullet.
        # Measured reason: with the bullet form, "at what value does the COO
        # approve?" was REFUSED on a chunk retrieved at score 1.000 whose text
        # contained the answer - the generator did not read "Any Value" as an
        # answer to "what value". Retrieval was never the problem.
        lines.append("")
        lines.append("Approval thresholds:")
        for e in mine:
            cond, who = e.get("condition"), e.get("approver")
            lines.append(f"  - A purchase value of \"{cond}\" requires approval by {who}. "
                         f"{who} approves at value: {cond}.")
    if unplaced:
        lines.append("")
        lines.append("Thresholds recorded on this page but not tied to this flow: "
                     + "; ".join(f"{e.get('condition')} -> {e.get('approver')}"
                                 for e in unplaced))

    if graph.get("revision"):
        lines.append("")
        lines.append(str(graph["revision"]))
    return "\n".join(lines)


# ------------------------------------------------------------------ chunks
def chunks_from_graph(graph: dict, filename: str, page: int, model: str,
                      warns: list[str]) -> list[dict]:
    """One page's graph -> one chunk per lane. No model call."""
    out = []
    for lane in graph.get("lanes") or []:
        out.append({
            "filename": filename,
            "doc_code": None,          # workflows carry no P-XX-NN code
            "title": graph.get("title") or Path(filename).stem,
            "section": lane.get("name") or "WORKFLOW",
            "step": None,
            "text": render_prose(graph, lane),
            "raw_heading": graph.get("title"),
            "label_method": "vlm",
            "label_score": 0.0,
            "page": page,
            "detector": "vision",
            # --- what marks this as not-ground-truth ---
            "source_type": "vlm_description",
            "vision_model": model,
            # Same convention as eval_set_ar.json: machine-produced until a
            # person has checked it against the page. Measured error rate on
            # the first run was one wrong threshold binding in six lanes, so
            # this is not a formality.
            "review": "machine",
            "audit_warnings": warns,
            "graph": lane,
        })
    return out


def rebuild_from_audit(audit_path: Path, model: str = VISION_MODEL) -> list[dict]:
    """Re-render chunks from a saved audit file, without touching the VLM.

    The graphs are the expensive artifact; the prose around them is not. When
    render_prose changes - and it has, twice - re-running three 27B vision
    calls to regenerate text that is a pure function of saved JSON is waste.
    """
    return rebuild_chunks(json.load(open(audit_path, encoding="utf-8")), model=model)


def rebuild_chunks(records: list[dict], model: str = VISION_MODEL) -> list[dict]:
    """Audit records -> lane chunks. Pure function of the saved graphs.

    Split out from rebuild_from_audit so the extraction loop can call it after
    every document to checkpoint, without a round trip through the filesystem.
    """
    chunks = []
    for rec in records:
        resolve_shared_lanes(rec["graph"])
        # `reference_text` is the current field; `ocr_text` is what audit files
        # written before the native-text channel existed call it.
        ref = rec.get("reference_text") or rec.get("ocr_text") or ""
        warns = audit(rec["graph"], ref, exact_reference=rec.get("channel") == "native_text")
        chunks += chunks_from_graph(rec["graph"], rec["filename"], rec["page"],
                                    model, warns)
    return chunks


def to_chunks(pdf: Path, model: str = VISION_MODEL, host: str = OLLAMA_HOST,
              on_page=None) -> tuple[list[dict], list[dict]]:
    """Whole scanned PDF -> (chunks, per-page audit records).

    One chunk per LANE, not per file. `CS Signature Matrix.pdf` holds six
    distinct approval flows across three pages; as a single chunk, a question
    about steel would retrieve caravans and formwork too and leave the
    generator to guess which chain applies.
    """
    chunks, audits = [], []
    pages = render(pdf)

    # Reference text first: on a native-text diagram this is a free file read,
    # so a PDF that turns out to have no usable text costs nothing before the
    # expensive pass starts.
    refs, channel = reference_texts(pdf, pages, host=host)
    exact = channel == "native_text"

    # One model at a time. Interleaving describe() and the OCR fallback would
    # swap a 17 GB model and a 2 GB model in and out of VRAM once per page.
    graphs = [describe(png, model=model, host=host) for png in pages]

    for i, (graph, ref) in enumerate(zip(graphs, refs), start=1):
        # Repair first, then audit - so warnings describe the graph that
        # actually becomes chunks, not the raw model output.
        repairs = resolve_shared_lanes(graph)
        warns = audit(graph, ref, exact_reference=exact)
        audits.append({"filename": pdf.name, "source": pdf.as_posix(), "page": i,
                       "title": graph.get("title"), "warnings": warns,
                       "repairs": repairs, "graph": graph,
                       "reference_text": ref, "channel": channel})
        if on_page:
            on_page(i, graph, warns, repairs)

        chunks += chunks_from_graph(graph, pdf.name, i, model, warns)
    return chunks, audits


def main() -> None:
    import argparse

    from translate import enable_utf8_stdout

    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="Extract workflow diagrams into lane chunks")
    ap.add_argument("--src", type=Path, nargs="+", default=[DEFAULT_SRC],
                    help="directories to scan; searched recursively")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--audit", type=Path, default=None,
                    help="per-page audit file. With --resume this is also the "
                         "checkpoint, and it is rewritten after every document.")
    ap.add_argument("--model", default=VISION_MODEL)
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N documents - sample before committing to "
                         "a full run")
    ap.add_argument("--resume", action="store_true",
                    help="skip documents already present in --audit")
    ap.add_argument("--from-audit", type=Path, default=None,
                    help="re-render chunks from a saved audit file instead of "
                         "calling the vision model again")
    args = ap.parse_args()

    if args.from_audit:
        chunks = rebuild_from_audit(args.from_audit, model=args.model)
        args.out.write_text(json.dumps(chunks, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        warned = sum(1 for c in chunks if c["audit_warnings"])
        print(f"Rebuilt {args.out.name} from {args.from_audit.name}: "
              f"{len(chunks)} lane chunks, {warned} carrying warnings (no model calls)")
        return

    from document_router import classify_dir

    records = []
    for src in args.src:
        if not src.exists():
            raise SystemExit(f"No such directory: {src}")
        records += classify_dir(src)
    workflows = [Path(r["path"]) for r in records if r["kind"] == "workflow"]
    n_scanned = sum(1 for r in records if r["kind"] == "workflow" and r["scanned"])
    print(f"{len(records)} PDFs: {len(workflows)} workflow "
          f"({len(workflows) - n_scanned} native text, {n_scanned} scanned), "
          f"{len(records) - len(workflows)} process")

    # Resume against the audit file, which is the only complete record of what
    # has already been through the vision model. Keyed by source path, so two
    # diagrams with the same basename in different folders stay distinct.
    all_audits = []
    if args.resume and args.audit and args.audit.exists():
        all_audits = json.load(open(args.audit, encoding="utf-8"))
        done = {a.get("source") or a.get("filename") for a in all_audits}
        before = len(workflows)
        workflows = [p for p in workflows if p.as_posix() not in done]
        print(f"resuming: {before - len(workflows)} already extracted, "
              f"{len(workflows)} to go")

    if args.limit:
        workflows = workflows[:args.limit]
        print(f"limited to {len(workflows)} document(s) this run")

    def _flush():
        """Write both artifacts. Called after every document, because a run of
        this length WILL be interrupted and a crash at document 130 must not
        throw away 129 vision passes."""
        chunks = rebuild_chunks(all_audits, model=args.model)
        args.out.write_text(json.dumps(chunks, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        if args.audit:
            args.audit.write_text(json.dumps(all_audits, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        return chunks

    for n, p in enumerate(workflows, start=1):
        print(f"\n--- [{n}/{len(workflows)}] {p.name} ---")

        def _log(i, graph, warns, repairs=()):
            lanes = [l.get("name") for l in graph.get("lanes") or []]
            print(f"  p{i}: {graph.get('title')!r}  lanes={lanes}")
            for r in repairs:
                print(f"      repaired: {r}")
            for w in warns:
                print(f"      WARNING: {w}")

        try:
            _, a = to_chunks(p, model=args.model, on_page=_log)
        except Exception as e:
            # One unreadable diagram must not end the run. It is absent from
            # the audit file, so --resume retries it next time.
            print(f"      FAILED: {type(e).__name__}: {e}")
            continue
        all_audits += a
        _flush()

    chunks = _flush()
    n_warn = sum(len(a["warnings"]) for a in all_audits)
    warned = sum(1 for c in chunks if c["audit_warnings"])
    print(f"\nWrote {args.out.name}: {len(chunks)} lane chunks from "
          f"{len(all_audits)} pages, {n_warn} audit warning(s) on {warned} chunk(s)")
    if args.audit:
        print(f"Wrote {args.audit.name}")
    if n_warn:
        print("\nWarnings are not fatal, but every one is a place the diagram was "
              "read incompletely. Check those pages against the PDF before trusting "
              "answers built from them.")


if __name__ == "__main__":
    main()

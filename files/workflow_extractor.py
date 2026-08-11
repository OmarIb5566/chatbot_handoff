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
import math
import re
import statistics
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
DEFAULT_SRC = HERE.parent / "processes_pdf"
DEFAULT_OUT = HERE / "workflow_chunks.json"

OLLAMA_HOST = "http://localhost:11434"
# Needs a model with the `vision` capability. qwen3:14b - the model that
# answers questions - does NOT have one; check with /api/tags before swapping.
VISION_MODEL = "qwen3.6:27b"

# Force the vision model onto the CPU, and ONLY the vision model.
#
# With Ollama's Vulkan backend enabled (OLLAMA_VULKAN=1, needed to reach an
# Intel Arc GPU) qwen3:14b runs fully on the GPU and the chatbot gets much
# faster - but this model dies partway through every request:
#
#     slot process_mtmd: id 0 | task 0 | encoding mtmd batch from idx = 4
#     [GIN] 500 | POST "/api/generate"
#     an error was encountered while running the model: ... forcibly closed
#
# The LLM half loads and runs; it is the multimodal projector that Vulkan
# cannot execute. llama.cpp has --no-mmproj-offload for exactly this, and the
# flag is present inside the ollama binary, but no OLLAMA_* env var exposes it -
# so the projector cannot be placed separately and the whole model has to come
# back to the CPU.
#
# Set per-request rather than by unsetting OLLAMA_VULKAN, because that variable
# is global and the interactive path genuinely wants it. None = let Ollama
# decide, which is correct on a CUDA machine and on a CPU-only one.
VISION_NUM_GPU: int | None = 0

RENDER_DPI = 140          # legible for 8pt diagram labels without huge payloads
MIN_CHARS_PER_PAGE = 20   # below this a page counts as having no real text

# --- tiling: see plan_tiles() for the measurement these come from ---
TARGET_TILE_PX = 1400     # long edge every tile is rendered to
MIN_LABEL_PX = 12         # a label must survive the model's own downscale
MAX_TILES_PER_PAGE = 12   # cost ceiling; the split coarsens rather than exceed it
TILE_OVERLAP_FRAC = 0.12  # so an arrow crossing a split is whole in one tile
MIN_GUTTER_PT = 8         # empty run wide enough to cut through without hitting a box
CONTENT_PAD_PT = 8

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
    """Whole page, one PNG each. Kept for callers that want the old behaviour;
    the extraction path uses render_tiles()."""
    import pymupdf

    out = []
    with pymupdf.open(pdf) as doc:
        for page in doc:
            out.append(page.get_pixmap(dpi=dpi).tobytes("png"))
    return out


# -------------------------------------------------------------------- tiles
# WHY A WHOLE PAGE IS SOMETIMES THE WRONG UNIT
# --------------------------------------------
# These diagrams are Visio exports, and the canvas is whatever Visio was set to.
# Across the 133 extracted pages, grouped by the longest page edge:
#
#     under 1700 pt (A4/A3)   85 pages   58 clean   0.4 warnings/page
#     1700-3000 pt            27 pages   12 clean   0.8
#     over 3000 pt            21 pages    1 clean   8.9
#
# `Subcontracts - Subcontract Preparation V4.pdf` is 7082 x 4642 pt - a 66-inch
# canvas. Rendered whole at RENDER_DPI it is a 9288 px PNG, which the vision
# model's own preprocessor then downscales to roughly 1-1.5k px on the long
# edge. A 16 pt label ends up two or three pixels tall. The model is not
# misreading those pages, it cannot see them: on one 399-line diagram it
# returned 16 nodes and 21 edges, and 136 printed labels never appeared at all.
#
# What matters is therefore not the render DPI but the ratio
#
#     label height / page extent
#
# because the model's input size is fixed no matter what we send. So the page is
# cut into tiles small enough that a label survives that downscale, each tile is
# read separately, and the graphs are merged on label identity.
#
# Cuts are placed in GUTTERS - runs of empty space no box or line crosses - so a
# tile boundary never bisects a node and invents two half-labelled ones. Tiles
# then overlap, because an *arrow* crossing a boundary is unavoidable and the
# overlap keeps both of its endpoints visible in at least one tile.
#
# A page that already fits is a single tile cropped to its content, rendered
# exactly as before - so the 85 pages that were clean stay on their current path.
def _content_rects(page) -> list:
    """Every ink-bearing rectangle on the page: word boxes and vector shapes."""
    import pymupdf

    rects = [pymupdf.Rect(w[:4]) for w in page.get_text("words")]
    for d in page.get_drawings():
        r = d.get("rect")
        if r is not None and r.width > 0 and r.height > 0:
            rects.append(r)
    return rects


def content_bbox(page, rects: list | None = None):
    """The drawn area, padded. Visio pages are routinely half empty margin, and
    cropping to content is free resolution before any tiling happens."""
    rects = _content_rects(page) if rects is None else rects
    if not rects:
        return page.rect
    box = rects[0]
    for r in rects[1:]:
        box |= r
    box += (-CONTENT_PAD_PT, -CONTENT_PAD_PT, CONTENT_PAD_PT, CONTENT_PAD_PT)
    return box & page.rect


def _gutters(rects: list, axis: int, lo: float, hi: float) -> list[float]:
    """Centres of the empty runs along `axis` (0 = x, 1 = y) between lo and hi.

    A cut here passes through whitespace, so no node box is bisected.
    """
    spans = sorted((r.x0, r.x1) if axis == 0 else (r.y0, r.y1) for r in rects)
    merged: list[list[float]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out = []
    for (_, end), (start, _) in zip(merged, merged[1:]):
        if start - end >= MIN_GUTTER_PT and lo < (end + start) / 2 < hi:
            out.append((end + start) / 2)
    return out


def _split_axis(rects: list, axis: int, lo: float, hi: float,
                max_extent: float) -> list[float]:
    """Cut positions spanning [lo, hi], each band at most `max_extent` wide.

    Bands are equal-width in intent and gutter-snapped in practice: the ideal
    cut is computed first, then moved to the nearest gutter if one is close
    enough. `close enough` is half a band - beyond that the gutter is serving a
    different part of the diagram and snapping to it would produce one huge tile
    and one sliver.
    """
    n = max(1, math.ceil((hi - lo) / max_extent))
    if n == 1:
        return [lo, hi]
    band = (hi - lo) / n
    gut = _gutters(rects, axis, lo, hi)
    cuts = set()
    for i in range(1, n):
        ideal = lo + band * i
        if gut:
            best = min(gut, key=lambda g: abs(g - ideal))
            cuts.add(best if abs(best - ideal) <= band / 2 else ideal)
        else:
            cuts.add(ideal)
    return [lo] + sorted(cuts) + [hi]


def plan_tiles(page) -> list:
    """The regions of `page` to render, in PDF coordinates.

    One tile means "render it whole, as before". Returns rects, never bytes, so
    the plan can be inspected and asserted on without rendering anything.
    """
    heights = [w[3] - w[1] for w in page.get_text("words") if w[3] > w[1]]
    if not heights:
        # A genuinely scanned page: no text layer, so no basis for a label-size
        # estimate and no checklist to seed either. Unchanged from before.
        return [page.rect]

    rects = _content_rects(page)
    box = content_bbox(page, rects)

    # The tile extent at which a median label still lands at MIN_LABEL_PX once
    # the model has scaled the tile down to its own input size.
    max_extent = statistics.median(heights) * TARGET_TILE_PX / MIN_LABEL_PX
    if max(box.width, box.height) <= max_extent:
        return [box]

    # Coarsen until the tile count is affordable. Better a slightly small label
    # on a few pages than 40 vision calls on one diagram.
    while True:
        xs = _split_axis(rects, 0, box.x0, box.x1, max_extent)
        ys = _split_axis(rects, 1, box.y0, box.y1, max_extent)
        if (len(xs) - 1) * (len(ys) - 1) <= MAX_TILES_PER_PAGE:
            break
        max_extent *= 1.25

    ov = max_extent * TILE_OVERLAP_FRAC
    tiles = []
    for y0, y1 in zip(ys, ys[1:]):
        for x0, x1 in zip(xs, xs[1:]):
            tiles.append(box & _rect(x0 - ov, y0 - ov, x1 + ov, y1 + ov))
    return tiles


def _rect(x0, y0, x1, y1):
    import pymupdf

    return pymupdf.Rect(x0, y0, x1, y1)


def tile_labels(page, rect) -> list[str]:
    """The PDF's own text inside `rect`, one entry per text block.

    Blocks rather than lines on purpose: the text layer word-wraps a single box
    caption across several lines ('Operation/MEP/Si' + 'te TO Engineer'), and
    feeding those to the model as separate items would ask it to find two
    labels that do not exist. A block is closer to one drawn box.
    """
    out, seen = [], set()
    for b in page.get_text("blocks", clip=rect):
        line = " ".join(str(b[4]).split())
        if len(line) < 3 or _FURNITURE_RE.match(line):
            continue
        if line.lower() not in seen:
            seen.add(line.lower())
            out.append(line)
    return out


def render_tiles(pdf: Path, enabled: bool = True) -> list[list[dict]]:
    """Per page, the tiles to send to the vision model.

    Each tile is {"rect", "png", "labels"}. `enabled=False` restores whole-page
    rendering, for A/B against this change.
    """
    import pymupdf

    out = []
    with pymupdf.open(pdf) as doc:
        for page in doc:
            regions = plan_tiles(page) if enabled else [page.rect]
            single = len(regions) == 1
            tiles = []
            for rect in regions:
                scale = TARGET_TILE_PX / max(rect.width, rect.height)
                if single:
                    # Never render a whole page smaller than it used to be.
                    # Multi-tile pages take the computed scale as-is: going
                    # above it only inflates the payload, since the model
                    # downscales to a fixed size regardless.
                    scale = max(scale, RENDER_DPI / 72)
                pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=rect)
                tiles.append({"rect": tuple(rect), "png": pix.tobytes("png"),
                              "labels": tile_labels(page, rect)})
            out.append(tiles)
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


# The audit already knows exactly what the page says - `reference_texts()` reads
# the PDF's own text layer on 130 of 133 pages - and until now it only ever used
# that to complain afterwards. Handing the same list to the model up front turns
# a post-hoc warning into a constraint it can act on, at no extra cost: the text
# is already in memory and the tokens are cheap next to the image.
#
# It is deliberately NOT a substitute for looking at the image. The list carries
# labels with every relationship destroyed, so it can say WHAT is on the page and
# never WHERE an arrow points - which is the entire reason this module exists.
# The instruction is therefore "place each of these", not "transcribe these".
CHECKLIST = """
CHECKLIST - {n} text items are printed on this image. This list comes from the
PDF's own text layer, so it is exact and complete; your reading of the image is
not. Place EVERY item somewhere in your output: as a node "label", as an edge
"condition", as an "escalation" condition, or as the "title"/"revision".

Work through the list item by item. An item drawn inside a box is a node. An item
sitting on or beside a line is that line's condition. An item that is genuinely
page furniture - a legend, a sheet number, a company name - may be left out, but
leave it out deliberately, not by overlooking it.

Do not paraphrase, re-spell or merge these items, and do not invent nodes or
arrows that are not drawn just to give an item somewhere to live. If you truly
cannot see where an item belongs, put it in a lane's nodes with kind "step"
rather than dropping it.

{items}
"""


def build_prompt(labels: list[str] | None) -> str:
    """The page prompt, plus the text-layer checklist when there is one."""
    if not labels:
        return PROMPT
    items = "\n".join(f"  - {l}" for l in labels)
    return PROMPT + "\n" + CHECKLIST.format(n=len(labels), items=items)


def _vision_options(opts: dict) -> dict:
    """Add the CPU pin to a vision request's options. See VISION_NUM_GPU."""
    if VISION_NUM_GPU is not None:
        opts = dict(opts, num_gpu=VISION_NUM_GPU)
    return opts


def describe(png: bytes, labels: list[str] | None = None,
             model: str = VISION_MODEL, host: str = OLLAMA_HOST) -> dict:
    """One tile -> the graph the VLM reads off it. Deterministic settings."""
    payload = {
        "model": model,
        "prompt": build_prompt(labels),
        "images": [base64.b64encode(png).decode()],
        "stream": False,
        "format": "json",
        "think": False,
        "options": _vision_options({"temperature": 0, "seed": 0,
                                    "num_ctx": 8192, "num_predict": -1}),
    }
    r = requests.post(f"{host}/api/generate", json=payload, timeout=1800)
    if r.status_code == 400:
        payload.pop("think")
        r = requests.post(f"{host}/api/generate", json=payload, timeout=1800)
    r.raise_for_status()
    return json.loads(r.json()["response"])


# ------------------------------------------------------------------- merge
def _norm(text) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def _slug(text) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", _norm(text)).strip("_")
    return s[:48] or "node"


def merge_graphs(graphs: list[dict]) -> dict:
    """Several tile graphs -> the one page graph they are pieces of.

    NODE IDENTITY IS THE LABEL, NOT THE ID. The model slugs ids per call, so the
    same drawn box arrives as `coo_approval` from one tile and `coo` from the
    tile that overlaps it. Merging on ids would keep both and split the flow in
    half at the seam; merging on the normalised label text - which is verbatim
    off the page, and now checklist-constrained to be exactly so - joins them.

    Edges are then rewritten into that shared namespace and deduplicated. Where
    the same arrow arrives labelled from the tile that could read its condition
    and unlabelled from the tile that only caught its tail, the labelled one
    wins: an unlabelled duplicate is what `audit()` flags as an invented copy,
    and here it is a known artefact of the seam rather than a model error.
    """
    if len(graphs) == 1:
        return graphs[0]

    merged = {"title": None, "revision": None, "lanes": [], "escalation": []}
    lanes_by_key: dict[str, dict] = {}
    esc_seen = set()

    for g in graphs:
        if not isinstance(g, dict):
            continue
        merged["title"] = merged["title"] or g.get("title")
        merged["revision"] = merged["revision"] or g.get("revision")

        for lane in g.get("lanes") or []:
            # Local id -> canonical id, built from this tile's own nodes.
            local = {}
            for n in lane.get("nodes") or []:
                local[n.get("id")] = _slug(n.get("label") or n.get("id"))

            key = _slug(lane.get("name") or merged["title"] or "workflow")
            tgt = lanes_by_key.get(key)
            if tgt is None:
                tgt = {"name": lane.get("name"), "nodes": [], "edges": []}
                lanes_by_key[key] = tgt
                merged["lanes"].append(tgt)

            have = {n["id"] for n in tgt["nodes"]}
            for n in lane.get("nodes") or []:
                cid = local[n.get("id")]
                if cid not in have:
                    have.add(cid)
                    tgt["nodes"].append({"id": cid, "label": n.get("label"),
                                         "kind": n.get("kind")})

            for e in lane.get("edges") or []:
                a = local.get(e.get("from")) or _slug(e.get("from"))
                b = local.get(e.get("to")) or _slug(e.get("to"))
                tgt["edges"].append({"from": a, "to": b,
                                     "condition": e.get("condition")})

        for e in g.get("escalation") or []:
            k = (_norm(e.get("condition")), _norm(e.get("approver")))
            if k not in esc_seen:
                esc_seen.add(k)
                merged["escalation"].append(e)

    for lane in merged["lanes"]:
        lane["edges"] = _dedupe_edges(lane["edges"])
    _drop_seam_fragments(merged)
    return merged


def _dedupe_edges(edges: list[dict]) -> list[dict]:
    """One entry per (from, to, condition), and drop a bare arrow that also
    exists with a condition - see merge_graphs for why that pair shows up."""
    by_pair: dict[tuple, list[dict]] = {}
    for e in edges:
        by_pair.setdefault((e["from"], e["to"]), []).append(e)

    out = []
    for _, group in by_pair.items():
        labelled = [e for e in group if _norm(e.get("condition"))]
        keep = labelled or group[:1]
        seen = set()
        for e in keep:
            c = _norm(e.get("condition"))
            if c not in seen:
                seen.add(c)
                out.append(e)
    return out


def _drop_seam_fragments(graph: dict) -> None:
    """Remove nodes that are a truncated reading of another node.

    Gutter-snapped cuts mean this should be rare - that is what they are for -
    but a box can still be clipped where no gutter was available and the tile
    was split on the ideal position instead. The signature is a node whose slug
    is a strict prefix of another node's in the same lane AND which nothing
    points at or away from: a real abbreviation would still have its edges.
    """
    for lane in graph.get("lanes") or []:
        ids = [n["id"] for n in lane.get("nodes") or []]
        used = {e[k] for e in lane.get("edges") or [] for k in ("from", "to")}
        drop = {i for i in ids if i not in used
                and any(o != i and o.startswith(i) for o in ids)}
        if drop:
            lane["nodes"] = [n for n in lane["nodes"] if n["id"] not in drop]


# ------------------------------------------------------------------ repair
def resolve_shared_lanes(graph: dict, title_hint: str | None = None) -> list[str]:
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

    # The name check used to be `lane name == graph title`, and it silently
    # stopped working when tiling arrived. A tiled page is merged from several
    # model calls, and this model returns `title: null` on pages whose heading it
    # cannot find - so `title` was "", nothing equalled it, and every pseudo-lane
    # survived. Measured on the Damietta RFQ diagram: six real flows plus one
    # lane named "Construction Company Approval Routing Flowchart", which is the
    # model describing the whole page rather than naming a flow.
    #
    # `title_hint` (the filename stem) is a SECOND name to match against, for a
    # page whose heading the model missed. It deliberately does not stand in for
    # the model's own title, because the two answer different questions: the
    # load-bearing change is that FULL ABSORPTION alone is sufficient when THE
    # MODEL reported no title, and a hint we supplied ourselves is no evidence
    # that it did. Conflating them re-blocks exactly the case this fixes.
    #
    # Absorption alone is safe, and the safety comes from what absorption means
    # rather than from any name: a lane every one of whose nodes was pulled into
    # sibling lanes has no content that is not already somewhere else. Dropping
    # it cannot lose anything - that is the whole premise of the pull above.
    model_title = _norm(graph.get("title"))
    names = {t for t in (model_title, _norm(title_hint)) if t}
    keep = []
    for lane in lanes:
        ids = {n.get("id") for n in lane.get("nodes") or []}
        fully_absorbed = bool(ids) and ids <= absorbed[id(lane)]
        looks_like_title = _norm(lane.get("name")) in names
        if fully_absorbed and (looks_like_title or not model_title):
            why = ("it was the shared approver column, now folded into the flows "
                   "that use it" if looks_like_title else
                   "every node in it also appears in a named flow, and the page "
                   "reported no title to identify it by")
            notes.append(f"dropped pseudo-lane {lane.get('name')!r} - {why}")
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
        "options": _vision_options({"temperature": 0, "seed": 0, "num_predict": -1}),
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
        resolve_shared_lanes(rec["graph"], Path(rec["filename"]).stem)
        # `reference_text` is the current field; `ocr_text` is what audit files
        # written before the native-text channel existed call it.
        ref = rec.get("reference_text") or rec.get("ocr_text") or ""
        warns = audit(rec["graph"], ref, exact_reference=rec.get("channel") == "native_text")
        chunks += chunks_from_graph(rec["graph"], rec["filename"], rec["page"],
                                    model, warns)
    return chunks


def to_chunks(pdf: Path, model: str = VISION_MODEL, host: str = OLLAMA_HOST,
              on_page=None, tile: bool = True) -> tuple[list[dict], list[dict]]:
    """Whole scanned PDF -> (chunks, per-page audit records).

    One chunk per LANE, not per file. `CS Signature Matrix.pdf` holds six
    distinct approval flows across three pages; as a single chunk, a question
    about steel would retrieve caravans and formwork too and leave the
    generator to guess which chain applies.
    """
    chunks, audits = [], []
    tiles_per_page = render_tiles(pdf, enabled=tile)

    # Reference text first: on a native-text diagram this is a free file read,
    # so a PDF that turns out to have no usable text costs nothing before the
    # expensive pass starts. It only needs the PNGs for the scanned fallback,
    # and a scanned page is always a single whole-page tile.
    refs, channel = reference_texts(pdf, [t[0]["png"] for t in tiles_per_page],
                                    host=host)
    exact = channel == "native_text"

    # One model at a time. Interleaving describe() and the OCR fallback would
    # swap a 17 GB model and a 2 GB model in and out of VRAM once per page.
    graphs = [merge_graphs([describe(t["png"], labels=t["labels"],
                                     model=model, host=host) for t in tiles])
              for tiles in tiles_per_page]

    for i, (graph, ref) in enumerate(zip(graphs, refs), start=1):
        # Repair first, then audit - so warnings describe the graph that
        # actually becomes chunks, not the raw model output.
        repairs = resolve_shared_lanes(graph, pdf.stem)
        warns = audit(graph, ref, exact_reference=exact)
        audits.append({"filename": pdf.name, "source": pdf.as_posix(), "page": i,
                       "title": graph.get("title"), "warnings": warns,
                       "repairs": repairs, "graph": graph,
                       "reference_text": ref, "channel": channel,
                       "tiles": len(tiles_per_page[i - 1])})
        if on_page:
            on_page(i, graph, warns, repairs, len(tiles_per_page[i - 1]))

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
    ap.add_argument("--no-tile", action="store_true",
                    help="send each page whole instead of splitting oversized "
                         "canvases into tiles - the pre-tiling behaviour, kept "
                         "so the change can be measured against it")
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

        def _log(i, graph, warns, repairs=(), ntiles=1):
            lanes = [l.get("name") for l in graph.get("lanes") or []]
            tiled = f"  [{ntiles} tiles]" if ntiles > 1 else ""
            print(f"  p{i}: {graph.get('title')!r}  lanes={lanes}{tiled}")
            for r in repairs:
                print(f"      repaired: {r}")
            for w in warns:
                print(f"      WARNING: {w}")

        try:
            _, a = to_chunks(p, model=args.model, on_page=_log,
                             tile=not args.no_tile)
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

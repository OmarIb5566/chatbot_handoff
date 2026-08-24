"""
Vector approval-flowcharts -> retrievable chunks, with no model in the loop.

WHY THIS EXISTS
---------------
`workflow_extractor.py` renders a flowchart to PNG and asks a vision model to
read it back as JSON. That is the right design for a scan, and it is what
`processes_pdf/other/CS Signature Matrix.pdf` needs - 3 pages of pure image, 0
characters.

It is the wrong design for `Workflows/`. Measured across all 110 files there:

    105/110   native text layer + vector drawings, ZERO raster images
      2/110   genuinely image-only (the Rowad Authority signature matrices)
      3/110   near-empty text layer
     66,095   extractable characters

These are Visio exports. Every label is addressable text at exact coordinates
and every box is a rectangle with a known bbox, so asking a model to guess at
them is not a hard problem solved badly - it is an easy problem made hard.

The cost of doing it the other way was measurable in the output:

    101/175 chunks (58%)  carried audit warnings
    119                   blank node labels, across 67 chunks
    500+                  warnings of the form "the page text has the amount
                          'N M' but it is absent from the graph"

That last one is the whole argument. The thresholds are what users ask these
diagrams about ("who approves bonds over 500 K"), the model kept dropping
them, and the correct value was sitting in the file as text the whole time.

WHAT REPLACES IT
----------------
    extract_shapes()   rects, diamonds, arrowheads, connector polylines
    assign_text()      span bbox centre inside a shape -> that shape's label
    cluster_loose()    leftover spans grouped into multi-line labels
    attach_labels()    each cluster -> the connector it sits nearest
    direct_edges()     arrowhead position decides which end is the head
    page_graph()       the same schema the VLM emitted, so nothing downstream
                       has to change

`render_prose`, `resolve_shared_lanes` and `main_sequence` are imported from
workflow_extractor unchanged. They were always deterministic; only the graph's
provenance changes.

COVERAGE IS THE CORRECTNESS ARGUMENT
------------------------------------
There is no eval budget for this and there does not need to be one. Every text
span on the page is classified as exactly one of: node label, decision label,
edge condition, lane header, title, or footer. Anything left over is counted
and reported per file as `coverage`.

That number is the deterministic replacement for "the amount is absent from the
graph", and unlike that warning it can be driven to 100%: an unclassified span
is a bug in this file, reproducible on a laptop with no GPU, not a model having
a bad day. A file below 100% names itself in the audit.

Usage:
    python workflow_vector.py                        # ../Workflows -> workflow_chunks.json
    python workflow_vector.py --src ../other_pdfs
    python workflow_vector.py --audit audit.json     # per-file coverage report
    python workflow_vector.py --explain FILE.pdf     # print one graph, chunk nothing
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import fitz

from workflow_extractor import MONEY_RE, render_prose, resolve_shared_lanes

from paths import (WORKFLOWS as DEFAULT_SRC, WORKFLOW_CHUNKS_JSON as DEFAULT_OUT,
                   WORKFLOW_CHUNKS_SCANNED)  # noqa: E402

# --- shape classification ---------------------------------------------------
MIN_NODE_W = 20.0        # smaller than this is an arrowhead or a tick, not a box
MIN_NODE_H = 15.0
ARROWHEAD_MAX = 25.0     # a filled non-rect under this is a head, not a diamond

# A decision diamond has to be big enough to hold its own label. Visio also
# draws small near-square filled polygons ON TOP of connectors as decoration -
# 44x45pt on the Bonds Request sheet - and those are not nodes: the connectors
# beneath them already run the full distance between the two real boxes, so
# admitting them split correct edges into two hops through a phantom. 70pt is
# above that decoration and below any diamond that could carry text at 12pt.
MIN_DIAMOND = 70.0

# How much of a span's width must lie inside a shape for the text to belong to
# it. Centre-containment alone is not enough: "Send for Approval" is ~90pt of
# text, so centring it over a 44pt decoration put its centre inside and stole
# the label off the arrow it actually belongs to.
TEXT_INSIDE = 0.6

# --- association tolerances -------------------------------------------------
# Both are in points, and both are deliberately tight. A label that attaches to
# the wrong connector is worse than one reported as unclassified: the first is
# a silent wrong answer, the second shows up in `coverage` and gets fixed.
SNAP_TOL = 30.0          # connector endpoint -> node box
LABEL_TOL = 70.0         # loose text cluster -> connector polyline

# Second pass for labels that reach no connector at LABEL_TOL. Some sheets
# park a condition well clear of its arrow ("Above 10M EGP" on the Subcontract
# Preparation sheets sits ~120pt off), and a monetary threshold left
# unattached is the one failure this module exists to prevent. Reaching
# further is only safe when the choice is not close, hence the margin: the
# nearest connector must be clearly nearer than the runner-up, or the label
# stays unattached and is reported.
LABEL_TOL_FAR = 170.0
LABEL_MARGIN = 0.6

# Vertical gap below which two loose spans are the same label. Visio wraps
# "Send back with / comments to / Clients & Trade / Manager" into four spans at
# ~13pt line pitch; anything under 1.8x the span height is the same phrase.
LINE_PITCH = 1.8

TITLE_BAND = 0.12        # top fraction of the page where a title may live
TITLE_SIZE_RATIO = 1.15  # ...and how much larger than body text it must be

_FOOTER_RE = re.compile(
    r"\b(rev(?:ision)?\.?\s*\d|page\s*\d|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|issue\s*date|effective\s*date)\b", re.I)


# =============================================================== shape reading
def _drawing_kind(d: dict) -> str:
    """One drawing -> what it represents on an approval flowchart.

    PyMuPDF hands back a display list, so this is a classification over paint
    operations rather than over anything semantic. Four outcomes matter:

      node       a rectangle big enough to hold a label
      arrowhead  a small filled shape that is not a rectangle
      diamond    a filled polygon big enough to hold a label - a decision
      junction   a filled polygon too small to hold one - a routing corner
      connector  anything whose items are lines or curves

    Order matters. A drawing carrying a rect is a node even if it also carries
    the lines of its own border, and a filled triangle has to be tested for
    size before it is mistaken for a decision.
    """
    items = d.get("items") or []
    rects = [it[1] for it in items if it[0] == "re"]
    for r in rects:
        r = fitz.Rect(r)
        if r.width >= MIN_NODE_W and r.height >= MIN_NODE_H:
            return "node"
    strokes = [it for it in items if it[0] in ("l", "c", "qu")]
    if d.get("fill") and not rects:
        box = fitz.Rect(d["rect"])
        if max(box.width, box.height) <= ARROWHEAD_MAX:
            return "arrowhead"
        if strokes and min(box.width, box.height) >= MIN_DIAMOND:
            return "diamond"
        if strokes:
            return "junction"
    return "connector" if strokes else "other"


def _polyline(d: dict) -> list[fitz.Point]:
    """A drawing's stroke items flattened to an ordered point list.

    Elbow connectors arrive as several segments inside ONE drawing, which is
    why the whole drawing is flattened rather than each segment taken alone:
    the first and last point of this list are the two ends of the connector,
    however many corners it turns on the way.
    """
    pts: list[fitz.Point] = []
    for it in d.get("items") or []:
        if it[0] == "l":
            pts.extend([fitz.Point(it[1]), fitz.Point(it[2])])
        elif it[0] == "c":                      # bezier: ends only
            pts.extend([fitz.Point(it[1]), fitz.Point(it[4])])
        elif it[0] == "qu":
            q = fitz.Quad(it[1])
            pts.extend([q.ul, q.lr])
    return pts


FRAGMENT_EPS = 1.5       # points within this of each other are the same corner


def merge_fragments(connectors: list[dict]) -> list[dict]:
    """Join connector fragments that share an endpoint into whole polylines.

    Visio does not always emit an elbow connector as one drawing. On the
    Subcontract Preparation sheets it emits each straight run separately, which
    is why that page yielded 387 "connectors" for 64 boxes: they are segments,
    not connections. Snapped individually almost none of them reach a box at
    both ends, so the edge is lost - and with it any threshold written beside
    it. "Below 500k SAR" was read correctly, attached to its 78pt stub, and
    then discarded because the stub joined a corner to nothing.

    Fragments are grouped by shared endpoint and each group is rebuilt into one
    polyline running between its two free ends. A group with more than two free
    ends is a genuine fan-out rather than one connection, and is left alone:
    guessing which arm pairs with which would invent routes.
    """
    def key(p: fitz.Point) -> tuple:
        return (round(p.x / FRAGMENT_EPS), round(p.y / FRAGMENT_EPS))

    parent = list(range(len(connectors)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    ends: dict[tuple, list[int]] = defaultdict(list)
    for i, c in enumerate(connectors):
        for p in (c["points"][0], c["points"][-1]):
            ends[key(p)].append(i)
    for members in ends.values():
        for other in members[1:]:
            union(members[0], other)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(connectors)):
        groups[find(i)].append(i)

    out = []
    for members in groups.values():
        if len(members) == 1:
            out.append(connectors[members[0]])
            continue
        degree: dict[tuple, int] = defaultdict(int)
        for i in members:
            for p in (connectors[i]["points"][0], connectors[i]["points"][-1]):
                degree[key(p)] += 1
        free = [k for k, n in degree.items() if n == 1]
        label = " ".join(dict.fromkeys(
            c for c in (connectors[i]["label"] for i in members) if c)).strip()
        if len(free) != 2:
            # A fan-out: keep the fragments, but let every arm carry the group's
            # text so a threshold beside the shared stem is not lost.
            for i in members:
                out.append({**connectors[i], "label": label})
            continue
        segs = [sg for i in members for sg in connectors[i]["segments"]]
        anchor = next(p for i in members for p in connectors[i]["points"]
                      if key(p) == free[0])
        far = next(p for i in members for p in connectors[i]["points"]
                   if key(p) == free[1])
        out.append({"points": [anchor, far], "segments": segs, "label": label})
    return out


def extract_shapes(page) -> dict:
    """Every drawing on the page, bucketed by what it is."""
    out: dict[str, list] = {"nodes": [], "diamonds": [], "arrowheads": [],
                            "junctions": [], "connectors": []}
    for d in page.get_drawings():
        kind = _drawing_kind(d)
        if kind == "node":
            for it in d.get("items") or []:
                if it[0] != "re":
                    continue
                r = fitz.Rect(it[1])
                if r.width >= MIN_NODE_W and r.height >= MIN_NODE_H:
                    out["nodes"].append({"rect": r, "fill": d.get("fill"),
                                         "kind": "step", "spans": []})
        elif kind == "diamond":
            out["diamonds"].append({"rect": fitz.Rect(d["rect"]),
                                    "fill": d.get("fill"),
                                    "kind": "decision", "spans": []})
        elif kind == "arrowhead":
            r = fitz.Rect(d["rect"])
            out["arrowheads"].append((r.tl + r.br) * 0.5)
        elif kind == "junction":
            out["junctions"].append({"rect": fitz.Rect(d["rect"]), "fill": d.get("fill"),
                                     "kind": "junction", "spans": []})
        elif kind == "connector":
            pts = _polyline(d)
            if len(pts) >= 2:
                # `segments` is what label distance is measured against;
                # `points` is only ever read for its two ends. Merged
                # fragments keep the distinction honest - their point list is
                # not in path order, so measuring along it would invent
                # diagonals that cross the page.
                segs = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
                out["connectors"].append({"points": pts, "segments": segs, "label": ""})
    return out


def page_spans(page) -> list[dict]:
    """Text spans in reading order, with their boxes."""
    spans = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            for s in line.get("spans", []):
                text = s["text"].strip()
                if not text:
                    continue
                r = fitz.Rect(s["bbox"])
                spans.append({"text": text, "rect": r, "size": round(s["size"], 1),
                              "centre": (r.tl + r.br) * 0.5, "role": None})
    spans.sort(key=lambda s: (round(s["rect"].y0, 1), s["rect"].x0))
    return spans


# ============================================================ text association
def assign_text(spans: list[dict], shapes: dict) -> list[dict]:
    """Spans whose centre falls inside a box belong to that box.

    Majority overlap, not centre-containment. A label that slightly overhangs
    its box - common where Visio autosizes text - still belongs to it, while a
    condition label centred over a small shape on its way past does not. See
    TEXT_INSIDE for the case that forced the distinction.

    Returns the spans that landed in nothing.
    """
    boxes = shapes["nodes"] + shapes["diamonds"]
    loose = []
    for s in spans:
        hit, best = None, TEXT_INSIDE
        for b in boxes:
            inter = fitz.Rect(s["rect"]) & b["rect"]
            frac = (inter.get_area() / s["rect"].get_area()) if s["rect"].get_area() else 0
            if frac >= best:
                hit, best = b, frac
        if hit is None:
            loose.append(s)
        else:
            hit["spans"].append(s)
            s["role"] = "node" if hit["kind"] == "step" else "decision"
    for b in boxes:
        b["spans"].sort(key=lambda s: (round(s["rect"].y0, 1), s["rect"].x0))
        b["label"] = " ".join(s["text"] for s in b["spans"]).strip()
    return loose


def cluster_loose(loose: list[dict]) -> list[dict]:
    """Group leftover spans into multi-line labels.

    A condition is one phrase wrapped over several spans, so clustering is on
    vertical adjacency plus horizontal overlap - the shape of wrapped text -
    rather than on raw distance, which would merge two labels sitting side by
    side on different arrows.
    """
    clusters: list[dict] = []
    for s in sorted(loose, key=lambda s: (round(s["rect"].y0, 1), s["rect"].x0)):
        placed = False
        for c in clusters:
            last = c["spans"][-1]
            gap = s["rect"].y0 - last["rect"].y1
            overlap = min(s["rect"].x1, last["rect"].x1) - max(s["rect"].x0, last["rect"].x0)
            if -last["rect"].height <= gap <= last["rect"].height * LINE_PITCH and overlap > 0:
                c["spans"].append(s)
                c["rect"] |= s["rect"]
                placed = True
                break
        if not placed:
            clusters.append({"spans": [s], "rect": fitz.Rect(s["rect"])})
    for c in clusters:
        c["text"] = " ".join(x["text"] for x in c["spans"]).strip()
        c["centre"] = (c["rect"].tl + c["rect"].br) * 0.5
        c["size"] = max(x["size"] for x in c["spans"])
    return clusters


def _point_segment_distance(p: fitz.Point, a: fitz.Point, b: fitz.Point) -> float:
    ax, ay, bx, by = a.x, a.y, b.x, b.y
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return abs(p.x - ax) + abs(p.y - ay)
    t = max(0.0, min(1.0, ((p.x - ax) * dx + (p.y - ay) * dy) / (dx * dx + dy * dy)))
    qx, qy = ax + t * dx, ay + t * dy
    return ((p.x - qx) ** 2 + (p.y - qy) ** 2) ** 0.5


def attach_labels(clusters: list[dict], connectors: list[dict]) -> list[dict]:
    """Each text cluster -> the connector it sits nearest, within LABEL_TOL.

    Returns the clusters that attached to nothing, for the caller to classify
    as title, lane header, footer, or - if it is none of those - to count
    against coverage.
    """
    def ranked(c):
        out = []
        for conn in connectors:
            d = min((_point_segment_distance(c["centre"], a, b)
                     for a, b in conn["segments"]), default=1e9)
            out.append((d, conn))
        out.sort(key=lambda x: x[0])
        return out

    def take(c, conn):
        conn["label"] = f"{conn['label']} {c['text']}".strip()
        for s in c["spans"]:
            s["role"] = "edge"

    unattached = []
    for c in clusters:
        near = ranked(c)
        if not near:
            unattached.append(c)
            continue
        best_d, best = near[0]
        if best_d <= LABEL_TOL:
            take(c, best)
            continue
        runner = near[1][0] if len(near) > 1 else 1e9
        if best_d <= LABEL_TOL_FAR and best_d <= runner * LABEL_MARGIN:
            take(c, best)
        else:
            unattached.append(c)
    return unattached


# ==================================================================== topology
def _box_distance(pt: fitz.Point, rect: fitz.Rect) -> float:
    return (abs(max(rect.x0 - pt.x, 0, pt.x - rect.x1))
            + abs(max(rect.y0 - pt.y, 0, pt.y - rect.y1)))


def direct_edges(connectors: list[dict], boxes: list[dict],
                 arrowheads: list[fitz.Point]) -> list[dict]:
    """Connectors -> directed edges, using the arrowhead to decide which way.

    Snapping alone gives an undirected pair; on a Customer Satisfaction page
    that produced `End -> COO`, which is backwards and reads as plausible
    prose. The arrowhead settles it: whichever endpoint is nearer a filled
    head is the destination. On that same page the margin was 4pt against
    128pt, so this is not a close call in practice.

    `head_margin` is kept per edge precisely because it is not always that
    clear. A small margin means the two ends were nearly equidistant from the
    nearest head, which is exactly the case a human should look at, and the
    audit lists them rather than quietly picking one.
    """
    edges = []
    for conn in connectors:
        pts = conn["points"]
        a, b = pts[0], pts[-1]
        ia = min(range(len(boxes)), key=lambda i: _box_distance(a, boxes[i]["rect"]),
                 default=None) if boxes else None
        ib = min(range(len(boxes)), key=lambda i: _box_distance(b, boxes[i]["rect"]),
                 default=None) if boxes else None
        if ia is None or ib is None:
            continue
        if _box_distance(a, boxes[ia]["rect"]) > SNAP_TOL: ia = None
        if _box_distance(b, boxes[ib]["rect"]) > SNAP_TOL: ib = None
        if ia is None or ib is None or ia == ib:
            continue
        da = min((abs(h.x - a.x) + abs(h.y - a.y) for h in arrowheads), default=1e9)
        db = min((abs(h.x - b.x) + abs(h.y - b.y) for h in arrowheads), default=1e9)
        src, dst = (ia, ib) if db <= da else (ib, ia)
        edges.append({"from": src, "to": dst,
                      "condition": conn["label"] or None,
                      "head_margin": round(abs(da - db), 1)})
    return edges


def contract_junctions(edges: list[dict], n_real: int) -> list[dict]:
    """Route edges through junction shapes and drop the junctions themselves.

    Visio breaks a connector at a routing corner, so `Commercial Manager
    Approval -> Clients & Trade Department` is drawn as two polylines meeting
    at a small filled shape. Treating that shape as a node invents a step the
    diagram does not have; ignoring it severs the two boxes it joins - which
    split the Bonds Request sheet into two disconnected flows, each rendering
    as its own chunk.

    So a junction is neither: every (a -> j) paired with every (j -> b) becomes
    (a -> b), carrying whichever condition was written on either half. Indices
    below `n_real` are real boxes; anything at or above it is a junction.

    ITERATIVE, because one hop is not enough. The Subcontract Preparation
    sheets route box -> j1 -> j2 -> box, and a single pass leaves the j1 -> j2
    hop with a junction at both ends, so it is discarded along with any
    threshold written on it - "Below 500k SAR" was attached to its connector,
    correctly, and then dropped with the edge. Contracting one junction at a
    time until none remain keeps the whole chain, and the pass count is bounded
    by the junctions themselves so a cycle cannot spin.
    """
    work = list(edges)
    for _ in range(len(edges) + 1):
        juncs = {i for e in work for i in (e["from"], e["to"]) if i >= n_real}
        if not juncs:
            break
        j = min(juncs)
        incoming = [e for e in work if e["to"] == j and e["from"] != j]
        outgoing = [e for e in work if e["from"] == j and e["to"] != j]
        rest = [e for e in work if j not in (e["from"], e["to"])]
        bridged = []
        for a in incoming:
            for b in outgoing:
                cond = " ".join(dict.fromkeys(
                    x for x in (a.get("condition"), b.get("condition")) if x))
                bridged.append({"from": a["from"], "to": b["to"],
                                "condition": cond or None,
                                "head_margin": max(a["head_margin"], b["head_margin"])})
        work = rest + bridged
    return [e for e in work if e["from"] < n_real and e["to"] < n_real]


def _dedupe(edges: list[dict]) -> list[dict]:
    """Collapse repeated (from, to) pairs.

    A connector drawn as two drawings - common where a line changes colour at
    a corner - snaps to the same pair twice. Conditions are unioned rather
    than dropped, so a threshold written beside one half of an elbow survives.
    """
    merged: dict[tuple, dict] = {}
    for e in edges:
        key = (e["from"], e["to"])
        if key not in merged:
            merged[key] = dict(e)
            continue
        cur = merged[key]
        if e["condition"] and e["condition"] not in (cur["condition"] or ""):
            cur["condition"] = " ".join(x for x in (cur["condition"], e["condition"]) if x)
        cur["head_margin"] = max(cur["head_margin"], e["head_margin"])
    return list(merged.values())


def _slug(text: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", (text or "node").lower()).strip("_")[:40] or "node"
    slug, n = base, 2
    while slug in taken:
        slug, n = f"{base}_{n}", n + 1
    taken.add(slug)
    return slug


def escalation_from(edges: list[dict], nodes: list[dict]) -> list[dict]:
    """Monetary thresholds, read off the edges that carry them.

    The VLM was asked for these as a separate list and kept omitting them.
    Here they are not extracted at all - they are a projection of edges whose
    condition contains an amount, so an escalation entry cannot go missing
    unless the edge itself did.
    """
    out = []
    by_id = {n["id"]: n for n in nodes}
    for e in edges:
        cond = e.get("condition") or ""
        if MONEY_RE.search(cond):
            approver = by_id.get(e["to"], {}).get("label")
            if approver:
                out.append({"condition": cond.strip(), "approver": approver})
    return out


# ====================================================================== lanes
def detect_lanes(page, boxes: list[dict], edges: list[dict],
                 title: str | None) -> list[dict]:
    """Split the page into swimlanes, or decline to.

    Swimlanes vary far more between templates than boxes do, and a wrong split
    is worse than no split: it strands edges pointing at nodes their lane does
    not contain, which is the exact breakage `resolve_shared_lanes` was written
    to repair. So this only splits on evidence - connected components of the
    graph, which is structural rather than a guess about layout - and
    otherwise returns one lane for the page.
    """
    if not boxes:
        return []
    adj = defaultdict(set)
    for e in edges:
        adj[e["from"]].add(e["to"])
        adj[e["to"]].add(e["from"])
    seen, groups = set(), []
    for i in range(len(boxes)):
        if i in seen:
            continue
        stack, comp = [i], []
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.append(cur)
            stack.extend(adj[cur] - seen)
        groups.append(sorted(comp))

    # A component of one node with no edges is a stray box (a legend, a key),
    # not a flow. Fold those into the largest component rather than emitting
    # single-node lanes that render as one-line chunks.
    real = [g for g in groups if len(g) > 1]
    if not real:
        real = [sorted(range(len(boxes)))]
    elif len(real) < len(groups):
        strays = [i for g in groups if len(g) == 1 for i in g]
        real[max(range(len(real)), key=lambda k: len(real[k]))].extend(strays)
        real = [sorted(g) for g in real]
    return real


# ================================================================== the graph
def page_graph(page, title_hint: str | None = None) -> tuple[dict, dict]:
    """One page -> (graph, coverage). No model call anywhere in here."""
    shapes = extract_shapes(page)
    shapes["connectors"] = merge_fragments(shapes["connectors"])
    spans = page_spans(page)
    loose = assign_text(spans, shapes)
    clusters = cluster_loose(loose)
    unattached = attach_labels(clusters, shapes["connectors"])

    # Junctions are snap targets but never nodes: connectors have to be able to
    # reach them, and contract_junctions then routes straight through.
    boxes = shapes["nodes"] + shapes["diamonds"]
    snap_targets = boxes + shapes["junctions"]
    raw_edges = _dedupe(contract_junctions(
        direct_edges(shapes["connectors"], snap_targets, shapes["arrowheads"]),
        n_real=len(boxes)))

    # --- classify what never reached a shape ---------------------------------
    body = statistics.median([s["size"] for s in spans]) if spans else 0
    title, revision = title_hint, None
    for c in unattached:
        top = c["rect"].y0 <= page.rect.height * TITLE_BAND
        if _FOOTER_RE.search(c["text"]):
            for s in c["spans"]:
                s["role"] = "footer"
            revision = revision or c["text"]
        elif top and c["size"] >= body * TITLE_SIZE_RATIO:
            for s in c["spans"]:
                s["role"] = "title"
            title = title or c["text"]
        elif top:
            for s in c["spans"]:
                s["role"] = "lane"

    # --- assemble ------------------------------------------------------------
    taken: set[str] = set()
    nodes = []
    for b in boxes:
        label = b.get("label") or ""
        nodes.append({"id": _slug(label, taken), "label": label, "kind": b["kind"]})

    # Lanes are grouped on the INDEX form of the edges, before ids are
    # substituted in - connected components are about which boxes touch, and
    # index arithmetic is what detect_lanes walks.
    lane_groups = detect_lanes(page, boxes, raw_edges, title)
    for e in raw_edges:
        e["from"] = nodes[e["from"]]["id"]
        e["to"] = nodes[e["to"]]["id"]
    lanes = []
    for n, group in enumerate(lane_groups, start=1):
        ids = {nodes[i]["id"] for i in group}
        lanes.append({
            # Not the document title: render_prose heads the chunk with
            # "title - section", so naming the only lane after the document
            # printed the name twice.
            "name": "Approval flow" if len(lane_groups) == 1 else f"Approval flow {n}",
            "nodes": [nodes[i] for i in group],
            "edges": [e for e in raw_edges if e["from"] in ids and e["to"] in ids],
        })

    graph = {"title": title, "revision": revision, "lanes": lanes,
             "escalation": escalation_from(raw_edges, nodes)}

    # Every amount printed on the page must survive into the graph. Once
    # coverage is 100% this is true by construction - the amount is in a node
    # label or an edge condition because every span is - so it is asserted
    # rather than hoped for, and a regression in clustering shows up here
    # instead of as a quietly wrong answer about who signs.
    # Per string, never over a joined blob: concatenating spans invents amounts
    # that are not on the page - a stray "2" beside a currency word reads as
    # "2 EGP" across the join and then reports itself as lost. The check has to
    # be at least as trustworthy as the thing it is checking.
    def _amounts(strings: list[str]) -> set[str]:
        return {m.group(0).replace(" ", "").upper()
                for t in strings for m in MONEY_RE.finditer(t or "")}

    on_page = _amounts([s["text"] for s in spans])
    in_graph = _amounts([n["label"] for n in nodes]
                        + [e.get("condition") for e in raw_edges])

    roles = [s["role"] for s in spans]
    total = len(roles)
    classified = sum(1 for r in roles if r)
    coverage = {
        "spans": total,
        "classified": classified,
        "coverage": round(classified / total, 4) if total else 1.0,
        "unclassified_text": [s["text"] for s in spans if not s["role"]][:12],
        "nodes": len(nodes), "edges": len(raw_edges), "lanes": len(lanes),
        "blank_labels": sum(1 for n in nodes if not n["label"]),
        "low_margin_edges": [f'{e["from"]}->{e["to"]}' for e in raw_edges
                             if e["head_margin"] < 5],
        "lost_amounts": sorted(on_page - in_graph),
    }
    return graph, coverage


# =================================================================== chunking
def audit_warnings(graph: dict, cov: dict) -> list[str]:
    """Deterministic checks. Every one of these is free and reproducible."""
    warns = []
    if cov["coverage"] < 1.0:
        warns.append(f"{cov['spans'] - cov['classified']} of {cov['spans']} text spans "
                     f"unclassified: {cov['unclassified_text']}")
    if cov["blank_labels"]:
        warns.append(f"{cov['blank_labels']} node(s) carry no label")
    if cov["low_margin_edges"]:
        warns.append(f"direction uncertain (arrowhead margin < 5pt) on: "
                     f"{cov['low_margin_edges']}")
    if cov["lost_amounts"]:
        warns.append(f"amount(s) printed on the page but absent from the graph: "
                     f"{cov['lost_amounts']}")
    return warns


def to_chunks(pdf: Path) -> tuple[list[dict], dict]:
    """One PDF -> chunks + its audit record."""
    chunks, pages = [], []
    with fitz.open(pdf) as doc:
        for page_no, page in enumerate(doc, start=1):
            graph, cov = page_graph(page, title_hint=pdf.stem)
            notes = resolve_shared_lanes(graph, title_hint=pdf.stem)
            warns = audit_warnings(graph, cov)
            for lane in graph.get("lanes") or []:
                chunks.append({
                    "filename": pdf.name,
                    "doc_code": None,
                    "title": graph.get("title") or pdf.stem,
                    "section": lane.get("name") or "WORKFLOW",
                    "step": None,
                    "text": render_prose(graph, lane),
                    "raw_heading": graph.get("title"),
                    "label_method": "geometry",
                    "label_score": 1.0,
                    "page": page_no,
                    "detector": "vector",
                    # Whitelist-eligible, unlike the VLM path: every token here
                    # is read from the PDF's own text layer, so a form number
                    # in a workflow cannot have been invented. The prose around
                    # it is rendered from the graph, so the SENTENCES are not
                    # verbatim even though the TOKENS are - which is why this
                    # is its own source_type rather than reusing the process
                    # chunks' provenance.
                    "source_type": "pdf_vector",
                    "vision_model": None,
                    "review": "machine",
                    "audit_warnings": warns,
                    "coverage": cov["coverage"],
                    "graph": lane,
                })
            pages.append({"page": page_no, **cov, "warnings": warns,
                          "shared_lane_notes": notes})
    return chunks, {"filename": pdf.name, "pages": pages}


def main() -> None:
    from translate import enable_utf8_stdout

    enable_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--audit", type=Path, default=None)
    ap.add_argument("--explain", type=Path, default=None)
    # The two Rowad Authority signature matrices have no text layer at all, so
    # this module reads nothing off them and they would simply vanish from the
    # corpus. workflow_extractor.py still covers them; merging here rather than
    # by hand means regenerating cannot quietly drop them again.
    ap.add_argument("--merge", type=Path, default=WORKFLOW_CHUNKS_SCANNED,
                    help="chunks from the scanned-only VLM path to append "
                         "(skipped silently if the file does not exist)")
    args = ap.parse_args()

    if args.explain:
        with fitz.open(args.explain) as doc:
            for n, page in enumerate(doc, start=1):
                g, cov = page_graph(page, title_hint=args.explain.stem)
                print(f"\n=== {args.explain.name} page {n} "
                      f"(coverage {100*cov['coverage']:.0f}%) ===")
                for lane in g["lanes"]:
                    print(f"  lane: {lane['name']}")
                    for nd in lane["nodes"]:
                        print(f"    [{nd['kind'][:4]}] {nd['label'][:60]!r}")
                    for e in lane["edges"]:
                        print(f"    {e['from'][:24]:<24} -> {e['to'][:24]:<24} "
                              f"{(e.get('condition') or '')[:34]!r}")
                if g["escalation"]:
                    print(f"  escalation: {g['escalation']}")
                if cov["unclassified_text"]:
                    print(f"  UNCLASSIFIED: {cov['unclassified_text']}")
        return

    pdfs = sorted(args.src.glob("*.pdf"))
    all_chunks, audits = [], []
    for pdf in pdfs:
        try:
            chunks, rec = to_chunks(pdf)
        except Exception as exc:                      # a bad file must not stop the run
            audits.append({"filename": pdf.name, "error": str(exc)})
            print(f"  !! {pdf.name}: {exc}")
            continue
        all_chunks += chunks
        audits.append(rec)

    merged = 0
    if args.merge and args.merge.exists():
        scanned = json.load(open(args.merge, encoding="utf-8"))
        # Keyed by filename: a document this run extracted geometrically does
        # not also want the vision model's reading of it.
        mine = {c["filename"] for c in all_chunks}
        extra = [c for c in scanned if c["filename"] not in mine]
        all_chunks += extra
        merged = len(extra)

    args.out.write_text(json.dumps(all_chunks, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    if args.audit:
        args.audit.write_text(json.dumps(audits, indent=2, ensure_ascii=False),
                              encoding="utf-8")

    spans = sum(p["spans"] for a in audits for p in a.get("pages", []))
    classified = sum(p["classified"] for a in audits for p in a.get("pages", []))
    perfect = sum(1 for a in audits
                  if a.get("pages") and all(p["coverage"] >= 1.0 for p in a["pages"]))
    print(f"\n{len(pdfs)} PDFs -> {len(all_chunks)} chunks"
          + (f" ({merged} merged from {args.merge.name}, scanned pages the "
             f"vision model read)" if merged else ""))
    print(f"  text spans classified : {classified}/{spans} "
          f"({100*classified/spans:.1f}%)" if spans else "  no spans")
    print(f"  files at 100% coverage: {perfect}/{len(audits)}")
    worst = sorted(((min((p['coverage'] for p in a.get('pages', [])), default=1.0), a['filename'])
                    for a in audits if a.get("pages")))[:10]
    print("  lowest coverage:")
    for c, f in worst:
        print(f"    {100*c:5.1f}%  {f[:66]}")
    print(f"\nWrote {args.out.name}")


if __name__ == "__main__":
    main()

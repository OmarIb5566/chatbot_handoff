"""Two questions, for the 18 files with a disconnected chunk:
  1) Does re-extraction with today's code reproduce today's output? (is a
     re-extract a no-op?)
  2) For each orphan node, is there a connector endpoint NEAR it that failed
     the SNAP_TOL=30 test - or is there no connector near it at all? The first
     is a tunable threshold; the second means the arrow is not in the vector
     layer and no amount of re-extraction recovers it.
Read-only: writes nothing but its own report."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from translate import enable_utf8_stdout; enable_utf8_stdout()
import fitz
import workflow_vector as V
from workflow_vector import (extract_shapes, merge_fragments, page_spans, assign_text,
                             cluster_loose, attach_labels, _box_distance, to_chunks)

FILES = json.load(open(ROOT/"_verify"/"disconnected_files.json", encoding="utf-8"))
STORED = json.load(open(ROOT/"data"/"workflow_chunks.json", encoding="utf-8"))
by_file = {}
for c in STORED:
    by_file.setdefault(c["filename"], []).append(c)

print("=" * 96)
print("1) IS RE-EXTRACTION A NO-OP?  (re-run today's code, compare to today's chunks)")
print("=" * 96)
same = diff = err = 0
for fn in FILES:
    pdf = ROOT / "Workflows" / fn
    if not pdf.exists():
        print(f"  MISSING PDF  {fn}"); err += 1; continue
    try:
        chunks, _ = to_chunks(pdf)
    except Exception as e:
        print(f"  ERROR  {fn[:60]}: {e!r}"); err += 1; continue
    old = by_file.get(fn, [])
    ok = (len(chunks) == len(old)
          and all((a.get("text") or "").strip() == (b.get("text") or "").strip()
                  for a, b in zip(chunks, old)))
    same += ok; diff += (not ok)
    if not ok:
        print(f"  DIFFERS  {fn[:66]}  chunks {len(old)} -> {len(chunks)}")
print(f"\n  identical: {same}   differs: {diff}   errors: {err}")

print("\n" + "=" * 96)
print("2) WHY IS THE ORPHAN ORPHANED?  nearest connector endpoint to each orphan box")
print(f"   (SNAP_TOL = {V.SNAP_TOL})")
print("=" * 96)
summary = {"tunable": 0, "no_connector": 0, "orphans": 0}
for fn in FILES:
    pdf = ROOT / "Workflows" / fn
    if not pdf.exists():
        continue
    orphan_labels = set()
    for c in by_file.get(fn, []):
        g = c.get("graph") or {}
        lab = {n["id"]: (n.get("label") or "").strip() for n in g.get("nodes") or []}
        touched = {e["from"] for e in g.get("edges") or []} | {e["to"] for e in g.get("edges") or []}
        orphan_labels |= {lab[i] for i in lab if i not in touched}
    if not orphan_labels:
        continue
    print(f"\n{fn}")
    with fitz.open(pdf) as doc:
        for page in doc:
            shapes = extract_shapes(page)
            shapes["connectors"] = merge_fragments(shapes["connectors"])
            # labels are attached to boxes here, not by extract_shapes
            spans = page_spans(page)
            assign_text(spans, shapes)
            boxes = shapes["nodes"] + shapes["diamonds"]
            targets = boxes + shapes["junctions"]
            print(f"   page {page.number+1}: {len(boxes)} boxes, "
                  f"{len(shapes['connectors'])} connectors, "
                  f"{len(shapes['arrowheads'])} arrowheads, "
                  f"{len(shapes['junctions'])} junctions")
            for b in boxes:
                lbl = (b.get("label") or "").strip()
                if lbl not in orphan_labels:
                    continue
                best = []
                for conn in shapes["connectors"]:
                    for end in (conn["points"][0], conn["points"][-1]):
                        best.append(_box_distance(end, b["rect"]))
                best.sort()
                near = best[:3]
                summary["orphans"] += 1
                if near and near[0] <= 120:
                    verdict = f"connector endpoint {near[0]:.1f}pt away -> TUNABLE (SNAP_TOL {V.SNAP_TOL})"
                    summary["tunable"] += 1
                else:
                    verdict = ("NO connector endpoint within 120pt -> arrow absent from "
                               "the vector layer")
                    summary["no_connector"] += 1
                print(f"      orphan {lbl[:44]!r:46} nearest ends {[round(x,1) for x in near]}  {verdict}")
print("\n" + "=" * 96)
print(f"orphan boxes examined: {summary['orphans']}   "
      f"tunable (a connector is near): {summary['tunable']}   "
      f"arrow absent: {summary['no_connector']}")

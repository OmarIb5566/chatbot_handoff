"""Where exactly is the End edge lost? Trace one connector that touches an orphan."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from translate import enable_utf8_stdout; enable_utf8_stdout()
import fitz, workflow_vector as V
from workflow_vector import (extract_shapes, merge_fragments, page_spans, assign_text,
                             cluster_loose, attach_labels, direct_edges,
                             contract_junctions, _dedupe, _box_distance, detect_lanes)
FILES = ["Internal Head Office NCR - New Internal Head Office NCR.pdf",
         "Potential Delay Events - NOD Auto Workflow.pdf",
         "Request For Information (RFI) - RME to Client V2.pdf",
         "Infrastructure Inspection Request (IIR) - Holding Tanks WF.pdf"]
for fn in FILES:
    pdf = ROOT/"Workflows"/fn
    print("=" * 94); print(fn)
    with fitz.open(pdf) as doc:
        page = doc[0]
        shapes = extract_shapes(page)
        shapes["connectors"] = merge_fragments(shapes["connectors"])
        spans = page_spans(page)
        loose = assign_text(spans, shapes)
        clusters = cluster_loose(loose)
        boxes = shapes["nodes"] + shapes["diamonds"]
        targets = boxes + shapes["junctions"]
        labels = [(b.get("label") or "").strip() for b in boxes]
        try:
            oi = next(i for i, l in enumerate(labels) if l.lower() == "end")
        except StopIteration:
            print(f"   no 'End' box; labels seen: {[l[:24] for l in labels]}"); continue
        print(f"   boxes={len(boxes)} junctions={len(shapes['junctions'])} "
              f"connectors={len(shapes['connectors'])}")
        print(f"   'End' box rect: {boxes[oi]['rect']}")
        for ci, conn in enumerate(shapes["connectors"]):
            a, b = conn["points"][0], conn["points"][-1]
            da, db = _box_distance(a, boxes[oi]["rect"]), _box_distance(b, boxes[oi]["rect"])
            if min(da, db) > V.SNAP_TOL:
                continue
            def nearest(pt):
                ds = sorted((_box_distance(pt, t["rect"]), i) for i, t in enumerate(targets))
                return ds[0]
            na, nb = nearest(a), nearest(b)
            def name(i):
                return (targets[i].get("label") or f"<{targets[i]['kind']}>")[:34]
            print(f"   connector #{ci}: {len(conn['points'])} pts, label={conn['label']!r}")
            print(f"      end A -> nearest {name(na[1])!r} at {na[0]:.1f}pt"
                  f"{'  (BEYOND SNAP_TOL)' if na[0] > V.SNAP_TOL else ''}")
            print(f"      end B -> nearest {name(nb[1])!r} at {nb[0]:.1f}pt"
                  f"{'  (BEYOND SNAP_TOL)' if nb[0] > V.SNAP_TOL else ''}")
            if na[1] == nb[1]:
                print("      -> BOTH ENDS SNAP TO THE SAME SHAPE: direct_edges drops it (ia == ib)")
        raw = direct_edges(shapes["connectors"], targets, shapes["arrowheads"])
        print(f"   direct_edges produced {len(raw)} edges; "
              f"touching End: {sum(1 for e in raw if oi in (e['from'], e['to']))}")
        con = contract_junctions(raw, n_real=len(boxes))
        print(f"   after contract_junctions: {len(con)}; "
              f"touching End: {sum(1 for e in con if oi in (e['from'], e['to']))}")
        ded = _dedupe(con)
        print(f"   after _dedupe: {len(ded)}; "
              f"touching End: {sum(1 for e in ded if oi in (e['from'], e['to']))}")
        groups = detect_lanes(page, boxes, ded, None)
        print(f"   detect_lanes -> {len(groups)} group(s); "
              f"End in group: {[gi for gi,g in enumerate(groups) if oi in g]}")

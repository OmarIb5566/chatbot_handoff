"""Dump the full geometry of one small failing page so the End arrow is visible."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from translate import enable_utf8_stdout; enable_utf8_stdout()
import fitz, workflow_vector as V
from workflow_vector import (extract_shapes, merge_fragments, page_spans,
                             assign_text, _box_distance)
for fn in ("Potential Delay Events - NOD Auto Workflow.pdf",
           "Internal Head Office NCR - New Internal Head Office NCR.pdf",
           "Infrastructure Inspection Request (IIR) - Holding Tanks WF.pdf"):
    print("=" * 96); print(fn)
    with fitz.open(ROOT/"Workflows"/fn) as doc:
        page = doc[0]
        raw = extract_shapes(page)
        n_raw = len(raw["connectors"])
        shapes = dict(raw); shapes["connectors"] = merge_fragments(raw["connectors"])
        spans = page_spans(page); assign_text(spans, shapes)
        boxes = shapes["nodes"] + shapes["diamonds"]
        print(f"  raw connectors {n_raw} -> merged {len(shapes['connectors'])}, "
              f"boxes {len(boxes)}, junctions {len(shapes['junctions'])}, "
              f"arrowheads {len(shapes['arrowheads'])}")
        print("  BOXES:")
        for i, b in enumerate(boxes):
            r = b["rect"]
            print(f"    [{i}] {(b.get('label') or '')[:38]:40} "
                  f"({r.x0:.0f},{r.y0:.0f})-({r.x1:.0f},{r.y1:.0f}) kind={b['kind']}")
        print("  MERGED CONNECTORS:")
        for i, c in enumerate(shapes["connectors"]):
            a, z = c["points"][0], c["points"][-1]
            def near(p):
                d = sorted(((_box_distance(p, b["rect"]), j) for j, b in enumerate(boxes)))
                return f"box{d[0][1]}@{d[0][0]:.0f}pt" if d else "-"
            print(f"    #{i:2} ({a.x:.0f},{a.y:.0f})->({z.x:.0f},{z.y:.0f})  "
                  f"segs={len(c['segments'])} A:{near(a)} B:{near(z)} label={c['label']!r}")
        print("  ARROWHEADS:")
        for h in shapes["arrowheads"]:
            d = sorted(((_box_distance(h, b["rect"]), j) for j, b in enumerate(boxes)))
            print(f"    ({h.x:.0f},{h.y:.0f}) nearest box{d[0][1]}@{d[0][0]:.0f}pt")

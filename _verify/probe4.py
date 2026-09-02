import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from translate import enable_utf8_stdout; enable_utf8_stdout()
import fitz, workflow_vector as V
from workflow_vector import extract_shapes, merge_fragments, page_spans, assign_text, _box_distance
fn = "Potential Delay Events - NOD Auto Workflow.pdf"
with fitz.open(ROOT/"Workflows"/fn) as doc:
    page = doc[0]
    raw = extract_shapes(page)
    print(f"RAW connectors ({len(raw['connectors'])}):")
    for i, c in enumerate(raw["connectors"]):
        pts = c["points"]
        print(f"  #{i:2} {len(pts)}pts  " +
              " -> ".join(f"({p.x:.0f},{p.y:.0f})" for p in pts[:6]) +
              ("..." if len(pts) > 6 else "") + f"   label={c['label']!r}")
    merged = merge_fragments([dict(c) for c in raw["connectors"]])
    print(f"\nMERGED ({len(merged)}):")
    for i, c in enumerate(merged):
        a, z = c["points"][0], c["points"][-1]
        print(f"  #{i} ({a.x:.0f},{a.y:.0f})->({z.x:.0f},{z.y:.0f}) segs={len(c['segments'])}")
    # which raw fragments ended up in the bad group?
    print("\nBoxes:")
    shapes = dict(raw); shapes["connectors"] = merged
    spans = page_spans(page); assign_text(spans, shapes)
    for i, b in enumerate(shapes["nodes"] + shapes["diamonds"]):
        r = b["rect"]; print(f"  [{i}] {(b.get('label') or '')[:30]:32} ({r.x0:.0f},{r.y0:.0f})-({r.x1:.0f},{r.y1:.0f})")

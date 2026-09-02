"""GATE: can render_prose() reproduce the stored text byte-for-byte?
If not, a repair pass cannot safely re-render, and I need to find what else
post-processes the text before doing anything to the corpus."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from workflow_extractor import render_prose
W = json.load(open(ROOT/"data"/"workflow_chunks.json", encoding="utf-8"))
same = diff = err = 0
examples = []
for c in W:
    lane = c.get("graph")
    if not lane:
        continue
    try:
        out = render_prose({"title": c.get("title")}, lane)
    except Exception as e:
        err += 1
        if len(examples) < 3: examples.append((c["filename"], c.get("section"), repr(e)))
        continue
    if out.strip() == (c.get("text") or "").strip():
        same += 1
    else:
        diff += 1
        if len(examples) < 3:
            examples.append((c["filename"], c.get("section"), "TEXT DIFFERS"))
print(f"chunks: {len(W)}   reproduced exactly: {same}   differ: {diff}   errors: {err}")
for fn, sec, why in examples:
    print(f"   {why}: {fn[:52]} | {sec}")
if diff:
    for c in W:
        lane = c.get("graph")
        if not lane: continue
        out = render_prose({"title": c.get("title")}, lane)
        if out.strip() != (c.get("text") or "").strip():
            import difflib
            a = (c.get("text") or "").strip().split("\n")
            b = out.strip().split("\n")
            print(f"\nfirst difference — {c['filename']} | {c.get('section')}")
            for l in list(difflib.unified_diff(a, b, "stored", "re-rendered", lineterm=""))[:25]:
                print("   " + l)
            break

"""Final corpus = re-extracted vector chunks (merge_fragments fix) + the 3
vision chunks the vector path never produces + label-collapse repair, text
re-rendered with render_prose. Writes data/workflow_chunks_fixed.json."""
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"backend"))
from translate import enable_utf8_stdout; enable_utf8_stdout()
from workflow_extractor import render_prose
OLD = json.load(open(ROOT/"data"/"workflow_chunks.json", encoding="utf-8"))
NEW = json.load(open(ROOT/"data"/"workflow_chunks_reextracted.json", encoding="utf-8"))
vision = [c for c in OLD if c.get("source_type") == "vlm_description"]
print(f"re-extracted vector chunks {len(NEW)}; vision chunks carried over {len(vision)}")
TAIL_RE = re.compile(r"(\n+)(Approval thresholds:|Thresholds recorded on this page)")
def split_tail(t):
    m = TAIL_RE.search(t or "")
    return (t, "", "") if not m else (t[:m.start(1)], m.group(1), t[m.start(2):])
def collapse(lane):
    nodes, edges = lane.get("nodes") or [], lane.get("edges") or []
    lab = {n["id"]: (n.get("label") or "").strip() for n in nodes}
    key = {i: (re.sub(r"[^a-z0-9]+", "_", L.lower()).strip("_") or i) for i, L in lab.items()}
    nn, seen = [], set()
    for n in nodes:
        k = key[n["id"]]
        if k in seen: continue
        seen.add(k); m = dict(n); m["id"] = k; nn.append(m)
    ne, sig, ds, dd = [], set(), 0, 0
    for e in edges:
        a, b = key[e["from"]], key[e["to"]]
        if a == b: ds += 1; continue
        s = (a, b, (e.get("condition") or "").strip())
        if s in sig: dd += 1; continue
        sig.add(s); m = dict(e); m["from"], m["to"] = a, b; ne.append(m)
    o = dict(lane); o["nodes"], o["edges"] = nn, ne
    return o, ds, dd
bad = 0
for c in NEW:
    lane = c.get("graph")
    if not lane: continue
    stored = c.get("text") or ""
    _, sep, tail = split_tail(stored)
    if render_prose({"title": c.get("title"), "escalation": []}, lane) + sep + tail != stored:
        bad += 1
print(f"GATE splice fidelity: {len(NEW)-bad}/{len(NEW)} exact")
if bad:
    print("GATE FAILED - nothing written."); sys.exit(1)
out, changed, tds, tdd = [], 0, 0, 0
for c in NEW:
    lane = c.get("graph")
    if not lane: out.append(c); continue
    nl, ds, dd = collapse(lane); tds += ds; tdd += dd
    n = dict(c)
    if len(nl["nodes"]) != len(lane["nodes"]) or len(nl["edges"]) != len(lane["edges"]):
        changed += 1
        _, sep, tail = split_tail(c.get("text") or "")
        n["text"] = render_prose({"title": c.get("title"), "escalation": []}, nl) + sep + tail
        n["graph"] = nl
        n["audit_warnings"] = list(c.get("audit_warnings") or []) + [
            f"repaired: merged {len(lane['nodes'])-len(nl['nodes'])} duplicate-label node(s), "
            f"dropped {ds} self-edge(s) and {dd} duplicate edge(s)"]
    out.append(n)
out += vision
(ROOT/"data"/"workflow_chunks_fixed.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nwrote data/workflow_chunks_fixed.json  ({len(out)} chunks)")
print(f"  collapse changed {changed}; self-edges {tds}, duplicate edges {tdd}")

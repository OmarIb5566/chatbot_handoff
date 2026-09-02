"""Build data/workflow_chunks_repaired.json. Writes a NEW file; never touches
data/workflow_chunks.json.

Repair = collapse nodes that share a label, drop the self-edges that creates,
dedupe edges. Text is re-rendered with the extractor's own render_prose() so
the house style is identical, EXCEPT the threshold tail, which is spliced from
the original verbatim: thresholds live on the page-level `escalation` list that
chunks do not store, and they key off condition strings rather than node ids,
so collapsing nodes cannot change them.

GATE: with the lane left UNMODIFIED, head-re-render + spliced tail must equal
the stored text byte for byte, on every chunk. If that fails anywhere the
splice is not faithful and nothing is written.
"""
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from translate import enable_utf8_stdout
enable_utf8_stdout()
from workflow_extractor import render_prose

SRC = ROOT / "data" / "workflow_chunks.json"
DST = ROOT / "data" / "workflow_chunks_repaired.json"
TAIL_RE = re.compile(r"(\n+)(Approval thresholds:|Thresholds recorded on this page)")

def split_tail(text):
    """Return (head, separator, tail). The separator is preserved verbatim so
    the rebuild is byte-identical, not merely equivalent."""
    m = TAIL_RE.search(text)
    if not m:
        return text, "", ""
    return text[:m.start(1)], m.group(1), text[m.start(2):]

def head_for(chunk, lane):
    # escalation deliberately empty: render_prose then emits no threshold tail,
    # so what comes back is exactly the topology head.
    return render_prose({"title": chunk.get("title"), "escalation": []}, lane)

def collapse(lane):
    nodes = lane.get("nodes") or []
    edges = lane.get("edges") or []
    lab = {n["id"]: (n.get("label") or "").strip() for n in nodes}
    key = {}
    for n in nodes:
        L = lab[n["id"]]
        key[n["id"]] = re.sub(r"[^a-z0-9]+", "_", L.lower()).strip("_") or n["id"]
    newnodes, seen = [], set()
    for n in nodes:
        k = key[n["id"]]
        if k in seen:
            continue
        seen.add(k)
        m = dict(n); m["id"] = k
        newnodes.append(m)
    newedges, sigs, dropped_self, dropped_dupe = [], set(), 0, 0
    for e in edges:
        a, b = key[e["from"]], key[e["to"]]
        if a == b:
            dropped_self += 1
            continue
        sig = (a, b, (e.get("condition") or "").strip())
        if sig in sigs:
            dropped_dupe += 1
            continue
        sigs.add(sig)
        m = dict(e); m["from"], m["to"] = a, b
        newedges.append(m)
    out = dict(lane); out["nodes"], out["edges"] = newnodes, newedges
    return out, dropped_self, dropped_dupe

W = json.load(open(SRC, encoding="utf-8"))

# ---- GATE -----------------------------------------------------------------
bad = []
for c in W:
    lane = c.get("graph")
    if not lane:
        continue
    stored = c.get("text") or ""
    _, sep, tail = split_tail(stored)
    rebuilt = head_for(c, lane) + sep + tail
    if rebuilt != stored:
        bad.append(c)
print(f"GATE  splice fidelity on UNMODIFIED lanes: "
      f"{len(W) - len(bad)}/{len(W)} exact, {len(bad)} mismatched")
if bad:
    import difflib
    c = bad[0]
    stored = c["text"]; _, sep, tail = split_tail(stored)
    rebuilt = head_for(c, c["graph"]) + sep + tail
    print(f"\nfirst mismatch: {c['filename']} | {c.get('section')}")
    for l in list(difflib.unified_diff(stored.split("\n"), rebuilt.split("\n"),
                                       "stored", "rebuilt", lineterm=""))[:30]:
        print("   " + l)
    print("\nGATE FAILED - nothing written.")
    sys.exit(1)

# ---- REPAIR ---------------------------------------------------------------
out, stats = [], {"chunks": 0, "changed": 0, "self": 0, "dupe": 0,
                  "nodes_before": 0, "nodes_after": 0,
                  "edges_before": 0, "edges_after": 0}
changed_files = []
for c in W:
    lane = c.get("graph")
    if not lane:
        out.append(c); continue
    stats["chunks"] += 1
    stats["nodes_before"] += len(lane.get("nodes") or [])
    stats["edges_before"] += len(lane.get("edges") or [])
    new_lane, ds, dd = collapse(lane)
    stats["self"] += ds; stats["dupe"] += dd
    stats["nodes_after"] += len(new_lane["nodes"])
    stats["edges_after"] += len(new_lane["edges"])
    n = dict(c)
    if (len(new_lane["nodes"]) != len(lane.get("nodes") or [])
            or len(new_lane["edges"]) != len(lane.get("edges") or [])):
        stats["changed"] += 1
        changed_files.append(f"{c['filename']} | {c.get('section')}")
        _, sep, tail = split_tail(c.get("text") or "")
        n["text"] = head_for(c, new_lane) + sep + tail
        n["graph"] = new_lane
        n["audit_warnings"] = list(c.get("audit_warnings") or []) + [
            f"repaired: merged {len(lane['nodes']) - len(new_lane['nodes'])} duplicate-label "
            f"node(s), dropped {ds} self-edge(s) and {dd} duplicate edge(s)"]
    out.append(n)
DST.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nwrote {DST.name}  ({len(out)} chunks)")
print(f"  chunks changed : {stats['changed']} / {stats['chunks']}")
print(f"  nodes  {stats['nodes_before']} -> {stats['nodes_after']}")
print(f"  edges  {stats['edges_before']} -> {stats['edges_after']}"
      f"   (self-edges dropped {stats['self']}, duplicate edges dropped {stats['dupe']})")
print("\n  first 12 changed chunks:")
for f in changed_files[:12]:
    print("   ", f)

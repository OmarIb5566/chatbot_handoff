"""Compare the repaired corpus with the original: structure, retrieval, text."""
import json, re, sys
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend")); sys.path.insert(0, str(ROOT / "_verify"))
from translate import enable_utf8_stdout; enable_utf8_stdout()
from retriever import Retriever
from wf_render import render
from wf_score import score as rscore

CH = json.load(open(ROOT/"data"/"chunks.json", encoding="utf-8"))
OLD = json.load(open(ROOT/"data"/"workflow_chunks.json", encoding="utf-8"))
NEW = json.load(open(ROOT/"data"/"workflow_chunks_repaired.json", encoding="utf-8"))

def health(W):
    n_clean = dup = multi = unreach = selfl = 0
    for c in W:
        g = c.get("graph") or {}
        nodes, edges = g.get("nodes") or [], g.get("edges") or []
        if not nodes: continue
        lab = {x["id"]: (x.get("label") or "").strip() for x in nodes}
        labels = list(lab.values())
        d = len(labels) - len(set(labels))
        oe, ind, adj = defaultdict(list), defaultdict(int), defaultdict(set)
        for e in edges:
            oe[e["from"]].append(e["to"]); ind[e["to"]] += 1
            adj[e["from"]].add(e["to"]); adj[e["to"]].add(e["from"])
        seen, comps = set(), 0
        for x in lab:
            if x in seen: continue
            comps += 1; st = [x]
            while st:
                y = st.pop()
                if y in seen: continue
                seen.add(y); st.extend(adj[y] - seen)
        starts = [x for x in lab if ind[x] == 0 and oe[x]] or \
                 [x for x in lab if re.search(r"creation|start", lab[x], re.I)] or \
                 ([edges[0]["from"]] if edges else [])
        reach = set()
        if starts:
            st = [starts[0]]
            while st:
                y = st.pop()
                if y in reach: continue
                reach.add(y); st.extend(oe[y])
        r = sum(1 for e in edges if e["from"] in reach) / len(edges) * 100 if edges else 100
        s = sum(1 for e in edges if e["from"] == e["to"])
        dup += bool(d); multi += comps > 1; unreach += r < 100; selfl += bool(s)
        if comps == 1 and d == 0 and r == 100: n_clean += 1
    return dict(clean=n_clean, dup=dup, multi=multi, unreach=unreach, selfl=selfl, n=len(W))

print("STRUCTURE")
for lbl, W in (("original", OLD), ("repaired", NEW)):
    h = health(W)
    print(f"  {lbl:9} clean {h['clean']:3}/{h['n']} ({100*h['clean']/h['n']:.1f}%)  "
          f"dup-labels {h['dup']:3}  >1 component {h['multi']:3}  "
          f"unreachable {h['unreach']:3}  self-loops {h['selfl']:3}")

print("\nRENDERER ROUTE COVERAGE (deterministic walk over each graph)")
for lbl, W in (("original", OLD), ("repaired", NEW)):
    cov = g = 0
    for c in W:
        try:
            o = render(c)
            if o:
                s = rscore(o, c); cov += s["covered"]; g += s["gold"]
        except Exception: pass
    print(f"  {lbl:9} {cov}/{g} = {100*cov/g:.1f}%")

print("\nRETRIEVAL (merged index, production defaults)")
for lbl, W in (("original", OLD), ("repaired", NEW)):
    r = Retriever(CH + W)
    line = f"  {lbl:9} "
    for name, k in (("eval_set_workflow", 6), ("eval_set_workflow", 3),
                    ("eval_set_workflow", 1), ("eval_set_v2", 3), ("eval_set", 3)):
        items = [it for it in json.load(open(ROOT/"evals"/f"{name}.json", encoding="utf-8"))
                 if it["doc"] != "none"]
        h = sum(1 for it in items
                if it["doc"] in [x["filename"] for x in r.search(it["question"], top_k=k)])
        line += f"{name.replace('eval_set','ES')}@{k}: {h}/{len(items)}   "
    print(line)

print("\nTEXT DIFF SAMPLE — Drawing List (E1 Log)")
o = next(c for c in OLD if "Drawing List" in c["filename"])
n = next(c for c in NEW if "Drawing List" in c["filename"])
print("--- before ---"); print(o["text"])
print("--- after ----"); print(n["text"])

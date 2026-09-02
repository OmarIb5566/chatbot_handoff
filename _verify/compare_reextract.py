import json, re, sys
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"backend")); sys.path.insert(0, str(ROOT/"_verify"))
from translate import enable_utf8_stdout; enable_utf8_stdout()
from retriever import Retriever
from wf_render import render
from wf_score import score as rscore
CH  = json.load(open(ROOT/"data"/"chunks.json", encoding="utf-8"))
OLD = json.load(open(ROOT/"data"/"workflow_chunks.json", encoding="utf-8"))
NEW = json.load(open(ROOT/"data"/"workflow_chunks_fixed.json", encoding="utf-8"))
def stats(W):
    clean=dup=multi=unreach=selfl=orph=0; edges=nodes=0
    for c in W:
        g=c.get("graph") or {}
        ns,es=g.get("nodes") or [],g.get("edges") or []
        if not ns: continue
        nodes+=len(ns); edges+=len(es)
        lab={n["id"]:(n.get("label") or "").strip() for n in ns}
        d=len(lab)-len(set(lab.values()))
        oe,ind,adj=defaultdict(list),defaultdict(int),defaultdict(set)
        for e in es:
            oe[e["from"]].append(e["to"]); ind[e["to"]]+=1
            adj[e["from"]].add(e["to"]); adj[e["to"]].add(e["from"])
        orph+=sum(1 for n in lab if not adj[n])
        seen=set(); comps=0
        for n in lab:
            if n in seen: continue
            comps+=1; st=[n]
            while st:
                x=st.pop()
                if x in seen: continue
                seen.add(x); st.extend(adj[x]-seen)
        starts=[n for n in lab if ind[n]==0 and oe[n]] or \
               [n for n in lab if re.search(r"creation|start",lab[n],re.I)] or \
               ([es[0]["from"]] if es else [])
        reach=set()
        if starts:
            st=[starts[0]]
            while st:
                x=st.pop()
                if x in reach: continue
                reach.add(x); st.extend(oe[x])
        r=sum(1 for e in es if e["from"] in reach)/len(es)*100 if es else 100
        dup+=bool(d); multi+=comps>1; unreach+=r<100
        selfl+=bool(sum(1 for e in es if e["from"]==e["to"]))
        if comps==1 and d==0 and r==100: clean+=1
    return dict(n=len(W),clean=clean,dup=dup,multi=multi,unreach=unreach,
                selfl=selfl,orph=orph,nodes=nodes,edges=edges)
print("STRUCTURE")
for lbl,W in (("original ",OLD),("reextract",NEW)):
    s=stats(W)
    print(f"  {lbl} chunks {s['n']:3}  clean {s['clean']:3} ({100*s['clean']/s['n']:.1f}%)  "
          f"dup {s['dup']:3}  >1comp {s['multi']:3}  unreach {s['unreach']:3}  "
          f"selfloop {s['selfl']:3}  orphan-nodes {s['orph']:3}  nodes {s['nodes']}  edges {s['edges']}")
def eset(W):
    d=defaultdict(set)
    for c in W:
        g=c.get("graph") or {}
        lab={n["id"]:(n.get("label") or "").strip() for n in g.get("nodes") or []}
        for e in g.get("edges") or []:
            a,b=lab.get(e["from"],""),lab.get(e["to"],"")
            if a and b and a!=b: d[c["filename"]].add((a,b))
    return d
eo,en=eset(OLD),eset(NEW)
gain=loss=gf=lf=0
for fn in set(eo)|set(en):
    g=en[fn]-eo[fn]; l=eo[fn]-en[fn]
    gain+=len(g); loss+=len(l); gf+=bool(g); lf+=bool(l)
print(f"\nEDGES  files gaining {gf}, losing {lf}   total gained {gain}, lost {loss}")
print("  files that LOST edges:")
shown=0
for fn in sorted(set(eo)|set(en)):
    l=eo[fn]-en[fn]
    if l:
        shown+=1; print(f"    -{len(l):2}  {fn[:62]}")
        for a,b in sorted(l)[:2]: print(f"          {a[:26]} -> {b[:26]}")
if not shown: print("    (none)")
print("\nTHE 18 PREVIOUSLY-DISCONNECTED FILES")
files=json.load(open(ROOT/"_verify"/"disconnected_files.json",encoding="utf-8"))
def orphans(W,fn):
    out=[]
    for c in W:
        if c["filename"]!=fn: continue
        g=c.get("graph") or {}
        lab={n["id"]:(n.get("label") or "").strip() for n in g.get("nodes") or []}
        adj=defaultdict(set)
        for e in g.get("edges") or []:
            adj[e["from"]].add(e["to"]); adj[e["to"]].add(e["from"])
        out+=[lab[n] for n in lab if not adj[n]]
    return out
fixed=0
for fn in files:
    a,b=orphans(OLD,fn),orphans(NEW,fn)
    tag="FIXED " if not b else ("better" if len(b)<len(a) else "same  ")
    fixed+= (not b)
    print(f"  {tag} {fn[:58]:60} {len(a)} -> {len(b)}" + (f"  {b[:2]}" if b else ""))
print(f"\n  fully fixed: {fixed}/{len(files)}")
print("\nRENDERER ROUTE COVERAGE")
for lbl,W in (("original ",OLD),("reextract",NEW)):
    cov=g=0
    for c in W:
        try:
            o=render(c)
            if o: s=rscore(o,c); cov+=s["covered"]; g+=s["gold"]
        except Exception: pass
    print(f"  {lbl} {cov}/{g} = {100*cov/g:.1f}%")
print("\nRETRIEVAL")
for lbl,W in (("original ",OLD),("reextract",NEW)):
    r=Retriever(CH+W); line=f"  {lbl} "
    for name,k in (("eval_set_workflow",6),("eval_set_workflow",3),
                   ("eval_set_v2",3),("eval_set",3)):
        items=[it for it in json.load(open(ROOT/"evals"/f"{name}.json",encoding="utf-8"))
               if it["doc"]!="none"]
        h=sum(1 for it in items
              if it["doc"] in [x["filename"] for x in r.search(it["question"],top_k=k)])
        line+=f"{name.replace('eval_set','ES')}@{k}: {h}/{len(items)}   "
    print(line)

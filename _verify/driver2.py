"""Gap-filling pass: discriminating top-3 sweep, and everything on the MERGED
(production-shaped) index. Read-only."""
import json, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
OUT = ROOT / "_verify" / "results"; OUT.mkdir(parents=True, exist_ok=True)
LOG = open(ROOT / "_verify" / "driver2_log.txt", "w", encoding="utf-8", buffering=1)
def log(*a): LOG.write(" ".join(str(x) for x in a) + "\n")
def dump(n, o): (OUT / n).write_text(json.dumps(o, indent=2, ensure_ascii=False), encoding="utf-8")

from retriever import Retriever, workflow_intent, WORKFLOW_SOURCE_TYPES
CH = json.load(open(ROOT / "data" / "chunks.json", encoding="utf-8"))
WF = json.load(open(ROOT / "data" / "workflow_chunks.json", encoding="utf-8"))
E = {n: json.load(open(ROOT / "evals" / f"{n}.json", encoding="utf-8"))
     for n in ("eval_set_v2", "eval_set", "eval_set_workflow")}
ANS = {k: [it for it in v if it["doc"] != "none"] for k, v in E.items()}
RM = Retriever(CH + WF)          # production shape: one index, both populations
log("merged index:", len(CH + WF), "chunks")

# which process questions trip the keyword detector at all
trip = {k: [it["id"] for it in ANS[k] if workflow_intent(it["question"])] for k in ANS}
log("workflow_intent fires on:", json.dumps(trip))
dump("06_intent_firing.json", {"firing": trip,
     "questions": {k: [{"id": it["id"], "q": it["question"], "doc": it["doc"]}
                       for it in ANS[k] if workflow_intent(it["question"])] for k in ANS}})

def run(setname, top_k, **kw):
    items, hits, misses, ns, nws = ANS[setname], 0, [], [], []
    for it in items:
        res = RM.search(it["question"], top_k=top_k, **kw)
        nw = sum(1 for x in res if x.get("source_type") in WORKFLOW_SOURCE_TYPES)
        ns.append(len(res)); nws.append(nw)
        if it["doc"] in [x["filename"] for x in res]: hits += 1
        else: misses.append(it["id"])
    mean = lambda x: round(sum(x)/len(x), 3)
    return {"hits": hits, "n": len(items), "acc": round(hits/len(items), 4),
            "misses": misses, "mean_returned": mean(ns),
            "mean_workflow": mean(nws),
            "mean_process": round(sum(a-b for a, b in zip(ns, nws))/len(ns), 3)}

def run_subset(setname, ids, top_k, **kw):
    items = [it for it in ANS[setname] if it["id"] in ids]
    if not items: return None
    hits, ns, nws, misses = 0, [], [], []
    for it in items:
        res = RM.search(it["question"], top_k=top_k, **kw)
        nw = sum(1 for x in res if x.get("source_type") in WORKFLOW_SOURCE_TYPES)
        ns.append(len(res)); nws.append(nw)
        if it["doc"] in [x["filename"] for x in res]: hits += 1
        else: misses.append(it["id"])
    mean = lambda x: round(sum(x)/len(x), 3)
    return {"hits": hits, "n": len(items), "misses": misses,
            "mean_returned": mean(ns), "mean_workflow": mean(nws),
            "mean_process": round(sum(a-b for a, b in zip(ns, nws))/len(ns), 3)}

# ---- A/B: boost x floor at top-3 AND top-6, merged index, all three sets ----
try:
    log("\n=== boost x floor, MERGED index ===")
    rows = []
    for b in (0, 0.06, 0.12, 0.2):
        for f in (0, 1, 2, 3):
            for setname in ("eval_set_workflow", "eval_set_v2", "eval_set"):
                for k in (3, 6):
                    o = run(setname, k, workflow_boost=b, workflow_floor=f)
                    sub = run_subset(setname, set(trip[setname]), k,
                                     workflow_boost=b, workflow_floor=f)
                    rows.append({"workflow_boost": b, "workflow_floor": f,
                                 "eval_set": setname, "top_k": k, **o,
                                 "intent_subset": sub})
                    log(f"  b={b:<5} f={f} k={k} {setname:18s} {o['hits']:>3}/{o['n']} "
                        f"({o['acc']:.1%}) ret={o['mean_returned']} proc={o['mean_process']} "
                        f"| intent-subset " +
                        (f"{sub['hits']}/{sub['n']} proc={sub['mean_process']}" if sub else "n/a"))
    dump("07_boost_floor_merged.json", rows)
except Exception:
    log("FAILED A/B\n" + traceback.format_exc())

# ---- C: min_rel x fusion on the MERGED index ----
try:
    log("\n=== min_rel x fusion, MERGED index ===")
    rows = []
    for fusion in ("weighted", "rrf"):
        RM.fusion = fusion
        for setname in ("eval_set_v2", "eval_set", "eval_set_workflow"):
            for mr in (0.0, 0.4, 0.55, 0.7):
                r = {"fusion": fusion, "eval_set": setname, "min_rel": mr}
                for k in (3, 6):
                    o = run(setname, k, min_rel=mr)
                    r[f"top{k}"] = o
                rows.append(r)
                log(f"  {fusion:8s} {setname:18s} min_rel={mr:<5} "
                    f"top3={r['top3']['hits']}/{r['top3']['n']} "
                    f"top6={r['top6']['hits']}/{r['top6']['n']} "
                    f"ret6={r['top6']['mean_returned']} proc6={r['top6']['mean_process']}")
    RM.fusion = "weighted"
    dump("08_min_rel_merged.json", rows)
except Exception:
    log("FAILED C\n" + traceback.format_exc())

# ---- D: what the change actually bought, on the production index ----
try:
    log("\n=== before/after on merged index ===")
    ba = {}
    for setname in ("eval_set_workflow", "eval_set_v2", "eval_set"):
        old = run(setname, 3, workflow_boost=0, workflow_floor=0, min_rel=0)
        new = run(setname, 6)
        new3 = run(setname, 3)
        ba[setname] = {"old_top3_no_mechanisms": old, "new_top6_defaults": new,
                       "new_top3_defaults": new3}
        log(f"  {setname:18s} OLD top-3 {old['hits']}/{old['n']}  ->  "
            f"NEW top-6 {new['hits']}/{new['n']} (proc {new['mean_process']})  "
            f"| NEW top-3 {new3['hits']}/{new3['n']}")
    dump("09_before_after.json", ba)
except Exception:
    log("FAILED D\n" + traceback.format_exc())

log("\nDRIVER2 DONE")
LOG.close()

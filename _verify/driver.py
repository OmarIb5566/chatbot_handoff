"""Verification driver. Writes JSON to _verify/results/. Changes nothing in the repo."""
import json, sys, os, time, traceback, statistics, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
OUT = ROOT / "_verify" / "results"
OUT.mkdir(parents=True, exist_ok=True)
LOG = open(ROOT / "_verify" / "driver_log.txt", "w", encoding="utf-8", buffering=1)

def log(*a):
    m = " ".join(str(x) for x in a)
    LOG.write(m + "\n")

def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    log("  wrote", name)

t_start = time.time()
log("driver start", time.ctime())

from retriever import Retriever, workflow_intent, WORKFLOW_SOURCE_TYPES
import retriever as R
log("retriever imported")

# ---- record what actually loaded, so 'was it the real embedder' is answerable
emb_info = {}
try:
    import sentence_transformers, torch
    emb_info["sentence_transformers"] = sentence_transformers.__version__
    emb_info["torch"] = torch.__version__
except Exception as e:
    emb_info["import_error"] = repr(e)
emb_info["DEFAULT_MODEL"] = R.DEFAULT_MODEL
emb_info["python"] = sys.version

CH = json.load(open(ROOT / "data" / "chunks.json", encoding="utf-8"))
WF = json.load(open(ROOT / "data" / "workflow_chunks.json", encoding="utf-8"))
log(f"chunks {len(CH)} process, {len(WF)} workflow")

def load_eval(name):
    return json.load(open(ROOT / "evals" / name, encoding="utf-8"))

EVALS = {
    "eval_set_v2": load_eval("eval_set_v2.json"),
    "eval_set": load_eval("eval_set.json"),
    "eval_set_workflow": load_eval("eval_set_workflow.json"),
}
ANS = {k: [it for it in v if it["doc"] != "none"] for k, v in EVALS.items()}
for k, v in ANS.items():
    log(f"  {k}: {len(EVALS[k])} questions, {len(v)} answerable")

# eval_retrieval.load_chunks parity: merge workflow chunks only when the set needs them
def corpus_for(setname):
    items = ANS[setname]
    have = {c["filename"] for c in CH}
    needs = {it["doc"] for it in items} - have
    return (CH + WF, True) if needs else (CH, False)

t0 = time.time()
log("building retriever: process-only ...")
R_PROC = Retriever(CH)
log(f"  built in {time.time()-t0:.1f}s")
t0 = time.time()
log("building retriever: merged ...")
R_MERGED = Retriever(CH + WF)
log(f"  built in {time.time()-t0:.1f}s")

emb_info["embedding_shape_process"] = list(R_PROC.embeddings.shape)
emb_info["embedding_dtype"] = str(R_PROC.embeddings.dtype)
emb_info["model_repr"] = str(type(R_PROC.model))
# a real MiniLM sanity signal: paraphrase pair should score far above an unrelated pair
try:
    import numpy as np
    v = R_PROC.model.encode(["who approves the purchase order",
                             "which person signs off on a purchase order",
                             "the cat sat on the mat"], normalize_embeddings=True)
    emb_info["sanity_paraphrase_cos"] = float(np.dot(v[0], v[1]))
    emb_info["sanity_unrelated_cos"] = float(np.dot(v[0], v[2]))
except Exception as e:
    emb_info["sanity_error"] = repr(e)
dump("00_environment.json", emb_info)
log("environment:", json.dumps(emb_info))

RET = {"eval_set_v2": R_PROC, "eval_set": R_PROC, "eval_set_workflow": R_MERGED}

def run(setname, top_k, retr=None, **kw):
    """Returns (hits, n, misses, mean_returned, mean_process, mean_workflow, per_q)."""
    r = retr or RET[setname]
    items = ANS[setname]
    hits, misses, ns, nps, nws, per_q = 0, [], [], [], [], []
    for it in items:
        res = r.search(it["question"], top_k=top_k, **kw)
        found = [x["filename"] for x in res]
        nw = sum(1 for x in res if x.get("source_type") in WORKFLOW_SOURCE_TYPES)
        ns.append(len(res)); nws.append(nw); nps.append(len(res) - nw)
        ok = it["doc"] in found
        if ok: hits += 1
        else: misses.append(it["id"])
        per_q.append({"id": it["id"], "hit": ok, "n": len(res), "n_wf": nw,
                      "found": found,
                      "scores": [round(float(x["score"]), 4) for x in res]})
    mean = lambda x: round(sum(x) / len(x), 3) if x else 0
    return {"hits": hits, "n": len(items), "acc": round(hits / len(items), 4),
            "misses": misses, "mean_returned": mean(ns),
            "mean_process": mean(nps), "mean_workflow": mean(nws), "per_q": per_q}

results = {}

# ============================ STEP 1 ============================
try:
    log("\n=== STEP 1: regression ===")
    s1 = {}
    for setname in ("eval_set_v2", "eval_set"):
        s1[setname] = {}
        for k in (1, 3, 5):
            out = run(setname, k)
            s1[setname][f"top{k}"] = out
            log(f"  {setname} top-{k}: {out['hits']}/{out['n']} = {out['acc']:.1%} "
                f"(mean returned {out['mean_returned']})")
        # cutoff off, to isolate min_rel as the suspect
        out = run(setname, 3, min_rel=0)
        s1[setname]["top3_min_rel_0"] = out
        log(f"  {setname} top-3 min_rel=0: {out['hits']}/{out['n']} = {out['acc']:.1%}")
        # no workflow mechanisms at all
        out = run(setname, 3, min_rel=0, workflow_boost=0, workflow_floor=0)
        s1[setname]["top3_all_off"] = out
        log(f"  {setname} top-3 all-off: {out['hits']}/{out['n']} = {out['acc']:.1%}")
        # production-shaped index: merged corpus, which eval_retrieval never uses
        out = run(setname, 3, retr=R_MERGED)
        s1[setname]["top3_merged_index"] = out
        log(f"  {setname} top-3 on MERGED index: {out['hits']}/{out['n']} = {out['acc']:.1%}")
    tot3 = s1["eval_set_v2"]["top3"]["hits"] + s1["eval_set"]["top3"]["hits"]
    totn = s1["eval_set_v2"]["top3"]["n"] + s1["eval_set"]["top3"]["n"]
    s1["combined_top3"] = {"hits": tot3, "n": totn}
    log(f"  COMBINED top-3: {tot3}/{totn}  (baseline 113/124)")
    results["step1"] = s1
    dump("01_step1_regression.json", s1)
except Exception:
    log("STEP1 FAILED\n" + traceback.format_exc()); results["step1"] = {"error": traceback.format_exc()}

# ============================ STEP 2 ============================
try:
    log("\n=== STEP 2: workflow baseline ===")
    s2 = {"curve": {}, "curve_min_rel_0": {}, "channels": {}}
    for k in (1, 3, 6, 10):
        out = run("eval_set_workflow", k)
        s2["curve"][f"top{k}"] = out
        log(f"  fused top-{k}: {out['hits']}/{out['n']} = {out['acc']:.1%} "
            f"(mean returned {out['mean_returned']}, wf {out['mean_workflow']})")
    for k in (1, 3, 6, 10):
        out = run("eval_set_workflow", k, min_rel=0)
        s2["curve_min_rel_0"][f"top{k}"] = out
        log(f"  fused top-{k} (min_rel=0): {out['hits']}/{out['n']} = {out['acc']:.1%}")
    items = ANS["eval_set_workflow"]
    for label, fn in (("bm25", R_MERGED.search_bm25), ("dense", R_MERGED.search_dense)):
        hits, misses, per_q = 0, [], []
        for it in items:
            res = fn(it["question"], top_k=6)
            found = [x["filename"] for x in res]
            ok = it["doc"] in found
            hits += ok
            if not ok: misses.append(it["id"])
            nw = sum(1 for x in res if x.get("source_type") in WORKFLOW_SOURCE_TYPES)
            per_q.append({"id": it["id"], "hit": ok, "n_wf": nw, "found": found})
        s2["channels"][label] = {"hits": hits, "n": len(items),
                                 "acc": round(hits/len(items), 4),
                                 "misses": misses, "per_q": per_q,
                                 "mean_workflow": round(sum(p["n_wf"] for p in per_q)/len(per_q), 3)}
        log(f"  {label} top-6: {hits}/{len(items)} = {hits/len(items):.1%}")
    s2["channels"]["fused"] = s2["curve"]["top6"]
    # same three channels on the process set, for a reference point
    ch_proc = {}
    for label, fn in (("bm25", R_PROC.search_bm25), ("dense", R_PROC.search_dense)):
        h = sum(1 for it in ANS["eval_set_v2"]
                if it["doc"] in [x["filename"] for x in fn(it["question"], top_k=6)])
        ch_proc[label] = {"hits": h, "n": len(ANS["eval_set_v2"])}
        log(f"  [v2 reference] {label} top-6: {h}/{len(ANS['eval_set_v2'])}")
    s2["channels_process_reference"] = ch_proc
    results["step2"] = s2
    dump("02_step2_workflow.json", s2)
except Exception:
    log("STEP2 FAILED\n" + traceback.format_exc()); results["step2"] = {"error": traceback.format_exc()}

# ============================ STEP 3 ============================
try:
    log("\n=== STEP 3: min_rel sweep ===")
    s3 = []
    for fusion in ("weighted", "rrf"):
        R_PROC.fusion = fusion; R_MERGED.fusion = fusion
        for setname in ("eval_set_v2", "eval_set", "eval_set_workflow"):
            for mr in (0.0, 0.4, 0.55, 0.7):
                row = {"fusion": fusion, "eval_set": setname, "min_rel": mr}
                for k in (3, 6):
                    o = run(setname, k, min_rel=mr)
                    row[f"top{k}_hits"] = o["hits"]; row[f"top{k}_n"] = o["n"]
                    row[f"top{k}_acc"] = o["acc"]; row[f"top{k}_mean_returned"] = o["mean_returned"]
                    row[f"top{k}_mean_process"] = o["mean_process"]
                    row[f"top{k}_mean_workflow"] = o["mean_workflow"]
                    row[f"top{k}_misses"] = o["misses"]
                s3.append(row)
                log(f"  {fusion:8s} {setname:18s} min_rel={mr:<5} "
                    f"top3={row['top3_hits']}/{row['top3_n']} ({row['top3_acc']:.1%}) "
                    f"top6={row['top6_hits']}/{row['top6_n']} ({row['top6_acc']:.1%}) "
                    f"ret6={row['top6_mean_returned']}")
    R_PROC.fusion = "weighted"; R_MERGED.fusion = "weighted"
    results["step3"] = s3
    dump("03_step3_min_rel_sweep.json", s3)
except Exception:
    log("STEP3 FAILED\n" + traceback.format_exc()); results["step3"] = {"error": traceback.format_exc()}

# ============================ STEP 4 evidence ============================
try:
    log("\n=== STEP 4: retrieval evidence at default config ===")
    ev = []
    for it in ANS["eval_set_workflow"]:
        res = R_MERGED.search(it["question"], top_k=6)
        ev.append({
            "id": it["id"], "question": it["question"], "gold": it["doc"],
            "must_include": it["must_include"], "type": it.get("type"),
            "wf_intent": workflow_intent(it["question"]),
            "hit": it["doc"] in [x["filename"] for x in res],
            "retrieved": [{"rank": i+1, "score": round(float(x["score"]), 4),
                           "filename": x["filename"], "section": x.get("section"),
                           "source_type": x.get("source_type"),
                           "text": (x.get("text") or "")[:900]}
                          for i, x in enumerate(res)],
        })
        log(f"  {it['id']} hit={ev[-1]['hit']} n={len(res)}")
    dump("04_step4_evidence.json", ev)
    # full text of every gold doc, so 'is another diagram equally correct' is checkable
    golds = {it["doc"] for it in ANS["eval_set_workflow"]}
    seen = {}
    for c in CH + WF:
        if c["filename"] in golds:
            seen.setdefault(c["filename"], []).append(
                {"section": c.get("section"), "text": c.get("text"),
                 "coverage": c.get("coverage"), "audit_warnings": c.get("audit_warnings")})
    dump("04_gold_docs.json", seen)
    results["step4"] = "ok"
except Exception:
    log("STEP4 FAILED\n" + traceback.format_exc()); results["step4"] = {"error": traceback.format_exc()}

# ============================ STEP 5 ============================
try:
    log("\n=== STEP 5: boost x floor sweep ===")
    s5 = []
    for b in (0, 0.06, 0.12, 0.2):
        for f in (0, 1, 2, 3):
            for setname in ("eval_set_workflow", "eval_set_v2", "eval_set"):
                o = run(setname, 6, workflow_boost=b, workflow_floor=f)
                # starvation is only meaningful on queries where the intent fires
                wf_q = [p for p, it in zip(o["per_q"], ANS[setname])
                        if workflow_intent(it["question"])]
                mp = round(sum(p["n"] - p["n_wf"] for p in wf_q) / len(wf_q), 3) if wf_q else None
                mw = round(sum(p["n_wf"] for p in wf_q) / len(wf_q), 3) if wf_q else None
                s5.append({"workflow_boost": b, "workflow_floor": f, "eval_set": setname,
                           "hits": o["hits"], "n": o["n"], "acc": o["acc"],
                           "mean_returned": o["mean_returned"],
                           "n_wf_intent_queries": len(wf_q),
                           "mean_process_on_wf_intent": mp,
                           "mean_workflow_on_wf_intent": mw,
                           "misses": o["misses"]})
                log(f"  boost={b:<5} floor={f} {setname:18s} "
                    f"{o['hits']}/{o['n']} ({o['acc']:.1%}) ret={o['mean_returned']} "
                    f"wf_intent_q={len(wf_q)} proc_on_wf={mp}")
    results["step5"] = s5
    dump("05_step5_boost_floor.json", s5)
except Exception:
    log("STEP5 FAILED\n" + traceback.format_exc()); results["step5"] = {"error": traceback.format_exc()}

log(f"\nDRIVER DONE in {time.time()-t_start:.0f}s")
LOG.close()

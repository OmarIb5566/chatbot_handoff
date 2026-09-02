"""Round 4 A2/A3 — diagnosis only, reads nothing but the corpus. No code changes."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from chatbot import load_corpus
from retriever import Retriever, tokenize, embed_text, workflow_intent

chunks = load_corpus()
r = Retriever(chunks)
print(f"corpus: {len(chunks)} chunks over {len({c['filename'] for c in chunks})} documents\n")

ROC  = "Rule of Credit Amendments - ROC Amendments.pdf"
REVO = "Subcontract Amendments - Rental Equipment Variation Order WF.pdf"
BUV1 = "Subcontract Amendments - Subcontract Amendments BU-V1 WF.pdf"
def rows(fn): return [i for i, c in enumerate(chunks) if c["filename"] == fn]
roc_i  = rows(ROC)[0]
revo_i = rows(REVO)[0]
buv1_i = rows(BUV1)

QUERIES = [
 ("R2-Q1", "What is the approval workflow for a subcontract amendment?"),
 ("R3-Q1", "Who prepares the subcontract amendment request?"),
 ("R3-Q2", "Who signs the subcontract amendment request after it is prepared by the concerned department?"),
]

ALL = np.arange(len(chunks))

def rank_of(ranking, i):
    for p, (j, s) in enumerate(ranking, 1):
        if j == i: return p, s
    return None, None

print("="*104)
print("A2 — per-channel raw score and rank (full corpus, no routing)")
print("="*104)
a2 = []
for label, q in QUERIES:
    bm = r._bm25_ranking(q, ALL)
    dn = r._dense_ranking(q, ALL)
    fused = r.search(q, top_k=len(chunks), min_rel=0, workflow_boost=0, workflow_floor=0, route=False)
    fused_rank = {c["filename"] + "|" + str(c.get("section")): (p, c["score"])
                  for p, c in enumerate(fused, 1)}
    def fr(i):
        key = chunks[i]["filename"] + "|" + str(chunks[i].get("section"))
        return fused_rank.get(key, (None, None))
    print(f"\n--- {label}: {q}")
    print(f"    workflow_intent(query) = {workflow_intent(q)}")
    print(f"    {'chunk':<58} {'BM25':>18} {'dense':>18} {'fused':>18}")
    for name, i in (("Rule of Credit (ROC Amendments)", roc_i),
                    ("Rental Equipment Variation Order", revo_i),
                    ("BU-V1 flow 1", buv1_i[0]), ("BU-V1 flow 2", buv1_i[1]),
                    ("BU-V1 flow 3", buv1_i[2])):
        bp, bs = rank_of(bm, i); dp, ds = rank_of(dn, i); fp, fs = fr(i)
        print(f"    {name:<58} {f'#{bp} {bs:.4f}':>18} {f'#{dp} {ds:.4f}':>18} "
              f"{(f'#{fp} {fs:.4f}' if fp else 'n/a'):>18}")
        a2.append({"query": label, "chunk": name, "bm25_rank": bp, "bm25_score": bs,
                   "dense_rank": dp, "dense_score": ds, "fused_rank": fp, "fused_score": fs})
    # which terms of the query the ROC chunk actually carries
    qt = set(tokenize(q)); ct = set(tokenize(embed_text(chunks[roc_i])))
    vt = set(tokenize(embed_text(chunks[revo_i])))
    print(f"    query terms                : {sorted(qt)}")
    print(f"    ROC  chunk covers          : {sorted(qt & ct)}")
    print(f"    REVO chunk covers          : {sorted(qt & vt)}")
    print(f"    doc length (BM25 tokens)   : ROC={len(tokenize(embed_text(chunks[roc_i])))}  "
          f"REVO={len(tokenize(embed_text(chunks[revo_i])))}  "
          f"BU-V1 f1={len(tokenize(embed_text(chunks[buv1_i[0]])))}")

print("\n" + "="*104)
print("A3 — is the Rule of Credit chunk a hub?")
print("="*104)
# (a) its own text as a query
own = r.search(chunks[roc_i]["text"], top_k=10, min_rel=0, workflow_boost=0, workflow_floor=0, route=False)
print("\n(a) ROC chunk's own text as query, fused top-10:")
for p, c in enumerate(own, 1):
    print(f"   {p:2}. [{c['score']:.4f}] {c['filename'][:64]:64s} | {c.get('section')}")

# (b) appearances across every eval set, top-10, fused, production defaults
print("\n(b) appearances in fused top-10 across eval sets (production defaults):")
tot = {}
for name in ("eval_set_workflow", "eval_set_v2", "eval_set"):
    items = [it for it in json.load(open(ROOT/"evals"/f"{name}.json", encoding="utf-8"))
             if it["doc"] != "none"]
    hits, ranks, top1 = 0, [], 0
    for it in items:
        res = r.search(it["question"], top_k=10, min_rel=0)
        fn = [x["filename"] for x in res]
        if ROC in fn:
            hits += 1; p = fn.index(ROC) + 1; ranks.append(p)
            if p == 1: top1 += 1
    tot[name] = (hits, len(items), ranks, top1)
    mr = f"{sum(ranks)/len(ranks):.1f}" if ranks else "-"
    print(f"   {name:20s} ROC in top-10 on {hits:3}/{len(items):3} questions "
          f"(mean rank when present {mr}, rank-1 on {top1})")

# (c) same measurement for a few comparison chunks, so 'hub' has a scale
print("\n(c) same measure for comparison chunks (eval_set_workflow, top-10):")
items = [it for it in json.load(open(ROOT/"evals"/"eval_set_workflow.json", encoding="utf-8"))]
def appearances(fn):
    n, ranks = 0, []
    for it in items:
        res = r.search(it["question"], top_k=10, min_rel=0)
        names = [x["filename"] for x in res]
        if fn in names:
            n += 1; ranks.append(names.index(fn) + 1)
    return n, ranks
for fn in (ROC, REVO, BUV1,
           "Memos - Memos WF.pdf",
           "Bid Closing & Approval Internal Process - WF.pdf"):
    n, ranks = appearances(fn)
    mr = f"{sum(ranks)/len(ranks):.1f}" if ranks else "-"
    print(f"   {n:3}/18  mean rank {mr:>5}   {fn[:66]}")

# (d) geometric hub test: cosine of every chunk to the 18 workflow questions
qs = [it["question"] for it in items]
qe = r.model.encode(qs, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
sims = r.embeddings @ qe.T                      # (n_chunks, 18)
mean_sim = sims.mean(axis=1)
order = np.argsort(-mean_sim)
print("\n(d) mean cosine to all 18 workflow questions — top 12 chunks in the whole corpus:")
for p, i in enumerate(order[:12], 1):
    mark = "  <== ROC" if i == roc_i else ("  <== REVO" if i == revo_i else "")
    print(f"   {p:2}. {mean_sim[i]:.4f}  {chunks[i]['filename'][:60]:60s} | "
          f"{chunks[i].get('section')}{mark}")
roc_pos = int(np.where(order == roc_i)[0][0]) + 1
revo_pos = int(np.where(order == revo_i)[0][0]) + 1
print(f"\n   ROC  mean-cosine rank in corpus: {roc_pos} of {len(chunks)}  (mean {mean_sim[roc_i]:.4f})")
print(f"   REVO mean-cosine rank in corpus: {revo_pos} of {len(chunks)}  (mean {mean_sim[revo_i]:.4f})")
print(f"   corpus mean of mean-cosine     : {mean_sim.mean():.4f}   std {mean_sim.std():.4f}")
print(f"   ROC is {(mean_sim[roc_i]-mean_sim.mean())/mean_sim.std():.2f} sigma above the corpus mean")
json.dump(a2, open(ROOT/"_verify"/"results"/"16_r4_a2.json","w",encoding="utf-8"),
          indent=2, ensure_ascii=False)
print("\nDONE")

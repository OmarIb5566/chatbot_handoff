"""Round 4 A2 confirmation: is 'amendment' the operative term? Diagnosis only."""
import json, sys, math
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from chatbot import load_corpus
from retriever import Retriever, tokenize, embed_text
chunks = load_corpus(); r = Retriever(chunks)
ROC  = "Rule of Credit Amendments - ROC Amendments.pdf"
REVO = "Subcontract Amendments - Rental Equipment Variation Order WF.pdf"
roc_i  = next(i for i,c in enumerate(chunks) if c["filename"]==ROC)
revo_i = next(i for i,c in enumerate(chunks) if c["filename"]==REVO)
ALL = np.arange(len(chunks))
def rank(rk, i):
    for p,(j,s) in enumerate(rk,1):
        if j==i: return p, s
    return None, None

print("Does the ROC chunk contain the word 'subcontract' at all?")
toks = tokenize(embed_text(chunks[roc_i]))
print(f"  ROC BM25 tokens ({len(toks)}): {toks}")
print(f"  'subcontract' present: {'subcontract' in toks}")
print()
print("BM25 IDF of the query terms (rank_bm25 Okapi idf table):")
idf = r.bm25.idf
for t in ("subcontract","amendment","approval","workflow","request","prepare","sign","who","after","department","concerned"):
    df = sum(1 for d in r.corpus_tokens if t in d)
    print(f"  {t:12s} idf={idf.get(t, float('nan')):7.4f}  appears in {df:5d} / {len(chunks)} chunks")
print(f"\n  avg chunk length (BM25) = {r.bm25.avgdl:.1f} tokens;  ROC = {len(toks)};  "
      f"REVO = {len(tokenize(embed_text(chunks[revo_i])))}")

VARIANTS = [
 ("original          ", "Who signs the subcontract amendment request after it is prepared by the concerned department?"),
 ("amendment removed ", "Who signs the subcontract request after it is prepared by the concerned department?"),
 ("amendment->change ", "Who signs the subcontract change request after it is prepared by the concerned department?"),
 ("subcontract only  ", "Who signs the subcontract after it is prepared?"),
 ("no domain nouns   ", "Who signs after it is prepared by the concerned department?"),
]
print("\nROC vs REVO BM25 / fused rank as the query wording changes:")
print(f"  {'variant':<20} {'ROC bm25':>14} {'ROC fused':>14} {'REVO bm25':>14} {'REVO fused':>14}")
for label, q in VARIANTS:
    bm = r._bm25_ranking(q, ALL)
    fu = r.search(q, top_k=len(chunks), min_rel=0, workflow_boost=0, workflow_floor=0, route=False)
    fr = {c["filename"]+"|"+str(c.get("section")): (p, c["score"]) for p,c in enumerate(fu,1)}
    key = lambda i: chunks[i]["filename"]+"|"+str(chunks[i].get("section"))
    rb, sb = rank(bm, roc_i); vb, vs = rank(bm, revo_i)
    rf = fr.get(key(roc_i), (None,None)); vf = fr.get(key(revo_i), (None,None))
    print(f"  {label:<20} {f'#{rb} {sb:.2f}':>14} {f'#{rf[0]} {rf[1]:.3f}':>14} "
          f"{f'#{vb} {vs:.2f}':>14} {f'#{vf[0]} {vf[1]:.3f}':>14}")
print("\nDONE")

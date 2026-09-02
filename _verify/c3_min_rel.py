"""C3: which question(s) flip hit->miss between min_rel 0.55 and 0.7.
eval_retrieval.py exposes no --min-rel flag, so the parameter is passed straight
through Retriever.search here rather than adding a CLI flag."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from chatbot import load_corpus
from retriever import Retriever, WORKFLOW_SOURCE_TYPES
chunks = load_corpus()
r = Retriever(chunks)
items = [it for it in json.load(open(ROOT/"evals"/"eval_set.json", encoding="utf-8"))
         if it["doc"] != "none"]
print(f"corpus: {len(chunks)} chunks over {len({c['filename'] for c in chunks})} documents")
print(f"eval_set.json: {len(items)} answerable questions, top-6\n")
res = {}
for mr in (0.55, 0.7):
    hits, per = 0, {}
    ns = []
    for it in items:
        out = r.search(it["question"], top_k=6, min_rel=mr)
        found = [x["filename"] for x in out]
        ok = it["doc"] in found
        hits += ok; ns.append(len(out))
        per[it["id"]] = {"ok": ok, "found": found,
                         "rows": [{"rank": i+1, "score": round(float(x["score"]),4),
                                   "filename": x["filename"], "section": x.get("section"),
                                   "source_type": x.get("source_type")}
                                  for i, x in enumerate(out)]}
    res[mr] = per
    print(f"min_rel={mr}: {hits}/{len(items)} at top-6, mean chunks returned {sum(ns)/len(ns):.3f}")
flipped = [it for it in items if res[0.55][it["id"]]["ok"] and not res[0.7][it["id"]]["ok"]]
gained  = [it for it in items if not res[0.55][it["id"]]["ok"] and res[0.7][it["id"]]["ok"]]
print(f"\nflipped hit -> miss at 0.7: {[it['id'] for it in flipped]}")
print(f"flipped miss -> hit at 0.7: {[it['id'] for it in gained]}")
for it in flipped:
    print("\n" + "="*90)
    print(f"[{it['id']}] {it['question']}")
    print(f"  gold doc     : {it['doc']}")
    print(f"  must_include : {it['must_include']}")
    print(f"  type         : {it.get('type')}   review: {it.get('review')}")
    for label, mr in (("min_rel=0.55 (HIT)", 0.55), ("min_rel=0.70 (MISS)", 0.7)):
        print(f"  --- {label}: {len(res[mr][it['id']]['rows'])} chunks returned")
        for row in res[mr][it["id"]]["rows"]:
            mark = "  <== GOLD" if row["filename"] == it["doc"] else ""
            print(f"      [{row['score']:.4f}] {str(row['source_type']):15s} "
                  f"{row['filename'][:58]:58s} | {row['section']}{mark}")
json.dump({str(k): v for k, v in res.items()},
          open(ROOT/"_verify"/"results"/"13_c3_min_rel.json", "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)

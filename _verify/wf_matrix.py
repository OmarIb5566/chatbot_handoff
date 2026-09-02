"""Model x prompt-variant matrix on workflow answers. Read-only; the repo's
default WORKFLOW_FORMAT is monkeypatched per run, never edited on disk."""
import json, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "_verify"))
import chatbot as C
from chatbot import Chatbot, load_corpus, ollama_up
from wf_score import score, branch_faults

P1 = C.WORKFLOW_FORMAT
P2 = P1 + (
    "\n- After the final step, add one more line reading exactly "
    "'Returns and loops:' and under it one bullet per route that goes back to "
    "an earlier step, each written as: [step] -> [step] (label on the route). "
    "If the diagram has none, write 'Returns and loops: none.'"
)
QUESTIONS = [
    "how does the rental equipment variation order workflow go",
    "what is the document submittal ITP approval workflow",
    "how does the bonds request approval workflow go",
]
print("ollama_up:", ollama_up(), flush=True)
chunks = load_corpus()
by_name = {}
for c in chunks:
    by_name.setdefault(c["filename"], []).append(c)

results = []
for model in ("qwen3:14b", "qwen3.6:27b"):
    bot = Chatbot(chunks, model=model)
    for pname, ptext in (("P1_current", P1), ("P2_returns_line", P2)):
        C.WORKFLOW_FORMAT = ptext
        for q in QUESTIONS:
            t0 = time.time()
            try:
                rec = bot.ask(q, session_id=None)
                ans = rec.get("display") or ""
                # score against the diagram the answer itself cites
                cited = None
                for fn in by_name:
                    if fn.rsplit(".pdf", 1)[0] in ans and fn.endswith(".pdf"):
                        cand = [x for x in by_name[fn] if x.get("graph")]
                        if cand:
                            cited = cand[0]; break
                if cited is None:
                    wf = [h for h in rec["hits"] if h.get("graph")]
                    cited = wf[0] if wf else None
                row = {"model": model, "prompt": pname, "q": q,
                       "secs": round(time.time() - t0, 1),
                       "cited": cited["filename"] if cited else None,
                       "answer": ans}
                if cited:
                    s = score(ans, cited); f = branch_faults(ans, cited)
                    row.update({"gold": s["gold"], "covered": s["covered"],
                                "pct": s["pct"], "missing": s["missing"],
                                "extra": s["asserted_not_in_diagram"],
                                "borrowed": f["borrowed_conditions"],
                                "wrong_target": f["wrong_branch_targets"],
                                "pdf_mentions": s["pdf_mentions"]})
                results.append(row)
                print(f"{model:12} {pname:16} {row.get('pct')}% "
                      f"({row.get('covered')}/{row.get('gold')}) "
                      f"borrowed={len(row.get('borrowed') or [])} "
                      f"extra={len(row.get('extra') or [])} "
                      f"pdfs={row.get('pdf_mentions')} {row['secs']}s  | {q[:44]}", flush=True)
            except Exception:
                print(f"{model} {pname} FAILED on {q}\n{traceback.format_exc()}", flush=True)
                results.append({"model": model, "prompt": pname, "q": q,
                                "error": traceback.format_exc()})
            (ROOT/"_verify"/"results"/"20_wf_matrix.json").write_text(
                json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
C.WORKFLOW_FORMAT = P1
print("\nMATRIX DONE", flush=True)

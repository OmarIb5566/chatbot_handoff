"""Step 6: end-to-end through the real pipeline (needs Ollama). Read-only."""
import json, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
OUT = ROOT / "_verify" / "results"; OUT.mkdir(parents=True, exist_ok=True)
LOG = open(ROOT / "_verify" / "driver3_log.txt", "w", encoding="utf-8", buffering=1)
def log(*a): LOG.write(" ".join(str(x) for x in a) + "\n")

from chatbot import Chatbot, load_corpus, ollama_up, ollama_models, fit_context
log("ollama_up:", ollama_up())
log("models:", ollama_models())
bot = Chatbot(load_corpus())
log("chatbot ready; corpus", len(bot.chunks))

QS = [
    ("workflow", "what is the approval workflow for a subcontract amendment?", None),
    ("hr_policy", "how many days of annual leave do employees get?", None),
    ("arabic", "ما هي مدة الإجازة السنوية للموظفين؟", None),
    ("followup", "who signs after that?", "S1"),
]
recs = []
for label, q, sid in QS:
    log(f"\n--- {label}: {q}")
    t = time.time()
    try:
        rec = bot.ask(q, session_id="S1")
        slim = {k: rec.get(k) for k in
                ("workflow_intent", "n_workflow_hits", "route", "boost_prefixes",
                 "lang", "model", "retrieval_s", "generation_s", "total_s",
                 "had_reasoning", "was_rewritten", "reused_sources",
                 "gate_reasons", "query_used", "question_en", "spelling_repairs",
                 "rewrite_model")}
        slim["label"] = label; slim["question"] = q
        slim["validator"] = rec.get("validator")
        slim["n_hits"] = len(rec["hits"])
        slim["hits"] = [{"score": round(float(h["score"]), 4), "filename": h["filename"],
                         "section": h.get("section"), "source_type": h.get("source_type"),
                         "doc_code": h.get("doc_code")} for h in rec["hits"]]
        slim["answer"] = (rec.get("display") or rec.get("answer") or "")[:1600]
        recs.append(slim)
        log(f"  ok in {time.time()-t:.1f}s  wf_intent={slim['workflow_intent']} "
            f"n_wf={slim['n_workflow_hits']}/{slim['n_hits']} route={slim['route']} "
            f"validator={slim['validator'].get('verdict') if slim['validator'] else None}")
        for h in slim["hits"]:
            log(f"    [{h['score']:.3f}] {h['source_type']:15s} {h['filename'][:60]} | {h['section']}")
        log("  ANSWER: " + slim["answer"][:400].replace("\n", " "))
    except Exception:
        log("  FAILED\n" + traceback.format_exc())
        recs.append({"label": label, "question": q, "error": traceback.format_exc()})
    (OUT / "10_step6_smoke.json").write_text(
        json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8")
log("\nDRIVER3 DONE")
LOG.close()

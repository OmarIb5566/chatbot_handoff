"""Step 6 correction: follow-up in a clean session, immediately after the workflow
question, which is what the brief actually asks for."""
import json, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
OUT = ROOT / "_verify" / "results"
LOG = open(ROOT / "_verify" / "driver4_log.txt", "w", encoding="utf-8", buffering=1)
def log(*a): LOG.write(" ".join(str(x) for x in a) + "\n")
from chatbot import Chatbot, load_corpus
bot = Chatbot(load_corpus())
def slim(rec, label, q):
    return {"label": label, "question": q,
            **{k: rec.get(k) for k in ("workflow_intent","n_workflow_hits","route",
               "boost_prefixes","lang","reused_sources","was_rewritten","gate_reasons",
               "query_used","rewrite_model","retrieval_s","generation_s","total_s")},
            "validator_verdict": rec.get("validator",{}).get("verdict"),
            "hits": [{"score": round(float(h["score"]),4), "filename": h["filename"],
                      "section": h.get("section"), "source_type": h.get("source_type")}
                     for h in rec["hits"]],
            "answer": (rec.get("display") or "")[:1800]}
recs = []
for label, q in (("q1_workflow", "what is the approval workflow for a subcontract amendment?"),
                 ("q2_followup", "who signs after that?")):
    log(f"\n--- {label}: {q}")
    try:
        r = bot.ask(q, session_id="FRESH")
        s = slim(r, label, q); recs.append(s)
        log(f"  wf_intent={s['workflow_intent']} n_wf={s['n_workflow_hits']}/{len(s['hits'])} "
            f"reused={s['reused_sources']} rewritten={s['was_rewritten']} gate={s['gate_reasons']}")
        log(f"  query_used={s['query_used']!r} rewrite_model={s['rewrite_model']}")
        for h in s["hits"]:
            log(f"    [{h['score']:.3f}] {str(h['source_type']):15s} {h['filename'][:58]} | {h['section']}")
        log("  ANSWER: " + s["answer"][:500].replace("\n", " "))
    except Exception:
        log("FAILED\n" + traceback.format_exc()); recs.append({"label": label, "error": traceback.format_exc()})
    (OUT / "12_step6_followup.json").write_text(json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8")
log("\nDRIVER4 DONE")
LOG.close()

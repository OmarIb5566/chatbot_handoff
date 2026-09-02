"""C2: workflow question then pronoun follow-up, clean session."""
import json, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from chatbot import Chatbot, load_corpus, ollama_up
from contextualize import looks_like_followup
print("ollama_up:", ollama_up())
bot = Chatbot(load_corpus())
print("corpus:", len(bot.chunks), "chunks")
out = []
for label, q in (("Q1", "What is the approval workflow for a subcontract amendment?"),
                 ("Q2", "who signs after that?")):
    g = looks_like_followup(q, has_history=(label == "Q2"))
    print(f"\n--- {label}: {q}")
    print(f"  looks_like_followup(has_history={label=='Q2'}) -> is_followup={g.is_followup} "
          f"reasons={getattr(g,'reasons',None)}")
    rec = bot.ask(q, session_id="C2")
    d = {"label": label, "question": q, "gate_is_followup": g.is_followup,
         "gate_reasons_direct": list(getattr(g, "reasons", []) or []),
         **{k: rec.get(k) for k in ("workflow_intent","n_workflow_hits","route",
            "boost_prefixes","reused_sources","was_rewritten","gate_reasons",
            "query_used","rewrite_model","retrieval_s","generation_s","total_s")},
         "validator_verdict": rec.get("validator",{}).get("verdict"),
         "hits": [{"score": round(float(h["score"]),4), "filename": h["filename"],
                   "section": h.get("section"), "source_type": h.get("source_type")}
                  for h in rec["hits"]],
         "answer": (rec.get("display") or "")[:2200]}
    out.append(d)
    print(f"  rec: workflow_intent={d['workflow_intent']} n_wf={d['n_workflow_hits']}/{len(d['hits'])} "
          f"reused_sources={d['reused_sources']} was_rewritten={d['was_rewritten']}")
    print(f"  rec gate_reasons: {d['gate_reasons']}")
    print(f"  query_used: {d['query_used']!r}  rewrite_model={d['rewrite_model']}")
    for h in d["hits"]:
        print(f"    [{h['score']:.4f}] {str(h['source_type']):15s} {h['filename'][:60]:60s} | {h['section']}")
    print("  ANSWER:\n    " + d["answer"][:1200].replace("\n", "\n    "))
    (ROOT/"_verify"/"results"/"14_c2_followup.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

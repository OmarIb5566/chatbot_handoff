"""Re-run the exact question from the screenshots against the new prompt."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from chatbot import Chatbot, load_corpus, build_prompt, ollama_up
print("ollama_up:", ollama_up(), flush=True)
bot = Chatbot(load_corpus())
q = "how does the rental equipment variation order workflow go"
rec = bot.ask(q, session_id="UICHECK")
print(f"\nworkflow_intent={rec.get('workflow_intent')} "
      f"n_wf={rec.get('n_workflow_hits')}/{len(rec['hits'])} "
      f"validator={rec['validator']['verdict']}", flush=True)
for h in rec["hits"]:
    print(f"  [{h['score']:.3f}] {str(h.get('source_type')):15s} {h['filename'][:58]} | {h.get('section')}")
ans = rec.get("display") or ""
print("\n----- ANSWER -----")
print(ans)
print("----- END -----")
print(f"\nfilename mentions in answer: {ans.count('Rental Equipment Variation Order')}")
print(f"contains 'Source:' line     : {'Source:' in ans}")
# how many of the 12 routes in the cited chunk are named
chunk = next(c for c in bot.chunks if "Rental Equipment Variation Order" in c["filename"])
import re
routes = re.findall(r'^\s*-\s*(.+?)\s*->\s*(.+?)(?:\s*\[|$)', chunk["text"], re.M)
low = ans.lower()
named = [(a, b) for a, b in routes if a.strip().lower() in low and b.strip().lower() in low]
print(f"routes in chunk: {len(routes)}   routes both endpoints named in answer: {len(named)}")
for a, b in routes:
    hit = (a.strip().lower() in low and b.strip().lower() in low)
    print(f"   {'YES' if hit else ' no'}  {a.strip()} -> {b.strip()}")
(ROOT/"_verify"/"results"/"17_ui_check.json").write_text(
    json.dumps({"question": q, "answer": ans,
                "hits": [{"score": round(float(h['score']),4), "filename": h['filename'],
                          "section": h.get('section'), "source_type": h.get('source_type')}
                         for h in rec['hits']]}, indent=2, ensure_ascii=False), encoding="utf-8")

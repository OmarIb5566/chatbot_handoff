"""Re-test the workflow answer shape after the format change."""
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from chatbot import Chatbot, load_corpus, ollama_up
print("ollama_up:", ollama_up(), flush=True)
bot = Chatbot(load_corpus())
out = []
for q in ("how does the rental equipment variation order workflow go",
          "how many days of annual leave do employees get?"):
    rec = bot.ask(q, session_id=None)
    ans = rec.get("display") or ""
    print("\n" + "="*90)
    print(f"Q: {q}")
    print(f"workflow_intent={rec.get('workflow_intent')} "
          f"n_wf={rec.get('n_workflow_hits')}/{len(rec['hits'])} "
          f"validator={rec['validator']['verdict']}")
    print("----- ANSWER -----")
    print(ans)
    print("----- END -----")
    out.append({"q": q, "answer": ans,
                "n_workflow_hits": rec.get("n_workflow_hits"),
                "hits": [h["filename"] for h in rec["hits"]]})
    if "rental equipment" in q:
        chunk = next(c for c in bot.chunks if "Rental Equipment Variation Order" in c["filename"])
        routes = re.findall(r'^\s*-\s*(.+?)\s*->\s*(.+?)(?:\s*\[|$)', chunk["text"], re.M)
        low = ans.lower()
        print(f"\nroutes in cited chunk: {len(routes)} — endpoint-pair presence "
              f"(substring, indicative only):")
        for a, b in routes:
            a, b = a.strip(), b.strip()
            print(f"   {'YES' if (a.lower() in low and b.lower() in low) else ' no'}  {a} -> {b}")
        print(f"\nfilename mentions: {ans.count('Rental Equipment Variation Order')}")
        print(f"has 'Source:' line: {'Source:' in ans}")
        print(f"has old headings  : {'**Main path**' in ans or '**Branches**' in ans}")
        print(f"has 'begins with' : {'begins with' in ans.lower()}")
(ROOT/"_verify"/"results"/"18_ui_check2.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

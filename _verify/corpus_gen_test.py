"""Does the repaired corpus produce a better answer? Bonds Request, P2 prompt."""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"backend")); sys.path.insert(0, str(ROOT/"_verify"))
from translate import enable_utf8_stdout; enable_utf8_stdout()
import chatbot as C
from chatbot import Chatbot
from wf_score import score, branch_faults
CH  = json.load(open(ROOT/"data"/"chunks.json", encoding="utf-8"))
OLD = json.load(open(ROOT/"data"/"workflow_chunks.json", encoding="utf-8"))
NEW = json.load(open(ROOT/"data"/"workflow_chunks_repaired.json", encoding="utf-8"))
Q = "how does the bonds request approval workflow go"
for lbl, W in (("ORIGINAL corpus", OLD), ("REPAIRED corpus", NEW)):
    bot = Chatbot(CH + [c for c in W
                        if not any("carry no label" in w for w in (c.get("audit_warnings") or []))
                        or (c.get("graph") and any((n.get("label") or "").strip()
                                                   for n in c["graph"]["nodes"]))],
                  model="qwen3:14b")
    t0 = time.time(); rec = bot.ask(Q, session_id=None); ans = rec.get("display") or ""
    cited = None
    for c in W:
        if c.get("graph") and c["filename"].rsplit(".pdf", 1)[0] in ans:
            cited = c; break
    print("=" * 92); print(f"{lbl}   ({time.time()-t0:.0f}s)")
    print(ans)
    if cited:
        s = score(ans, cited); f = branch_faults(ans, cited)
        print(f"\n  cited: {cited['filename']}")
        print(f"  routes {s['covered']}/{s['gold']} = {s['pct']}%   "
              f"asserted-not-in-diagram {len(s['asserted_not_in_diagram'])}   "
              f"borrowed {len(f['borrowed_conditions'])}   wrong-target {len(f['wrong_branch_targets'])}")
    else:
        print("  (could not identify the cited diagram)")

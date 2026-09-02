import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"backend"))
from translate import enable_utf8_stdout; enable_utf8_stdout()
from chatbot import load_corpus
from retriever import Retriever, doc_code_prefix
c = load_corpus()
wf = [x for x in c if x.get("source_type") in ("pdf_vector","vlm_description")]
print(f"load_corpus() -> {len(c)} chunks ({len(c)-len(wf)} process + {len(wf)} workflow)")
print(f"  workflow chunks with family : {sum(1 for x in wf if x.get('family'))}/{len(wf)}")
print(f"  workflow chunks with variant: {sum(1 for x in wf if x.get('variant'))}/{len(wf)}")
print(f"  workflow chunks with doc_code: {sum(1 for x in wf if x.get('doc_code'))}/{len(wf)}")
print(f"  workflow chunks that now yield a routing prefix: "
      f"{sum(1 for x in wf if doc_code_prefix(x))}/{len(wf)}")
for x in wf:
    if doc_code_prefix(x):
        print(f"     {doc_code_prefix(x):6} {x.get('doc_code'):16} {x['filename'][:50]}")
r = Retriever(c)
print("\nRETRIEVAL on the live corpus")
for name, ks in (("eval_set_v2",(3,)), ("eval_set",(3,)), ("eval_set_workflow",(1,3,6))):
    items=[it for it in json.load(open(ROOT/"evals"/f"{name}.json",encoding="utf-8"))
           if it["doc"]!="none"]
    out=[]
    for k in ks:
        h=sum(1 for it in items
              if it["doc"] in [x["filename"] for x in r.search(it["question"], top_k=k)])
        out.append(f"top-{k}: {h}/{len(items)}")
    print(f"  {name:20} {'   '.join(out)}")

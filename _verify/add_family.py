"""family / variant on every workflow chunk; doc_code only where a REAL code
exists, resolved once per DOCUMENT so a file's chunks cannot disagree.
Preference: a code in the filename (that is the document's own identity) beats
one found in the page text; among text codes a process code (P-/M-) beats an
F- form reference, which is a pointer to a form, not the document's number."""
import json, re
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "workflow_chunks_fixed.json"
W = json.load(open(P, encoding="utf-8"))
CODE = re.compile(r'\b((?:F\s*-\s*)?[PM]\s*-\s*[A-Z]{2,4}\s*-\s*\d{1,2}(?:\s*-\s*\d{1,2})?)\b', re.I)
tidy = lambda s: re.sub(r'\s+', ' ', s).strip()
norm = lambda s: re.sub(r'\s*-\s*', '-', s.upper())

by_doc = {}
for c in W:
    by_doc.setdefault(c["filename"], []).append(c)

resolved = {}
for fn, cs in by_doc.items():
    m = CODE.search(fn)
    if m:
        resolved[fn] = (norm(m.group(1)), "filename"); continue
    found = Counter()
    for c in cs:
        for mm in CODE.finditer(c.get("text") or ""):
            found[norm(mm.group(1))] += 1
    if not found:
        continue
    process = [k for k in found if not k.startswith("F-")]
    pick = sorted(process or list(found), key=lambda k: (-found[k], k))[0]
    resolved[fn] = (pick, "page text")

for c in W:
    stem = c["filename"].rsplit(".pdf", 1)[0]
    fam, var = stem.split(" - ", 1) if " - " in stem else (stem, "")
    c["family"], c["variant"] = tidy(fam), tidy(var)
    c.pop("doc_code_source", None)
    if c["filename"] in resolved:
        c["doc_code"], c["doc_code_source"] = resolved[c["filename"]]
    else:
        c["doc_code"] = None
P.write_text(json.dumps(W, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"family/variant on {len(W)} chunks; {len({c['family'] for c in W})} families")
print(f"documents with a real doc_code: {len(resolved)} / {len(by_doc)}")
for fn, (code, src) in sorted(resolved.items()):
    print(f"   {code:16} ({src:9}) {fn[:56]}")
print(f"chunks carrying a doc_code: {sum(1 for c in W if c.get('doc_code'))}/{len(W)}")

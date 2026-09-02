"""Re-extract every workflow PDF with the patched extractor.
Writes data/workflow_chunks_reextracted.json. Never touches the original."""
import json, sys, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from translate import enable_utf8_stdout; enable_utf8_stdout()
from workflow_vector import to_chunks
SRC = ROOT / "Workflows"
pdfs = sorted(SRC.glob("*.pdf"))
print(f"{len(pdfs)} workflow PDFs")
out, audit, failed = [], [], []
for i, pdf in enumerate(pdfs, 1):
    try:
        chunks, rec = to_chunks(pdf)
        out += chunks; audit.append(rec)
    except Exception:
        failed.append((pdf.name, traceback.format_exc().strip().split("\n")[-1]))
    if i % 20 == 0:
        print(f"  ... {i}/{len(pdfs)}", flush=True)
print(f"\nextracted {len(out)} chunks from {len(pdfs) - len(failed)} PDFs; failures {len(failed)}")
for fn, e in failed[:10]:
    print(f"   FAIL {fn[:60]}: {e}")
(ROOT/"data"/"workflow_chunks_reextracted.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
(ROOT/"data"/"workflow_vector_audit_reextracted.json").write_text(
    json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
print("wrote data/workflow_chunks_reextracted.json")

"""
Extraction pipeline for the real RME process PDFs.

Primary path is PyMuPDF (fitz) text extraction. These source PDFs carry
selectable/embedded text, so native extraction is the expected path for
essentially every page. OCR is a *fallback only*: it fires when a page's
native text looks suspicious (too short, or too little alphabetic content
relative to its length), which is what a genuinely scanned or image-only
page looks like.

Every page records which path it took, and the run prints/writes a fallback
log. Watch that number as the corpus grows to 500-600 docs: if it stays at
or near zero, the OCR dependency (tesseract binary + PIL rendering) is
solving a problem this corpus does not have, and can be dropped from the
production deployment rather than carried as an install requirement.

Usage:
    python extract_pipeline.py                  # ../processes_pdf -> ./extracted_raw.json
    python extract_pipeline.py --src ../other_pdfs --out ./other_raw.json
    python extract_pipeline.py --no-ocr         # hard-disable fallback, just report it
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF

# Resolve everything relative to this file (the "files" folder), not the CWD,
# so the pipeline works no matter where it's invoked from.
HERE = Path(__file__).resolve().parent
DEFAULT_SRC = HERE.parent / "processes_pdf"
DEFAULT_OUT = HERE / "extracted_raw.json"
DEFAULT_LOG = HERE / "ocr_fallback_log.json"

# Fallback trigger thresholds. Deliberately conservative: we would rather OCR
# a handful of genuinely sparse pages than silently ship an empty page into
# the chunker, where it becomes a missing process step nobody notices.
MIN_CHARS = 20        # below this, the page yielded essentially nothing
MIN_ALPHA_RATIO = 0.4  # below this, output is mostly symbols/gibberish
OCR_DPI = 300


def normalize_filename(pdf_path: Path) -> str:
    """Map a real PDF filename to the key convention used by eval_set.json.

    'P-CM-01 Customer Satisfaction Process (1).pdf'
        -> 'PCM01_Customer_Satisfaction_Process_1.pdf'

    The eval harness matches retrieved chunks against eval_set's `doc` field
    by exact string, so this mapping is load-bearing: change it and the eval
    set silently stops matching anything.
    """
    stem = pdf_path.stem
    stem = re.sub(r"[-()]", "", stem)           # P-CM-01 -> PCM01, (1) -> 1
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem)  # spaces/punctuation -> _
    stem = re.sub(r"_+", "_", stem).strip("_")
    return f"{stem}.pdf"


def text_quality_ok(text: str) -> bool:
    """Heuristic: reject empty/near-empty or mostly-non-alpha text."""
    if not text or len(text.strip()) < MIN_CHARS:
        return False
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    return alpha_ratio > MIN_ALPHA_RATIO


def ocr_page(page: "fitz.Page") -> str:
    """Render the page and OCR it. Imported lazily so a missing tesseract
    binary costs nothing on the (expected) all-native-text corpus."""
    import io

    from PIL import Image
    import pytesseract

    pix = page.get_pixmap(dpi=OCR_DPI)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)


def process_pdf(path: Path, allow_ocr: bool = True) -> list[dict]:
    """Extract one PDF page-by-page, with OCR fallback on low-quality pages."""
    pages = []
    with fitz.open(path) as doc:
        for n, page in enumerate(doc, start=1):
            # "text" preserves reading order well enough for the section/step
            # regexes in chunker.py; "blocks"/"words" would need reassembly.
            native = page.get_text("text")

            if text_quality_ok(native):
                pages.append({"page": n, "text": native, "source": "native_text"})
                continue

            # --- fallback path: this page did not yield usable native text ---
            if not allow_ocr:
                pages.append({
                    "page": n,
                    "text": native,
                    "source": "native_text_low_quality",
                    "ocr_skipped": True,
                })
                continue

            try:
                ocr_text = ocr_page(page)
                pages.append({
                    "page": n,
                    "text": ocr_text,
                    "source": "ocr_fallback",
                    "ocr_quality_ok": text_quality_ok(ocr_text),
                })
            except Exception as e:
                # A missing tesseract binary must not kill a 600-doc run.
                # Keep whatever native text we got and flag the page loudly.
                pages.append({
                    "page": n,
                    "text": native,
                    "source": "ocr_failed",
                    "ocr_error": f"{type(e).__name__}: {e}",
                })
    return pages


def run(src_dir: Path, out_path: Path, log_path: Path, allow_ocr: bool = True) -> dict:
    pdfs = sorted(src_dir.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {src_dir}")

    results: dict[str, list[dict]] = {}
    fallback_log: list[dict] = []
    t0 = time.perf_counter()

    for pdf_path in pdfs:
        key = normalize_filename(pdf_path)
        pages = process_pdf(pdf_path, allow_ocr=allow_ocr)
        results[key] = pages

        flagged = [p for p in pages if p["source"] != "native_text"]
        for p in flagged:
            fallback_log.append({
                "file": key,
                "source_file": pdf_path.name,
                "page": p["page"],
                "reason": p["source"],
                "chars": len(p.get("text") or ""),
                "error": p.get("ocr_error"),
            })

        native = len(pages) - len(flagged)
        note = f", {len(flagged)} FELL BACK" if flagged else ""
        print(f"{key}: {len(pages)} pages, {native} native{note}")

    elapsed = time.perf_counter() - t0
    total_pages = sum(len(p) for p in results.values())

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    log_path.write_text(json.dumps({
        "docs": len(results),
        "total_pages": total_pages,
        "fallback_pages": len(fallback_log),
        "fallback_rate": round(len(fallback_log) / max(total_pages, 1), 4),
        "pages": fallback_log,
    }, indent=2), encoding="utf-8")

    print(f"\n{len(results)} docs, {total_pages} pages in {elapsed:.1f}s")
    print(f"OCR fallback pages: {len(fallback_log)}/{total_pages} "
          f"({len(fallback_log)/max(total_pages,1):.1%})")
    if not fallback_log:
        print("  -> zero fallbacks: native text extraction covered the whole corpus.")
    print(f"Saved  {out_path}")
    print(f"Log    {log_path}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="folder of source PDFs")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output extracted_raw.json")
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG, help="OCR fallback log")
    ap.add_argument("--no-ocr", action="store_true", help="disable OCR fallback entirely")
    args = ap.parse_args()
    run(args.src.resolve(), args.out.resolve(), args.log.resolve(), allow_ocr=not args.no_ocr)


if __name__ == "__main__":
    main()

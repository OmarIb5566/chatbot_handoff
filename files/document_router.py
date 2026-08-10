"""Which extractor does this PDF belong to?

The corpus holds two kinds of document that need completely different readers,
and sending one to the other's reader fails *silently* in both directions:

  process document  -> adaptive_chunker.py
        Multi-page ISO prose written to the house template. Meaning is in the
        text, and structure is in the layout (font size, bold, numbering).

  workflow diagram  -> workflow_extractor.py
        A one-page approval flowchart. Meaning is in the EDGES - which arrow
        points where, which diamond loops back on "Not Approved", which red
        monetary label decides who signs. A text extractor returns the labels
        with every relationship destroyed:

            Commercial Dept. / End / QAM No / COO / QAM Yes / Submit / Close

        which is a bag of words, not a workflow. So these go to a vision model
        that reads topology.

WHY THE OLD RULE NO LONGER WORKS
--------------------------------
`workflow_extractor.classify` routed on "no page has extractable text". That
was correct when the only workflow in the corpus was `CS Signature Matrix.pdf`,
3 pages of pure image. It is wrong for `Workflows/`: 136 of those 139 diagrams
carry a native text layer, so the old rule files every one of them as a process
document, where the chunker finds no headings, falls back to windowing, and
emits a chunk of disconnected labels that reads like content and answers
nothing.

THE RULE, AND WHAT IT IS MEASURED ON
------------------------------------
Two orthogonal signals, measured across the whole corpus (139 workflows in
`Workflows/`, 71 process documents in `processes_pdf/`):

    signal                  workflows            processes
    ISO section headings    137 have 0, 2 have 1  all >= 1, median 7
    page count              all <= 3              all >= 4

Either one separates the corpus on its own today. Both are used anyway, and
neither is load-bearing alone, because a signal that happens to be perfect on
the current corpus is exactly what this repo has been burned by before - the
chunker's predecessor matched literal heading strings, worked on the 9 docs it
was written against, and failed silently on everything else.

So: a document is a PROCESS if it has >= MIN_ISO_SECTIONS canonical headings,
OR >= MIN_PROCESS_PAGES pages. Everything else is a workflow. The thresholds sit
in the measured gap rather than on its edge - 2 sections against a workflow
maximum of 1, and 4 pages against a workflow maximum of 3.

Disagreement between the two signals is not silently resolved. `classify_one`
returns the evidence, and `report()` lists every file where the signals point
different ways, because that file is either a template outlier or a new
document shape - both worth a human look before 139 vision-model calls.

A PDF with no extractable text on any page is a workflow regardless, checked
first: that is the original scanned-diagram case, and page count means nothing
when the chunker cannot read the pages anyway.

Usage:
    python document_router.py                       # classify both default dirs
    python document_router.py --src Workflows processes_pdf
    python document_router.py --json routing.json   # machine-readable report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DIRS = [HERE.parent / "processes_pdf", HERE.parent / "Workflows"]

PROCESS = "process"
WORKFLOW = "workflow"

# Headings from the house ISO template. Matched as substrings of upper-cased
# page text, so wording drift inside a heading ("PROCESS OPERATIONS") still
# hits. This is a *detector*, not the chunker's taxonomy - it only has to
# answer "was this written to the process template at all", so it deliberately
# does not need adaptive_chunker.CANONICAL_SECTIONS' full list or its labeler.
ISO_SECTION_MARKERS = (
    "OBJECTIVE",
    "STAKEHOLDER",
    "TERMS AND DEFINITIONS",
    "RELATED DOCUMENTED",
    "PROCESS INPUT",
    "PROCESS OPERATION",
    "PROCESS OUTPUT",
    "PROCESS CONTROL",
    "PERFORMANCE MEASURES",
    "DOCUMENT CHANGE",
)

MIN_ISO_SECTIONS = 2    # workflows top out at 1
MIN_PROCESS_PAGES = 4   # workflows top out at 3
MIN_CHARS_PER_PAGE = 20  # below this a page carries no real text

# Directory names that are parking, not corpus. `processes_pdf/other/` holds
# the forms, signature matrices, flowcharts and annexes split out of the
# process folder - real documents, but not part of either extraction path
# today. Skipped by name so pointing --src at a parent never silently drags
# them back in; run them explicitly with `--src processes_pdf/other`.
EXCLUDE_DIRS = {"other"}


def page_text_lengths(pdf: Path) -> list[int]:
    import pymupdf

    with pymupdf.open(pdf) as doc:
        return [len(p.get_text().strip()) for p in doc]


def classify_one(pdf: Path) -> dict:
    """Classify one PDF and return the evidence behind the verdict.

    The evidence is returned rather than logged because the interesting cases
    are the ones where the two signals disagree, and a bare label cannot tell
    you that happened.
    """
    import pymupdf

    with pymupdf.open(pdf) as doc:
        pages = len(doc)
        texts = [p.get_text() for p in doc]

    full = "\n".join(texts).upper()
    iso = sum(1 for m in ISO_SECTION_MARKERS if m in full)
    scanned = all(len(t.strip()) < MIN_CHARS_PER_PAGE for t in texts)

    by_iso = iso >= MIN_ISO_SECTIONS
    by_pages = pages >= MIN_PROCESS_PAGES

    if scanned:
        # No readable text anywhere: the chunker is structurally blind to this
        # file, so page count is not evidence of anything.
        kind, why = WORKFLOW, "no extractable text on any page"
    elif by_iso or by_pages:
        kind = PROCESS
        why = " and ".join(
            ([f"{iso} ISO sections"] if by_iso else []) + ([f"{pages} pages"] if by_pages else [])
        )
    else:
        kind, why = WORKFLOW, f"{iso} ISO sections, {pages} page(s)"

    return {
        "path": pdf.as_posix(),
        "filename": pdf.name,
        "kind": kind,
        "why": why,
        "pages": pages,
        "iso_sections": iso,
        "chars": len(full.strip()),
        "scanned": scanned,
        # True when one signal says process and the other says workflow. Not an
        # error - just the file most worth looking at by hand.
        "signals_disagree": (by_iso != by_pages) and not scanned,
    }


def classify(pdf: Path) -> str:
    """'process' or 'workflow'. The one-line form, for callers that only route."""
    return classify_one(Path(pdf))["kind"]


def classify_dir(src: Path, recursive: bool = True) -> list[dict]:
    """Classify every PDF under a directory.

    Recursive by default: `Workflows/overseas/` holds 29 diagrams that a
    non-recursive glob silently omits. Case-insensitive on the extension,
    because the corpus contains both `.pdf` and `.PDF`. Anything under a
    directory named in EXCLUDE_DIRS is skipped unless it is `src` itself.
    """
    src = Path(src)
    pattern = "**/*" if recursive else "*"
    pdfs = sorted(
        p for p in src.glob(pattern)
        if p.suffix.lower() == ".pdf"
        and not (set(p.relative_to(src).parts[:-1]) & EXCLUDE_DIRS)
    )
    return [classify_one(p) for p in pdfs]


def report(records: list[dict]) -> dict:
    by_kind: dict[str, list[dict]] = {PROCESS: [], WORKFLOW: []}
    for r in records:
        by_kind[r["kind"]].append(r)
    return {
        "total": len(records),
        "process": len(by_kind[PROCESS]),
        "workflow": len(by_kind[WORKFLOW]),
        "scanned_workflows": sum(1 for r in by_kind[WORKFLOW] if r["scanned"]),
        "native_workflows": sum(1 for r in by_kind[WORKFLOW] if not r["scanned"]),
        "signals_disagree": [r for r in records if r["signals_disagree"]],
        "records": records,
    }


def print_report(rep: dict) -> None:
    print(f"\n{'=' * 74}\nDOCUMENT ROUTING — {rep['total']} PDFs\n{'=' * 74}")
    print(f"process  -> adaptive_chunker.py   : {rep['process']}")
    print(f"workflow -> workflow_extractor.py : {rep['workflow']}"
          f"  ({rep['native_workflows']} native text, {rep['scanned_workflows']} scanned)")
    dis = rep["signals_disagree"]
    print(f"signals disagree                  : {len(dis)}")
    for r in dis:
        print(f"    {r['filename'][:52]:<52} {r['kind']:<8} "
              f"pages={r['pages']} iso={r['iso_sections']}")
    if not dis:
        print("    (none — both signals agree on every file)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Route PDFs to their extractor")
    ap.add_argument("--src", type=Path, nargs="+", default=DEFAULT_DIRS)
    ap.add_argument("--json", type=Path, default=None,
                    help="write the full per-file report here")
    ap.add_argument("--list", choices=[PROCESS, WORKFLOW], default=None,
                    help="print just the filenames of one kind")
    args = ap.parse_args()

    records = []
    for src in args.src:
        if not src.exists():
            raise SystemExit(f"No such directory: {src}")
        found = classify_dir(src)
        print(f"{src.as_posix()}: {len(found)} PDFs")
        records += found

    rep = report(records)
    if args.list:
        for r in rep["records"]:
            if r["kind"] == args.list:
                print(r["path"])
        return

    print_report(rep)
    if args.json:
        args.json.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()

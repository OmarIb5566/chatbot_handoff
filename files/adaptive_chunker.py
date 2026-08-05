"""
Template-independent chunking: structural boundary detection + semantic labeling.

This is the pipeline's chunker. It replaced an earlier `chunker.py` that found
sections by matching literal strings from a hand-written SECTION_HEADERS list.
That worked on the 9 docs it was written against and failed silently on
anything else: a doc saying "RESPONSIBLE PARTIES" instead of "STAKEHOLDER"
matched nothing, produced one mega-chunk, and reported success. At 500-700 docs
from mixed templates that is not a viable design.

The fix is to decouple the two jobs that literal list was doing at once:

  BOUNDARY DETECTION  - *where* does a section start?  Structural, not lexical.
                        A heading is a short line that is visually distinct
                        from body text (bigger font, bold, or both). PyMuPDF's
                        `page.get_text("dict")` exposes per-span size and flags,
                        and that signal holds regardless of what word is on the
                        line. Where layout is flat (poor scans, OCR, or docs
                        like P-CN-01 that style headings identically to body
                        text) we fall back to the numbering pattern
                        "1. SOMETHING" / "6.1 SOMETHING", which is orthogonal
                        to both font and wording.

  LABELING            - *what kind* of section is it?  Semantic, not literal.
                        Embed the heading text and compare it against a
                        canonical taxonomy by cosine similarity, reusing the
                        MiniLM model retriever.py already loads. This absorbs
                        wording drift a literal dictionary never will.

Both stages are independently auditable, which is the third design goal: at
this corpus size you cannot eyeball per-file chunking quality, so every doc
carries a record of which detector fired, which headings failed to match a
canonical bucket, and whether it fell back to windowed chunking. The summary
prints on every run; `--audit PATH` writes the full per-document report.

--- Why labeling is three tiers and not just cosine ---

Measured on the real headings in this corpus, embedding-only labeling has one
hard failure. "STAKHOLDER" - a typo that appears in 3 of the 9 documents -
scores 0.130 against canonical "STAKEHOLDER", barely ahead of the unrelated
"TERMS AND DEFINITIONS" at 0.128. MiniLM's tokenizer shatters the misspelling
and the embedding carries almost no signal. Any threshold high enough to
reject junk (see below) rejects that real heading too.

Character similarity handles exactly that case (difflib puts STAKHOLDER vs
STAKEHOLDER at 0.95) and embeddings handle the synonym case that character
similarity cannot ("REVISION HISTORY" -> DOCUMENT CHANGE HISTORY, 0.64). They
fail in opposite directions, so the labeler tries exact, then fuzzy, then
cosine, and records which tier answered.

The cosine threshold is 0.45, from the measured gap between genuine wording
drift and non-headings:

    drift    RESPONSIBLE PARTIES -> LIABLE STAKEHOLDER      0.556
             DEFINITIONS AND ABBREVIATIONS -> TERMS...      0.686
             REVISION HISTORY -> DOCUMENT CHANGE HISTORY    0.641
             KPIs -> PERFORMANCE MEASURES                   0.497
    junk     "5 Million)"                                   0.238
             "8. Others (Tender clarifications, etc.)"      0.290

Anything below the threshold is labeled UNMATCHED, kept as a chunk with its
raw heading intact, and surfaced in the audit report. Unmatched is a flag for
a human, never a discard.

--- Its place in the pipeline ---

    PDFs -> adaptive_chunker.py -> chunks.json -> retriever.py

Layout signals (font size, bold flags) exist only in the PDF, not in extracted
text, so this reads the PDFs directly rather than an intermediate JSON. Two
consequences worth knowing:

  * `extract_pipeline.py` is not upstream of chunking. It stays as the
    OCR-health audit, and that role matters more, not less: this chunker
    sees only native text, so a page with no embedded text is a page it is
    blind to. `empty_pages` in the audit record is that signal - it lists
    pages that yielded zero lines. On this corpus that is page 1 of every
    document (the image-only signature cover page), which carries no process
    content.

  * There is no OCR fallback here. If a future batch arrives genuinely
    scanned, `empty_pages` will show it as a block of missing pages rather
    than as quietly incomplete chunks, and OCR would need wiring in at that
    point.

Usage:
    python adaptive_chunker.py                     # ../processes_pdf -> chunks.json
    python adaptive_chunker.py --src ../other_pdfs --out other_chunks.json
    python adaptive_chunker.py --audit audit.json  # also write the full report
    python adaptive_chunker.py --no-embed          # exact+fuzzy only, skips loading MiniLM
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

# Match retriever.py: use the locally cached MiniLM, never reach out to the Hub.
# Set before sentence_transformers is imported anywhere in the process.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import fitz  # PyMuPDF

HERE = Path(__file__).resolve().parent
DEFAULT_SRC = HERE.parent / "processes_pdf"
DEFAULT_OUT = HERE / "chunks.json"

# --- boundary detection thresholds ------------------------------------------
SIZE_RATIO_THRESHOLD = 1.15   # "bigger than body text" (vs the median line size)
MAX_HEADING_CHARS = 80        # headings are short; body paragraphs are not
BOLD_FLAG_BIT = 2 ** 4        # PyMuPDF span flags: bit 4 = bold
RUNNING_HEADER_PAGE_FRACTION = 0.4  # same line on >=40% of pages = template furniture
MIN_PAGES_FOR_HEADER_FILTER = 3     # below this, "repeats across pages" is meaningless
MAX_LIST_GAP = 2              # line-index gap below which numbered candidates are a list
MIN_LIST_RUN = 3              # this many tight consecutive numbers = an inline list
MIN_HEADINGS_FOR_STRUCTURE = 3  # fewer than this and we do not trust the structure

LONE_NUMBER_RE = re.compile(r"^\d{1,2}(?:\.\d{1,2})?\.?\s*[̶\-–—]?\s*$")
NUMBERED_HEADING_RE = re.compile(
    r"^(?P<number>\d{1,2}(?:\.\d{1,2}){0,3}\.?)\s+(?P<title>[A-Z][A-Za-z].{0,78})$"
)
LEADING_NUMBER_RE = re.compile(r"^\s*\d{1,2}(?:\.\d{1,2}){0,3}\.?\s*")

# --- chunking thresholds ----------------------------------------------------
MIN_CHUNK_CHARS = 30          # drop headings with no body of their own
WINDOW_CHARS = 1200           # fallback window size
WINDOW_OVERLAP = 150          # fallback overlap, so a fact is never cut in half

# --- labeling ---------------------------------------------------------------
# The canonical taxonomy. Adding a bucket here is the intended way to extend
# this to a new document family - it costs one string, not a regex.
CANONICAL_SECTIONS = [
    "OBJECTIVES",
    "STAKEHOLDER",
    "LIABLE STAKEHOLDER",
    "TERMS AND DEFINITIONS",
    "RELATED DOCUMENTED INFORMATION",
    "PROCESS DETAILS",
    "PROCESS INPUT",
    "PROCESS OPERATION",
    "PROCESS OUTPUT",
    "PROCESS RISK ASSESSMENT",
    "PROCESS CONTROL",
    "PERFORMANCE MEASURES",
    "DOCUMENT CHANGE HISTORY",
]
FUZZY_THRESHOLD = 0.85    # difflib ratio; catches OCR/typo drift
COSINE_THRESHOLD = 0.45   # MiniLM cosine; catches wording drift
UNMATCHED = "UNMATCHED"

# A section this common across the corpus is part of the house template, so a
# doc lacking it is worth a look. Used only for audit reporting.
COMMON_SECTION_FRACTION = 0.6

# Document metadata regexes. doc_code and title are what answers cite, so these
# match the document control block rather than being inferred from the filename.
DOC_ID_RE = re.compile(r"Doc\.?\s*No\.?:?\s*([A-Z\-\d]+)", re.IGNORECASE)
DOC_TITLE_RE = re.compile(r"^(.*?)(?:\r?\n)Effective Date", re.MULTILINE)


# ============================================================ layout extraction
def extract_layout_lines(pdf_path: Path) -> tuple[list[dict], int]:
    """Every text line in the PDF with the layout signals we detect on.

    Returns (lines, page_count). Each line carries its reading-order index, so
    downstream stages can slice body text between two headings without
    re-deriving position.
    """
    lines: list[dict] = []
    with fitz.open(pdf_path) as doc:
        page_count = len(doc)
        for page_num, page in enumerate(doc, start=1):
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    text = "".join(s["text"] for s in spans).strip()
                    if not text:
                        continue
                    lines.append({
                        "idx": len(lines),
                        "page": page_num,
                        "text": text,
                        "size": round(max(s["size"] for s in spans), 1),
                        "bold": any(s["flags"] & BOLD_FLAG_BIT for s in spans),
                    })
    return lines, page_count


def _repetition_key(text: str) -> str:
    """Normalise digits so "Page 2 of 6" and "Page 3 of 6" collapse to one key."""
    return re.sub(r"\d+", "#", text)


def find_running_headers(lines: list[dict], page_count: int,
                         min_fraction: float = RUNNING_HEADER_PAGE_FRACTION) -> set[str]:
    """Header/footer furniture, identified by repeating across pages.

    Content-based rather than position-based: a y-coordinate band assumes a
    consistent page geometry that a mixed-template corpus will not have.

    Disabled below MIN_PAGES_FOR_HEADER_FILTER pages. On a 1-page document
    every line trivially appears on 100% of pages, so the fraction test
    classifies the entire document as furniture and no heading survives - the
    doc then silently drops to windowed fallback. Repetition is only evidence
    of a template when there are enough pages for it to repeat *across*.
    """
    if page_count < MIN_PAGES_FOR_HEADER_FILTER:
        return set()

    pages_per_key: dict[str, set[int]] = {}
    for line in lines:
        pages_per_key.setdefault(_repetition_key(line["text"]), set()).add(line["page"])
    running = {k for k, pages in pages_per_key.items()
               if len(pages) >= 2 and len(pages) / max(page_count, 1) >= min_fraction}
    return {l["text"] for l in lines if _repetition_key(l["text"]) in running}


# ========================================================== boundary detection
def _candidates(lines: list[dict], running_headers: set[str]) -> list[dict]:
    """Flag every line that looks structurally like a heading."""
    body_size = statistics.median(l["size"] for l in lines)
    out = []
    for line in lines:
        if line["text"] in running_headers:
            continue
        text = line["text"]
        short = len(text) <= MAX_HEADING_CHARS
        bigger = line["size"] >= body_size * SIZE_RATIO_THRESHOLD
        numbered = NUMBERED_HEADING_RE.match(text)
        lone_number = bool(LONE_NUMBER_RE.match(text))

        if bigger and short:
            detector = "font_size"
        elif line["bold"] and lone_number:
            detector = "lone_number"
        elif short and numbered:
            # Deliberately NOT gated behind bold/size: some docs (P-CN-01,
            # P-VMO-02) style real headings identically to body text, and the
            # numbering pattern is the only usable signal there.
            detector = "numbering"
        else:
            continue

        out.append({**line, "detector": detector,
                    "number": numbered.group("number").rstrip(".") if numbered else None,
                    "plain": not line["bold"] and not bigger})
    return out


def _demote_inline_lists(cands: list[dict]) -> set[int]:
    """Drop inline numbered lists that look like headings but are body content.

    P-VMO-02 embeds "1. Draft contract / 2. Draft Bonds / ... / 8. Others"
    inside a process step. Every one matches the numbering pattern. What
    separates them from real headings is *density*: sections have body text
    between them, list items do not. So a run of >=3 visually-plain numbered
    candidates packed within a couple of lines of each other, incrementing by
    one, is a list rather than eight consecutive empty sections.

    Only visually-plain candidates are eligible - a bold or larger heading is
    never demoted, which is what keeps genuinely adjacent headings like
    "6. PROCESS DETAILS" / "6.1. PROCESS INPUT" intact.
    """
    demoted: set[int] = set()
    run: list[dict] = []

    def flush():
        if len(run) >= MIN_LIST_RUN:
            demoted.update(c["idx"] for c in run)

    for cand in cands:
        eligible = cand["plain"] and cand["detector"] == "numbering" and cand["number"]
        if eligible and run:
            prev = run[-1]
            try:
                consecutive = int(cand["number"]) == int(prev["number"]) + 1
            except ValueError:          # sub-numbered ("6.1") - not a flat list
                consecutive = False
            if cand["idx"] - prev["idx"] <= MAX_LIST_GAP and consecutive:
                run.append(cand)
                continue
        flush()
        run = [cand] if eligible else []
    flush()
    return demoted


def _demote_duplicate_numbers(cands: list[dict], already: set[int]) -> set[int]:
    """Drop a plain candidate reusing a section number already claimed.

    P-TN-02 wraps a line mid-sentence so "...up to 5 Million)" starts a line and
    matches the numbering pattern as number=5, title="Million)". Section 5 was
    already claimed by RELATED DOCUMENTED INFORMATION two lines earlier. Real
    templates do not number two sections the same, so a plain duplicate is a
    wrapped body line.
    """
    demoted: set[int] = set()
    seen: set[str] = set()
    for cand in cands:
        if cand["idx"] in already or not cand["number"]:
            continue
        if cand["number"] in seen and cand["plain"]:
            demoted.add(cand["idx"])
            continue
        seen.add(cand["number"])
    return demoted


def _stitch_lone_numbers(cands: list[dict], lines: list[dict],
                         demoted: set[int]) -> list[dict]:
    """Merge a bold lone number with the label on the following line.

    Some docs put "5.2" and "PROCESS OPERATIONS" on separate lines. The number
    alone is not a usable section label, and the text alone is not detectable.
    """
    cand_idx = {c["idx"] for c in cands} - demoted
    headings, consumed = [], set()
    for cand in cands:
        if cand["idx"] in demoted or cand["idx"] in consumed:
            continue
        if not LONE_NUMBER_RE.match(cand["text"]):
            headings.append(cand)
            continue
        nxt = lines[cand["idx"] + 1] if cand["idx"] + 1 < len(lines) else None
        if (nxt and nxt["page"] == cand["page"] and nxt["idx"] not in cand_idx
                and len(nxt["text"]) <= MAX_HEADING_CHARS):
            number = cand["text"].rstrip(". ̶-–—")
            headings.append({**cand,
                             "text": f"{number}. {nxt['text']}",
                             "detector": "lone_number+stitch"})
            consumed.add(nxt["idx"])
        else:
            headings.append(cand)
    return [h for h in headings if h["idx"] not in consumed]


def detect_headings(lines: list[dict], page_count: int) -> tuple[list[dict], dict]:
    """Structural heading detection. Returns (headings, detection stats)."""
    if not lines:
        return [], {"candidates": 0, "demoted_list": 0, "demoted_duplicate": 0}

    running = find_running_headers(lines, page_count)
    cands = _candidates(lines, running)
    demoted_list = _demote_inline_lists(cands)
    demoted_dupe = _demote_duplicate_numbers(cands, demoted_list)
    demoted = demoted_list | demoted_dupe
    headings = _stitch_lone_numbers(cands, lines, demoted)

    stats = {
        "candidates": len(cands),
        "demoted_list": len(demoted_list),
        "demoted_duplicate": len(demoted_dupe),
        "demoted_text": [c["text"] for c in cands if c["idx"] in demoted],
        "detectors": dict(Counter(h["detector"] for h in headings)),
        "running_headers": len(running),
    }
    return headings, stats


# ==================================================================== labeling
class SectionLabeler:
    """Map a raw heading string onto the canonical taxonomy.

    Three tiers, tried in order, each recording how it answered:
      exact  - normalised string equality
      fuzzy  - difflib ratio >= FUZZY_THRESHOLD (OCR damage, typos, plurals)
      embed  - MiniLM cosine >= COSINE_THRESHOLD (genuine wording drift)
    Nothing above threshold gives UNMATCHED, which is a flag, not a discard.

    `use_embeddings=False` skips loading MiniLM entirely, for when you want to
    inspect boundary detection without paying the model load.
    """

    def __init__(self, canonical: list[str] = None, use_embeddings: bool = True,
                 model_name: str = "all-MiniLM-L6-v2"):
        self.canonical = list(canonical or CANONICAL_SECTIONS)
        self.use_embeddings = use_embeddings
        self.model_name = model_name
        self._norm = {self._normalise(c): c for c in self.canonical}
        self.model = None
        self._canon_emb = None
        if use_embeddings:
            from sentence_transformers import SentenceTransformer
            # local_files_only mirrors retriever.py: the model is already cached.
            self.model = SentenceTransformer(model_name, local_files_only=True)
            self._canon_emb = self.model.encode(
                self.canonical, normalize_embeddings=True, convert_to_numpy=True)

    @staticmethod
    def _normalise(text: str) -> str:
        text = LEADING_NUMBER_RE.sub("", text)
        return re.sub(r"[^A-Z ]+", "", text.upper()).strip()

    def label(self, heading: str) -> dict:
        key = self._normalise(heading)
        if not key:
            return {"section": UNMATCHED, "method": "empty", "score": 0.0}

        if key in self._norm:
            return {"section": self._norm[key], "method": "exact", "score": 1.0}

        best, best_score = None, 0.0
        for norm_key, canon in self._norm.items():
            ratio = difflib.SequenceMatcher(None, key, norm_key).ratio()
            if ratio > best_score:
                best, best_score = canon, ratio
        if best_score >= FUZZY_THRESHOLD:
            return {"section": best, "method": "fuzzy", "score": round(best_score, 3)}

        if self.use_embeddings:
            emb = self.model.encode([key], normalize_embeddings=True, convert_to_numpy=True)
            sims = (self._canon_emb @ emb[0])
            i = int(sims.argmax())
            if float(sims[i]) >= COSINE_THRESHOLD:
                return {"section": self.canonical[i], "method": "embed",
                        "score": round(float(sims[i]), 3)}
            return {"section": UNMATCHED, "method": "below_threshold",
                    "score": round(float(sims[i]), 3),
                    "nearest": self.canonical[i]}

        return {"section": UNMATCHED, "method": "below_threshold",
                "score": round(best_score, 3), "nearest": best}


# ==================================================================== chunking
def _window_text(text: str, size: int = WINDOW_CHARS,
                 overlap: int = WINDOW_OVERLAP) -> list[str]:
    """Paragraph-greedy windows with overlap, for docs with no usable structure.

    Packs whole paragraphs up to `size` so a window rarely splits mid-sentence,
    and carries `overlap` characters forward so a fact spanning a boundary
    survives in at least one window.

    The unit-splitting below is deliberately belt-and-braces. Layout extraction
    joins lines with a single newline, so a PDF can contain no blank lines at
    all - in which case a blank-line split returns the whole document as one
    "paragraph" and the packer emits it unsplit. That reproduces the exact
    mega-chunk this fallback exists to prevent, so each stage re-splits
    anything still over `size`: paragraphs -> lines -> hard slice.
    """
    units = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    if any(len(u) > size for u in units):    # no usable blank lines: split on lines
        relined = []
        for unit in units:
            relined.extend([l.strip() for l in unit.split("\n") if l.strip()]
                           if len(unit) > size else [unit])
        units = relined

    paragraphs = []                          # a single line can still be huge
    for unit in units:
        while len(unit) > size:
            paragraphs.append(unit[:size])
            unit = unit[size - overlap:] if overlap else unit[size:]
        if unit:
            paragraphs.append(unit)

    windows, current = [], ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > size:
            windows.append(current.strip())
            current = (current[-overlap:] + "\n\n" + para) if overlap else para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current.strip():
        windows.append(current.strip())
    return windows


def _doc_metadata(full_text: str, filename: str) -> dict:
    code = DOC_ID_RE.search(full_text)
    title = DOC_TITLE_RE.search(full_text)
    return {
        "filename": filename,
        "doc_code": code.group(1) if code else None,
        "title": title.group(1).strip() if title else filename,
    }


def normalize_filename(pdf_path: Path) -> str:
    """Same mapping as extract_pipeline.normalize_filename.

    Load-bearing: eval_set.json matches retrieved chunks against its `doc`
    field by exact string, so this must stay identical or the eval silently
    stops matching anything.
    """
    stem = re.sub(r"[-()]", "", pdf_path.stem)
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem)
    return f"{re.sub(r'_+', '_', stem).strip('_')}.pdf"


def chunk_pdf(pdf_path: Path, labeler: SectionLabeler | None = None) -> tuple[list[dict], dict]:
    """Chunk one PDF. Returns (chunks, audit record).

    Chunk schema, consumed as-is by the retriever and validator:
        filename, doc_code, title, section, step, text
    plus raw_heading / label_method / label_score / page / detector for audit.
    """
    lines, page_count = extract_layout_lines(pdf_path)
    filename = normalize_filename(pdf_path)
    full_text = "\n".join(l["text"] for l in lines)
    meta = _doc_metadata(full_text, filename)

    # Pages that yielded no native text at all. This chunker has no OCR path,
    # so these are pages it is structurally blind to - the audit has to say so
    # rather than let them vanish into a lower chunk count.
    seen_pages = {l["page"] for l in lines}
    empty_pages = [p for p in range(1, page_count + 1) if p not in seen_pages]

    audit = {
        "filename": filename,
        "source_file": pdf_path.name,
        "pages": page_count,
        "empty_pages": empty_pages,
        "lines": len(lines),
        "chars": len(full_text),
        "mode": "structural",
        "headings": 0,
        "unmatched": [],
        "label_methods": {},
        "fallback_reason": None,
    }

    headings, stats = detect_headings(lines, page_count)
    audit.update({"headings": len(headings), "detection": stats})

    # --- graceful fallback: no usable structure -> windowed, never one mega-chunk
    if len(headings) < MIN_HEADINGS_FOR_STRUCTURE:
        audit["mode"] = "windowed_fallback"
        audit["fallback_reason"] = (
            f"only {len(headings)} heading(s) detected, need {MIN_HEADINGS_FOR_STRUCTURE}"
        )
        chunks = [
            {**meta, "section": UNMATCHED, "step": None, "text": w,
             "raw_heading": None, "label_method": "windowed", "label_score": 0.0,
             "page": None, "detector": "windowed_fallback", "window": i}
            for i, w in enumerate(_window_text(full_text))
            if len(w) >= MIN_CHUNK_CHARS
        ]
        audit["chunks"] = len(chunks)
        audit["chunk_chars"] = sum(len(c["text"]) for c in chunks)
        return chunks, audit

    labeler = labeler or SectionLabeler()
    bounds = [h["idx"] for h in headings] + [len(lines)]
    chunks = []

    # Front matter: anything before the first heading (cover page, doc control
    # block). Kept rather than dropped - it carries the doc code and title.
    if bounds[0] > 0:
        front = "\n".join(l["text"] for l in lines[:bounds[0]]).strip()
        if len(front) >= MIN_CHUNK_CHARS:
            chunks.append({**meta, "section": "FRONT MATTER", "step": None, "text": front,
                           "raw_heading": None, "label_method": "front_matter",
                           "label_score": 0.0, "page": 1, "detector": "front_matter"})

    for i, heading in enumerate(headings):
        body = "\n".join(l["text"] for l in lines[heading["idx"]:bounds[i + 1]]).strip()
        if len(body) < MIN_CHUNK_CHARS:
            continue
        lab = labeler.label(heading["text"])
        if lab["section"] == UNMATCHED:
            audit["unmatched"].append({"heading": heading["text"],
                                       "nearest": lab.get("nearest"),
                                       "score": lab["score"]})
        chunks.append({
            **meta,
            "section": lab["section"],
            "step": None,          # step-level splitting is a separate concern
            "text": body,
            "raw_heading": heading["text"],
            "label_method": lab["method"],
            "label_score": lab["score"],
            "page": heading["page"],
            "detector": heading["detector"],
        })

    audit["label_methods"] = dict(Counter(c["label_method"] for c in chunks))
    audit["chunks"] = len(chunks)
    audit["chunk_chars"] = sum(len(c["text"]) for c in chunks)
    audit["coverage"] = round(audit["chunk_chars"] / max(len(full_text), 1), 3)
    return chunks, audit


def chunk_corpus(src_dir: Path = DEFAULT_SRC,
                 labeler: SectionLabeler | None = None,
                 use_embeddings: bool = True) -> tuple[list[dict], dict]:
    """Chunk every PDF in a folder. Returns (chunks, corpus audit report)."""
    pdfs = sorted(Path(src_dir).glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {src_dir}")
    labeler = labeler or SectionLabeler(use_embeddings=use_embeddings)

    all_chunks, records = [], []
    for pdf in pdfs:
        chunks, audit = chunk_pdf(pdf, labeler)
        all_chunks.extend(chunks)
        records.append(audit)
    return all_chunks, build_audit_report(all_chunks, records)


def build_audit_report(chunks: list[dict], records: list[dict]) -> dict:
    """Corpus-level chunking health.

    This is the "audit, don't trust" surface. At 500-700 docs the per-file
    detail is unreadable, so the top-level counters are what you watch: docs
    that fell back, headings that matched nothing, and sections that went
    missing from a doc that otherwise looks fine.
    """
    fell_back = [r for r in records if r["mode"] == "windowed_fallback"]
    unmatched = [{"filename": r["filename"], **u} for r in records for u in r["unmatched"]]
    label_methods = Counter()
    for r in records:
        label_methods.update(r.get("label_methods", {}))

    seen_sections = Counter(c["section"] for c in chunks)

    # Template-outlier detection, self-calibrating against the corpus rather
    # than against CANONICAL_SECTIONS. A canonical bucket that no document
    # actually produces (PROCESS DETAILS is a container heading whose body is
    # entirely its own subsections, so it never yields a chunk) would otherwise
    # be reported missing from every doc - noise that teaches people to ignore
    # the report. What matters is a doc missing a section its *peers* have.
    by_doc = {}
    for c in chunks:
        by_doc.setdefault(c["filename"], set()).add(c["section"])
    n_docs = max(len(records), 1)
    common = [s for s, n in seen_sections.items()
              if n / n_docs >= COMMON_SECTION_FRACTION and s != UNMATCHED]
    per_doc_sections = {
        r["filename"]: [s for s in common if s not in by_doc.get(r["filename"], set())]
        for r in records
    }

    no_text = {r["filename"]: r["empty_pages"] for r in records if r.get("empty_pages")}

    return {
        "docs": len(records),
        "chunks": len(chunks),
        "pages_with_no_native_text": sum(len(v) for v in no_text.values()),
        "empty_pages_by_doc": no_text,
        "docs_windowed_fallback": len(fell_back),
        "fallback_files": [r["filename"] for r in fell_back],
        "unmatched_headings": unmatched,
        "label_methods": dict(label_methods),
        "sections_found": dict(seen_sections.most_common()),
        "missing_canonical_sections": {k: v for k, v in per_doc_sections.items() if v},
        "demoted_candidates": {r["filename"]: r["detection"]["demoted_text"]
                               for r in records
                               if r.get("detection", {}).get("demoted_text")},
        "per_doc": records,
    }


def print_audit(report: dict) -> None:
    print(f"\n{'=' * 74}\nCHUNKING AUDIT — {report['docs']} docs, {report['chunks']} chunks\n{'=' * 74}")
    print(f"windowed fallback : {report['docs_windowed_fallback']}/{report['docs']} docs"
          + (f"  {report['fallback_files']}" if report["fallback_files"] else "  (none)"))
    n_empty = report["pages_with_no_native_text"]
    print(f"pages with no text: {n_empty}"
          + ("  (no OCR path here - these pages are not chunked)" if n_empty else ""))
    print(f"label methods     : {report['label_methods']}")
    print(f"unmatched headings: {len(report['unmatched_headings'])}")
    for u in report["unmatched_headings"]:
        print(f"    {u['filename'][:38]:<38} {u['heading'][:30]:<30} "
              f"nearest={u.get('nearest')} ({u['score']})")
    if report["missing_canonical_sections"]:
        print("docs missing canonical sections (template outliers):")
        for fn, missing in report["missing_canonical_sections"].items():
            print(f"    {fn[:44]:<44} missing {len(missing)}: {', '.join(missing[:4])}"
                  + (" ..." if len(missing) > 4 else ""))
    if report["demoted_candidates"]:
        n = sum(len(v) for v in report["demoted_candidates"].values())
        print(f"demoted false headings: {n} (inline lists / wrapped body lines)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Adaptive structural chunker")
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--audit", type=Path, default=None,
                    help="also write the full audit report to this path (JSON). "
                         "The summary always prints; the file is opt-in so a "
                         "routine re-chunk does not drop an artifact nobody reads.")
    ap.add_argument("--no-embed", action="store_true",
                    help="exact+fuzzy labeling only; skips loading MiniLM")
    args = ap.parse_args()

    chunks, report = chunk_corpus(args.src.resolve(), use_embeddings=not args.no_embed)
    args.out.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print_audit(report)
    print(f"\nWrote {args.out}")
    if args.audit:
        args.audit.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {args.audit}")


if __name__ == "__main__":
    main()

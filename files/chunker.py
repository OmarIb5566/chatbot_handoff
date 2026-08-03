"""
Chunk process documents by logical unit rather than fixed token windows.

Why: these docs are ISO-style tables (step -> description -> responsibility ->
form number). A fixed-size chunker will happily cut a step away from its own
form number. We instead chunk by:
  1. Named sections (OBJECTIVES, STAKEHOLDER, TERMS AND DEFINITION, etc.)
  2. Numbered process-cycle steps within "PROCESS OPERATIONS" sections

Every chunk carries doc_id, doc_title, section, and step_number metadata so
retrieval results can be filtered/boosted and so answers can cite exactly
where they came from.
"""
import re, json
from pathlib import Path

# All paths resolve relative to this file (the "files" folder) so the module
# works regardless of the caller's working directory.
HERE = Path(__file__).resolve().parent
EXTRACTED_JSON = HERE / "extracted_raw.json"
CHUNKS_JSON = HERE / "chunks.json"

SECTION_HEADERS = [
    "OBJECTIVES", "STAKEHOLDER", "LIABLE STAKEHOLDER", "TERMS AND DEFINITION",
    "TERMS AND DEFINITIONS", "RELATED DOCUMENTED INFORMATION", "PROCESS DETAILS",
    "PROCESS INPUT", "PROCESS OPERATION", "PROCESS OPERATIONS", "PROCESS OUTPUT",
    "PROCESS RISK ASSESSMENT", "PROCESS RISK ASSESMENT", "PROCESS CONTROL",
    "PERFORMANCE MEASURE", "PERFORMANCE MEASURES", "DOCUMENT CHANGE HISTORY",
]

DOC_ID_RE = re.compile(r"Doc\.?\s*No\.?:?\s*([A-Z\-\d]+)", re.IGNORECASE)
DOC_TITLE_RE = re.compile(r"^(.*?)(?:\r?\n)Effective Date", re.MULTILINE)
STEP_RE = re.compile(r"(?m)^(\d{1,2})\s*$")  # a lone number on its own line = step marker in these tables


def full_text_for_doc(pages: list[dict]) -> str:
    return "\n".join(p["text"] for p in pages if p.get("text"))


def extract_doc_metadata(text: str, filename: str) -> dict:
    doc_id_match = DOC_ID_RE.search(text)
    title_match = DOC_TITLE_RE.search(text)
    return {
        "filename": filename,
        "doc_code": doc_id_match.group(1) if doc_id_match else None,
        "title": title_match.group(1).strip() if title_match else filename,
    }


def chunk_document(pages: list[dict], filename: str) -> list[dict]:
    """Return a list of chunk dicts: {doc_code, title, filename, section, text}."""
    text = full_text_for_doc(pages)
    meta = extract_doc_metadata(text, filename)

    # Split on section headers, keep header with following content
    pattern = "|".join(re.escape(h) for h in sorted(SECTION_HEADERS, key=len, reverse=True))
    splits = re.split(f"(?=\\n?\\d?\\.?\\s*(?:{pattern}))", text)

    chunks = []
    for seg in splits:
        seg = seg.strip()
        if len(seg) < 30:
            continue
        header_match = re.search(pattern, seg[:60])
        section_name = header_match.group(0) if header_match else "GENERAL"

        # Within PROCESS OPERATION sections, further split by numbered step
        if "PROCESS OPERATION" in section_name.upper():
            step_chunks = re.split(r"(?m)^(\d{1,2})\n", seg)
            # step_chunks alternates [preamble, num, body, num, body, ...]
            if len(step_chunks) > 1:
                preamble = step_chunks[0]
                if len(preamble.strip()) > 20:
                    chunks.append({**meta, "section": section_name, "step": None, "text": preamble.strip()})
                for i in range(1, len(step_chunks) - 1, 2):
                    step_num, body = step_chunks[i], step_chunks[i + 1]
                    chunks.append({**meta, "section": section_name, "step": step_num, "text": body.strip()})
                continue

        chunks.append({**meta, "section": section_name, "step": None, "text": seg})

    return chunks


def iter_chunks(extracted_json_path: str | Path = EXTRACTED_JSON):
    """Yield chunks one document at a time.

    Scaling note: at 9 docs a single json.load is fine. At 500-600 docs the
    whole-corpus dict is the thing that hurts, so callers that only need to
    stream chunks into an index (see build_index in retriever.py) should use
    this generator and never hold the full chunk list. `chunk_all` stays as
    the eager convenience wrapper for the notebook.

    For a corpus too large for even one json.load, run the extractor to
    sharded JSON per doc-code prefix and call this once per shard; the chunk
    schema is per-document, so nothing here needs cross-document state.
    """
    data = json.load(open(extracted_json_path, encoding="utf-8"))
    for filename, pages in data.items():
        yield from chunk_document(pages, filename)


def chunk_all(extracted_json_path: str | Path = EXTRACTED_JSON) -> list[dict]:
    return list(iter_chunks(extracted_json_path))


# --- doc-code prefix routing -------------------------------------------------
# Pre-retrieval narrowing for the 500-600 doc corpus: if a query clearly names
# a process family, search only that family instead of the whole index.
DOC_PREFIXES = ["PSE", "PQD", "PCM", "PTN", "PVMO", "PCN"]

# Maps a doc-code prefix to the words that should route a query to it.
PREFIX_HINTS = {
    "PSE":  ["self-execution", "self execution", "sem", "selfexecution"],
    "PQD":  ["quality", "inspection", "testing", "itp", "pqp", "qc", "hold point"],
    "PCM":  ["customer", "satisfaction", "branding", "marketing", "survey", "commercial"],
    "PTN":  ["tender", "tendering", "initiation", "launching", "bid submission", "loa", "handover"],
    "PVMO": ["vendor", "procurement", "purchase", "rfq", "quotation", "sourcing", "supplier"],
    "PCN":  ["subcontract", "subcontractor", "contract", "agreement", "btb", "back-to-back"],
}


def doc_code_prefix(chunk: dict) -> str | None:
    """'P-VMO-01' -> 'PVMO'. Falls back to the filename when doc_code is missing."""
    code = chunk.get("doc_code") or chunk.get("filename", "")
    m = re.match(r"P-?([A-Z]{2,3})-?\d", code.upper())
    return f"P{m.group(1)}" if m else None


def route_prefixes(query: str) -> list[str]:
    """Return the doc-code prefixes a query should be restricted to.

    Empty list means "no confident route, search everything" - that is the
    safe default, since a wrong route is an unrecoverable miss while a
    missing route only costs latency.
    """
    q = query.lower()
    # An explicit code in the query wins outright: "what is P-VMO-01" or "F-P-CN-01-11"
    explicit = {f"P{m}" for m in re.findall(r"\bF?-?P-([A-Z]{2,3})-\d", query.upper())}
    if explicit:
        return sorted(explicit & set(DOC_PREFIXES))
    hits = [p for p, words in PREFIX_HINTS.items() if any(w in q for w in words)]
    # Only route when the signal is unambiguous; 2+ families means don't narrow.
    return hits if len(hits) == 1 else []


if __name__ == "__main__":
    chunks = chunk_all()
    print(f"Total chunks: {len(chunks)}")
    CHUNKS_JSON.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {CHUNKS_JSON}")
    for c in chunks[:3]:
        print(c["filename"], "|", c["section"][:30], "| step:", c["step"], "|", c["text"][:80].replace("\n", " "))

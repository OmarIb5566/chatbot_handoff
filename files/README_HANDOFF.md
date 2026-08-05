# RME Process Chatbot

A retrieval-augmented chatbot over RME's ISO process documents. Ask a question in **English or
Arabic**, get an answer built only from the process PDFs, with the source documents cited and
every form number it mentions checked against the corpus.

Everything runs locally — PDF extraction, retrieval, and generation through Ollama. Nothing
leaves the machine.

The model is **`qwen3:14b`**, chosen on a 36-question eval whose results are kept in
`model_eval_results.json` (100% overall, 100% on the `unanswerable` category, no flagged
citations). The eval harness that produced that file has been removed now that the decision is
made, so treat the JSON as a record rather than something reproducible from this repo.

---

## 1. Layout

Keep these two folders as siblings. `adaptive_chunker.py` and `extract_pipeline.py` both
resolve their PDF source as `../processes_pdf`, so moving `files/` on its own will break
chunking and extraction.

```
RME Chatbot/
  files/            <- code, notebook, eval set, extracted artifacts
  processes_pdf/    <- the 9 source PDFs
```

## 2. Install

```bash
pip install -r requirements.txt
```

Python **3.10+** required. No dependency is needed for the Arabic support — the same local
model does the translating.

First run downloads the `all-MiniLM-L6-v2` embedding model from HuggingFace (~90 MB). If the
machine is offline, pre-fetch it — the retriever loads it unconditionally, because query
encoding needs it even when the corpus embeddings are cached.

## 3. Pull the model

```bash
ollama pull qwen3:14b
```

~8.6 GB. Then make sure the server is up:

```bash
ollama serve
```

## 4. (Optional) Rebuild the chunks

`chunks.json` ships prebuilt, so **you can skip this**. The notebook re-chunks the PDFs anyway
and checks the result matches. Run this only to rebuild the shipped artifacts themselves:

```bash
python adaptive_chunker.py                     # ../processes_pdf -> chunks.json
python adaptive_chunker.py --audit audit.json  # ...and the full per-document report
```

Chunking reads the PDFs **directly**, not any intermediate JSON, because it detects section
boundaries from font size and bold flags — layout that only exists in the PDF. So there is no
required order here and no extraction step to run first. An audit summary prints on every run;
the report file is opt-in.

`extract_pipeline.py` is a separate, independent **OCR-health audit**:

```bash
python extract_pipeline.py    # ../processes_pdf -> extracted_raw.json + ocr_fallback_log.json
```

It matters because the chunker has no OCR path, so a page with no embedded text is a page it
cannot see. The chunker reports those as `pages with no text`; this log is the other half of
the same picture. Neither output file is committed — the demo notebook runs this stage itself
into its own `demo_*.json` files.

**Expect 9 `ocr_failed` lines. This is not a broken run.** Page 1 of every document is a
scanned signature/approval cover page with no embedded text, so it trips the OCR fallback. If
`tesseract` is not installed those pages log as `ocr_failed` and the run continues on native
text. No eval question touches cover-page content, and retrieval results are identical whether
or not tesseract is present — so installing it is not recommended.

## 5. Run it

```bash
jupyter lab
```

Open `demo_chatbot.ipynb` → **Run All**, then go to §5.2 or §5.3 and ask your own questions.

- **§1–§4** rebuild the pipeline from the PDFs: OCR-health check → chunk → hybrid BM25 +
  MiniLM/FAISS index → form-number whitelist. Fast, and works with Ollama down.
- **§5** is the chatbot. §5.1 worked examples, §5.2 a cell to edit, §5.3 an interactive REPL,
  §5.4 a question the documents do not answer, §5.5 the Arabic path.

Generation is deterministic — `temperature=0`, `seed=0`, so a repeated question gives a
repeated answer.

`qwen3:14b` on CPU is roughly 20–30 s for a short English answer once warm (a form number, a
deadline, a role), plus a one-off model load of up to a few minutes on the first call. Arabic
questions cost 2–3× that; see below.

Answers are **not length-capped** (`num_predict: -1`), so an explanatory question — "explain the
process operations in procurement" — legitimately takes several minutes on CPU rather than
being cut off mid-sentence at 200 tokens, which is what it used to do. `num_ctx` is pinned at
8192 so the context window cannot quietly reintroduce that ceiling. There is no upper bound on
a vague question beyond the 1800 s HTTP timeout.

## 6. Arabic support

`translate.py`. An Arabic question is translated to English *before* it is embedded, the whole
existing pipeline runs on English, and the answer is translated back to Arabic *after* it has
been validated. Detection is automatic — there is no flag to set, `ask()` checks the script.

Two consequences worth knowing:

- Retrieval, doc-code routing, BM25 and the form-number validator all keep operating on
  English, so nothing about the measured retrieval accuracy changes. The translation prompts
  pin form numbers, doc codes and filenames as verbatim Latin script, because a form number
  re-rendered in Arabic-Indic digits is a citation nobody can look up.
- An Arabic question makes three generation calls instead of one. The latency line breaks
  translate-in / generation / translate-out out separately.

Arabic answer quality is **unmeasured** — the eval set is English-only. The path works; that is
a different claim from it being accurate, and an Arabic eval set is the obvious next step.

Smoke-test the translation on its own, without Jupyter:

```bash
python translate.py
```

---

## Notes on what the notebook does that isn't obvious

**Reasoning traces are stripped before validation.** Qwen 3 emits `<think>…</think>` blocks. A
trace musing *"it might be F-P-CM-01-01 or F-P-CN-01-11"* would register as a hallucinated form
citation the model never actually made. Requests set `think: False`, and `strip_reasoning()`
removes anything that slips through — including on translations. The raw response is kept in
the returned record for audit.

**The validator is a hallucination proxy independent of the answer itself.** It extracts form
numbers from each answer and checks them against the 67 that actually occur in the corpus. It
*flags* rather than *blocks*, because at 9 documents the whitelist is incomplete — the corpus
cites form families (FW, HR, OP, PU, QP) whose source documents aren't here yet, so "unknown"
currently means *unverifiable*, not *fabricated*. `policy="block"` becomes safe once the full
500–600 document corpus is indexed.

**Chunking is template-independent by design.** `adaptive_chunker.py` decides *where* a section
starts from page layout (font size, bold flags, falling back to the `1. SOMETHING` numbering
pattern) and *what* it is by comparing the heading to a canonical taxonomy — exact, then fuzzy,
then MiniLM cosine. Those are two separate stages on purpose. The chunker it replaced did both
with one literal string list, so a document written to a different template matched nothing and
collapsed into a single chunk without reporting a problem.

Two measured caveats. The **font signal is nearly inert on this corpus** — headings are 12 pt
against 11 pt body, under the 1.15 ratio, so `numbering` fires 103 times to `font_size`'s 3 and
the layout half is largely untested here. And **every threshold is calibrated on 9 documents**
(0.85 fuzzy, 0.45 cosine, 1.15 size ratio); the first real batch of non-ISO documents is the
actual test.

**Chunking audits itself.** Every run prints which detectors fired, how each section label was
resolved, which documents fell back to windowed chunking, which headings matched no canonical
bucket, and which pages yielded no text. `--audit PATH` writes the full per-document report.
At 500–700 documents that report is the only practical way to notice a template outlier before
it shows up as a bad answer.

**Extending it to a new document family** is one string in `CANONICAL_SECTIONS`, not a regex.
Headings that match nothing are labeled `UNMATCHED`, still chunked and still retrievable, and
listed in the audit — a flag for a human, never a discard.

**No chunk carries a step number.** Step-level splitting inside `PROCESS OPERATION` is not
implemented — `step` is `None` on all 114 chunks. Harmless at 9 documents, where sections are
small; at 500–600 it is what keeps a process step attached to its own form number, and it is
the next thing to build.

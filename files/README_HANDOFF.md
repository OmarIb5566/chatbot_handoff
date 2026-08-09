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

## 4b. Scanned workflow diagrams

Not every PDF here is a process document. `CS Signature Matrix.pdf` is three pages of **pure
image** — approval flowcharts with roles, decision diamonds, rejection loops and red monetary
thresholds. `adaptive_chunker.py` reads PDF layout, so it sees nothing there, and before
`workflow_extractor.py` existed it failed *silently*: the file contributed zero chunks and only
moved a counter.

OCR alone does not fix it, which is the whole point — the meaning is in the **edges**, and OCR
returns role names with every relationship destroyed. So documents are routed on extractable
text, and scanned ones go to a vision model that reads the topology:

```bash
python workflow_extractor.py --audit workflow_audit.json   # -> workflow_chunks.json
python workflow_extractor.py --from-audit workflow_audit.json   # re-render, no model calls
```

Needs a **vision-capable** model — `qwen3.6:27b` by default. `qwen3:14b`, the model that answers
questions, has no vision capability; check `/api/tags` before changing it. `glm-ocr:latest` is
used as a second, independent reader to cross-check.

The model emits a **structured graph** (nodes, edges, conditions, per lane) and the embedded
prose is rendered deterministically from it. Free prose would be unauditable, and a paraphrased
threshold is invisible. Chunking is **per lane, not per file**: this one PDF holds six distinct
approval flows, and as a single chunk a question about steel would also retrieve caravans.

**These chunks are the only part of the corpus that is not ground truth.** They carry
`source_type: "vlm_description"` and `review: "machine"`, they are excluded from the validator's
form whitelist, and the app labels any answer built from them. Three extraction runs at
temperature 0 misread the same crowded junction three different ways — dropping a threshold,
inventing a reverse arrow to hang it on, duplicating an arrow to hang it on. The audit catches
all three shapes plus omissions found by the second reader. It **cannot** catch a clean
misreading (`Over 3M -> VP` instead of `CEO`), which is what `review: "machine"` is for: six
lanes against three pages is about twenty minutes of human checking.

`processes_pdf/` is gitignored, so the source PDF is not in the repo — `workflow_chunks.json` is
the committed artifact, the same relationship `chunks.json` has to the process PDFs.

## 5. Run it

Two front ends over the same pipeline. **If you just want to ask questions, use the app:**

```bash
streamlit run app.py
```

Opens in a browser. Type in English or Arabic — the language is detected, not selected. Every
answer comes with an expander holding the chunks it was built from, the validator's verdict and
the latency breakdown, because an answer alone cannot tell you whether a wrong result was a
retrieval failure or a generation failure. The sidebar switches model, chunk count, and whether
unverifiable citations are blocked or merely flagged.

**The notebook is the explanation**, not the interface:

```bash
jupyter lab
```

Open `demo_chatbot.ipynb` → **Run All**, then go to §5.2 or §5.3 and ask your own questions.

Both import `chatbot.py`, which holds the actual pipeline — one copy of the prompt text, the
reasoning-strip regex and the validation order. The notebook and the app differ only in how
they render the result.

- **§1–§4** rebuild the pipeline from the PDFs: OCR-health check → chunk → hybrid BM25 +
  MiniLM/FAISS index → form-number whitelist. Fast, and works with Ollama down.
- **§5** is the chatbot. §5.1 worked examples, §5.2 a cell to edit, §5.3 an interactive REPL,
  §5.4 a question the documents do not answer, §5.5 the Arabic path.

Generation is deterministic — `temperature=0`, `seed=0`, so a repeated question gives a
repeated answer.

**Latency depends almost entirely on whether the model is on a GPU, so the notebook prints that
alongside the numbers** (§5.1, from Ollama's `/api/ps` — `torch` cannot tell you, since the
`torch` pulled in here is a CPU-only build while Ollama ships its own CUDA runtime).

For a short English answer once warm — a form number, a deadline, a role:

| | short answer | long explanatory answer |
|---|---|---|
| model resident on GPU | ~3 s | tens of seconds |
| CPU only | 20–30 s | ~8 min |

Plus a one-off model load on the first call. Arabic questions cost about 2× the English figure —
one extra call to translate the question; see §6. Retrieval is ~10–40 ms either way.

Answers are **not length-capped** (`num_predict: -1`), so an explanatory question — "explain the
process operations in procurement" — legitimately takes several minutes on CPU rather than
being cut off mid-sentence at 200 tokens, which is what it used to do. `num_ctx` is pinned at
8192 so the context window cannot quietly reintroduce that ceiling. There is no upper bound on
a vague question beyond the 1800 s HTTP timeout.

## 6. Arabic support

`translate.py`. An Arabic **question** is translated to English before it is embedded, and the
whole existing pipeline then runs on English. The **answer is generated in Arabic directly from
the English context** — it is not translated back. Detection is automatic; there is no flag to
set, `ask()` checks the script.

The path is asymmetric on purpose: **Egyptian dialect in, Modern Standard Arabic out.** Users
write dialect, so the inbound translation has to handle it — as it does code-switching
("الـ procurement process بيبدأ ازاي"), because the language check counts the fraction of
*letters* in Arabic script rather than requiring a majority. Answers come back in formal Arabic,
which is the right register for a reply quoting ISO process documents and the one `qwen3:14b`
produces reliably.

Four things worth knowing:

- Retrieval, doc-code routing, BM25 and the form-number whitelist all keep operating on English,
  so nothing about the measured retrieval accuracy changes.
- **Form numbers and doc codes are masked out before translation and restored after**
  (`mask_codes`), so a code cannot be corrupted by a model that never saw it. The verbatim-Latin
  instruction in the prompt remains as a second line of defence for PDF filenames. This matters
  because `validator.py` identifies citations by regex: one mangled digit turns a correct answer
  into a false hallucination flag, and on the way in it silently breaks routing.
- An Arabic question makes two generation calls instead of one, so roughly 2× the latency of the
  equivalent English question. Generating the answer in Arabic rather than translating it back
  removed a third call that measured 6–18 s per question.
- Validation runs on the Arabic answer. That is safe because `FORM_RE` is Latin-script, but
  `ask()` verifies rather than assumes: it warns if an answer contains Arabic-Indic digits,
  since a code written in those is invisible to the validator, not merely wrong.

**Measured:** retrieval hit rate on the Arabic path, via `eval_arabic.py` — the one metric whose
ground truth is language-independent, so it sits on the same scale as the English number.

```bash
python make_arabic_eval.py   # eval_set.json -> eval_set_ar.json (needs Ollama)
python eval_arabic.py        # English vs Arabic->English vs raw Arabic, top-3 doc accuracy
```

```
English (baseline)    32/32 = 100.0%
Arabic -> English     31/32 =  96.9%
Arabic (raw, no MT)   18/32 =  56.2%      <- control
```

The control row shows the translation step is what carries the result. It is higher than the
near-zero you might expect: with BM25 contributing nothing on Arabic, ranking falls entirely to
the dense channel, and MiniLM retains some cross-lingual signal — against a chance baseline
already around 33%, since top-3 of 9 documents is not a hard target.

The one miss is instructive. PTN01-3 asks about the *Commercial dept*, the Arabic renders it
`قسم التجارة`, and that returns as "trade department" — losing a word that is both a BM25 term
and a routing keyword. A translation failure, not a retrieval one, and the silent kind. Pinning
RME's vocabulary into the translation prompt with a glossary is the obvious next fix.

**Read the number as an upper bound.** The Arabic questions are generated by the same model that
translates them back, so the round trip flatters itself; and they came out in MSA despite being
asked for in Egyptian, so the set does not exercise dialect *input* — the half that still
matters, now that output is formal Arabic by choice. Dialect input does work: the §5.5 questions
are hand-written Egyptian and retrieve correctly. Every record is tagged `"review": "machine"`;
naturalising them into real dialect and setting `"human"` is the change that would make the
number quotable, and the script reports that subset separately.

**Still unmeasured:** Arabic *answer* quality. Scoring it needs `must_include` strings in Arabic,
which a person has to write.

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

# RME Process Chatbot — handoff

Everything in this prototype runs **except** the three-model generation eval, which needs
Ollama. Ollama could not be installed on the machine this was built on (`github.com` is
blocked there, and the installer is GitHub-hosted). That is the only reason this is being
handed over.

**What you need to do: run the notebook, then send back two files.** Nothing needs editing.

---

## 1. Layout

Keep these two folders as siblings. `extract_pipeline.py` resolves its PDF source as
`../processes_pdf`, so moving `files/` on its own will break extraction.

```
RME Chatbot/
  files/            <- code, notebook, eval set, extracted artifacts
  processes_pdf/    <- the 9 source PDFs
```

## 2. Install

```bash
cd "RME Chatbot/files"
pip install -r requirements.txt
```

Python **3.10+** required.

First run downloads the `all-MiniLM-L6-v2` embedding model from HuggingFace (~90 MB). If
the machine is offline, pre-fetch it — the retriever loads it unconditionally, because
query encoding needs it even when the corpus embeddings are cached.

## 3. Pull the models

```bash
ollama pull llama3.2:latest
ollama pull gemma4:e4b
ollama pull qwen3:14b
```

~19.5 GB total (1.88 + 8.95 + 8.64). All three tags were verified against
`registry.ollama.ai`.

Then make sure the server is up:

```bash
ollama serve
```

## 4. (Optional) Re-run extraction

The extracted artifacts ship prebuilt, so **you can skip this**. Run it only if you want to
verify the PDF → text path end to end:

```bash
python extract_pipeline.py    # ../processes_pdf -> extracted_raw.json + ocr_fallback_log.json
python chunker.py             # extracted_raw.json -> chunks.json
```

That order matters, and it overwrites `extracted_raw.json` and `chunks.json`. It is
reproducible — re-extracting reproduces the BM25 retrieval baseline exactly.

**Expect 9 `ocr_failed` lines. This is not a broken handoff.** Page 1 of every document is
a scanned signature/approval cover page with no embedded text, so it trips the OCR fallback.
If `tesseract` is not installed those pages log as `ocr_failed` and the run continues on
native text. No eval question touches cover-page content, and retrieval results are
identical whether or not tesseract is present — so installing it is not recommended.

## 5. Run the notebook

```bash
jupyter lab       # launch from inside files/
```

Open `prototype.ipynb` → **Run All**.

- §1–3 reproduce extraction, retrieval and validator numbers. Fast, no model needed.
- §4.1 records the hardware — **if the GPU is not auto-detected, please note manually what
  you ran on** (`ollama ps` shows whether models are resident on GPU or CPU). The brief asks
  for latency on real hardware, and that number can't be interpreted without it.
- §5.1 runs 36 questions × 3 models against byte-identical frozen retrieval context.

Wall time: minutes on a CUDA GPU; well over an hour on CPU (`qwen3:14b` and `gemma4:e4b`
dominate, and qwen3 spends extra tokens on reasoning).

Generation is deterministic — `temperature=0`, `seed=0`.

## 6. Send back

- `prototype.ipynb` — executed, with outputs saved
- `model_eval_results.json` — written by §5.2

---

## Notes on what the notebook does that isn't obvious

**Reasoning traces are stripped before scoring.** Qwen 3 emits `<think>…</think>` blocks.
Scoring is naive substring matching, so a trace musing *"it might be F-P-CM-01-01 or
F-P-CN-01-11"* would count as correct **and** register a hallucinated form citation the
model never actually made. Requests set `think: False`, and `strip_reasoning()` removes
anything that slips through. The raw response is kept in the results for audit.

**Retrieval context is frozen once**, before any model runs, so the comparison measures the
models rather than retrieval variance.

**The validator is a hallucination proxy independent of the accuracy score.** It extracts
form numbers from each answer and checks them against the 67 that actually occur in the
corpus. It *flags* rather than *blocks*, because at 9 documents the whitelist is incomplete
— the corpus cites form families (FW, HR, OP, PU, QP) whose source documents aren't here
yet, so "unknown" currently means *unverifiable*, not *fabricated*.

**§6 is intentionally blank.** The recommendation gets written from your numbers, against a
decision rule fixed in advance (disqualify on the `unanswerable` category first, then on
validator catch rate, then compare accuracy, latency as tiebreaker).

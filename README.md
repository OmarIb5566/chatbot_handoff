# RME Process Chatbot

A retrieval-augmented chatbot over RME's ISO process documents and approval workflow diagrams.
Ask a question in **English or Arabic**, get an answer built only from the corpus, with the
source document cited and every form number checked against what actually occurs in the
documents.

Everything runs locally — PDF extraction, retrieval, and generation through Ollama. Nothing
leaves the machine.

> This file is the repository-level map: what is here, how the pieces connect, what regenerates,
> and what is currently known to be broken.
> **[README_HANDOFF.md](README_HANDOFF.md) is the operating manual** — install, model choice,
> latency, Arabic path, and the measured reasoning behind each early design decision. It predates
> both the folder split and the vector extraction path, so where the two disagree, **this file is
> current**.

---

## 1. Layout

```
chatbot_handoff/
  backend/            the pipeline (extraction, retrieval, generation, validation)
  evals/              eval harnesses and eval sets
  data/               committed extraction artifacts (chunks, audits, embed cache)
  eval_results/       recorded eval output
  processes_pdf/      71 ISO process documents      -> adaptive_chunker.py
    other/            59 forms, matrices, annexes — parked, in neither pipeline
  Workflows/          110 approval flowcharts       -> workflow_vector.py
  app.py              Streamlit front end
  demo_chatbot.ipynb  the explanation notebook
```

`backend/paths.py` resolves every path from the repository root, so the folders have to stay
where they are. The PDFs are **deliberately tracked** (see `.gitignore`) so a clone has the
deduplicated corpus and extraction can be re-run.

**Corpus as served.** `data/chunks.json` — 1063 chunks over 71 process documents.
`data/workflow_chunks_fixed.json` — 137 chunks over 110 workflow diagrams (134 `pdf_vector`,
3 `vlm_description`), of which `drop_unreadable()` excludes one that carries no step names.
`chatbot.load_corpus()` merges them into **one index of 1199 chunks**; a user does not know
which kind of document answers their question.

`data/workflow_chunks.json` is the raw extraction, kept so the repaired corpus can be diffed
against it. `paths.WORKFLOW_CHUNKS_ACTIVE` names the one actually loaded, and `load_corpus()`
falls back to the raw file if it has not been built on a given checkout.

## 2. The three readers

The corpus holds two kinds of document that need completely different readers, and sending one
to the other's reader fails *silently* in both directions. `backend/document_router.py` decides
which is which — on ISO section headings and page count, two orthogonal signals, with
disagreements reported rather than resolved.

```
                       document_router.classify()
                        /                      \
          process document                   workflow diagram
                 |                                  |
        adaptive_chunker.py                 workflow_vector.py
        layout-aware sectioning             reads the PDF's own vector layer:
                 |                          shapes, arrows, arrowheads
                 |                                  |
                 |                          (no text layer? fall back to)
                 |                          workflow_extractor.py — vision model
                 |                                  |
         chunks.json (1063)              workflow_chunks.json (137)
                 \                                  /
                  \------> retriever.py <----------/
                        BM25 + MiniLM/FAISS hybrid, one index
                                  |
                            chatbot.py  ->  app.py / demo_chatbot.ipynb
```

Why the split matters: a flowchart's meaning is in the **edges** — which arrow points where,
which diamond loops back on "Not Approved", which monetary label decides who signs. A text
extractor returns the labels with every relationship destroyed, which reads like content and
answers nothing.

**`workflow_vector.py` is the primary workflow reader and involves no model at all.** It reads
shapes and connector polylines out of the PDF's vector layer and derives the topology
geometrically, so every token in a workflow chunk is verbatim from the file. The vision path
(`workflow_extractor.py`) now handles only the two image-only signature matrices that have no
text layer to read — 3 chunks of 137.

```bash
python backend/document_router.py
```

## 3. Modules

| File | What it does |
|---|---|
| `backend/paths.py` | Every path in the repo, resolved from the root. Import this rather than recomputing |
| `backend/document_router.py` | Routes each PDF to its extractor; reports files where the two signals disagree |
| `backend/adaptive_chunker.py` | Process PDFs → `chunks.json`. Finds section boundaries from layout, labels them against a canonical taxonomy. Audits itself on every run |
| `backend/workflow_vector.py` | Workflow PDFs → `workflow_chunks.json`, geometrically, no model. Shapes → nodes, connector polylines → edges, arrowheads → direction |
| `backend/workflow_extractor.py` | Vision fallback for scanned pages, plus `render_prose()` — the single place a graph becomes the text that gets embedded |
| `backend/extract_pipeline.py` | Independent OCR-health audit — which pages carry no embedded text |
| `backend/retriever.py` | Hybrid BM25 + MiniLM/FAISS retrieval, doc-code routing, workflow-intent boost and mix floor, relevance cutoff. Embeddings cached in `data/.embed_cache/` |
| `backend/contextualize.py` | Follow-up resolution — rewrites "who signs after that" into a standalone query, only when a cheap gate fires |
| `backend/translate.py` | Arabic in (Egyptian dialect), MSA out. Form numbers masked before translation, restored after |
| `backend/validator.py` | Hallucination proxy: extracts form numbers from each answer, checks them against those occurring in the corpus. Flags, does not block |
| `backend/config.py` | Every tunable in one place, each overridable by an `RME_*` environment variable. The defaults are the tuned ones |
| `backend/errors.py` | The expected runtime failures as distinct types, each carrying the sentence the UI shows |
| `backend/logs.py` | One logging setup, and the one-line-per-answer operational record. Question and answer text is opt-in, not default |
| `backend/chatbot.py` | The pipeline itself — corpus loading, prompt text, context budget, the validation order |
| `app.py` | Streamlit front end. Product view by default; diagnostics behind **Developer view** in the sidebar |
| `demo_chatbot.ipynb` | The explanation — rebuilds the pipeline end to end and shows the reasoning |

## 4. Quick start

```bash
pip install -r requirements.txt
ollama pull qwen3:14b
streamlit run app.py
```

First question is slow: it builds the MiniLM index. After that, retrieval is ~15 ms and
generation is 60–110 s on `qwen3:14b`.

Nothing above needs configuring, but nothing above is hardcoded either. Every setting reads an
environment variable and falls back to the tuned default (`backend/config.py`), so a different
host or model is a prefix rather than a diff:

```bash
RME_OLLAMA_HOST=http://gpu-box:11434 RME_MODEL=qwen3.6:27b streamlit run app.py
```

| Variable | Default | Notes |
|---|---|---|
| `RME_OLLAMA_HOST` | `http://localhost:11434` | |
| `RME_MODEL` | `qwen3:14b` | Smallest size tested that holds the workflow answer format |
| `RME_REWRITE_MODEL` | same as `RME_MODEL` | Follow-up rewriting only |
| `RME_TOP_K` | `6` | Tuned against `evals/`; a workflow mix floor depends on it |
| `RME_CONTEXT_BUDGET_CHARS` | `18000` | Over this, Ollama drops the *instruction*, not the tail |
| `RME_GEN_TIMEOUT` | `1800` | Seconds. Long workflow answers legitimately run for minutes |
| `RME_POLICY` | `flag` | `block` withholds answers citing an unverifiable form number |
| `RME_LOG_LEVEL` / `RME_LOG_FILE` | `INFO` / stderr | |
| `RME_LOG_CONTENT` | `0` | Off: operational logs record shape and timing, not what was asked |

An unparseable value falls back to the default and is reported at startup rather than crashing
the import or being silently ignored.

### Tests

```bash
pytest
```

48 tests, under a second, no model and no network. They cover the logic *between* retrieval and
generation — the context budget, the two workflow caveats, version-collision detection, the
failure mapping at the model boundary — which is where every regression found so far has
actually lived. Answer and retrieval quality are not testable this way; that is what `evals/`
is for.

## 5. Rebuilding the artifacts

Both chunk files ship prebuilt, so **you can skip this**.

```bash
python backend/adaptive_chunker.py --audit audit.json     # processes_pdf -> chunks.json
python backend/workflow_vector.py                          # Workflows     -> workflow_chunks.json
```

`workflow_vector.py` is deterministic geometry — same PDFs and same code give byte-identical
output, verified across all 110 files. Re-extraction is therefore only worth running after the
extractor itself changes.

**Workflow chunks are not ground truth.** They carry `review: "machine"`, they are excluded from
the validator's form whitelist, and the app labels any answer built from them. The tokens are
verbatim from the PDF, but *which arrow points where* is inferred, and `coverage` plus
`audit_warnings` on each chunk record how much of the page the extractor could account for.

## 6. Retrieval and answer shape

Two populations share one index, roughly 8:1 in favour of process text. Left alone, BM25 favours
the process side on almost any query, because process PDFs are long repetitive prose while a
workflow chunk is a short rendering of a graph. Four mechanisms address that, and **none of them
is a router** — nothing is excluded from the candidate pool on the basis of `source_type`:

| Mechanism | Default | What it does |
|---|---|---|
| `workflow_intent()` | — | Keyword test for route-shaped questions ("who signs", "what happens after") |
| `workflow_boost` | `0.06` | Soft score nudge toward diagram chunks when the intent fires |
| `workflow_floor` | `2` | Guarantees both populations a minimum share of top-k, capped at a third each |
| `min_rel` / `min_hits` | `0.55` / `3` | Relevance cutoff: `top_k` becomes a ceiling, not a quota |
| `top_k` | `6` | Raised from 3; `fit_context()` keeps 6 large workflow chunks from overflowing `num_ctx` |

`workflow_boost` was lowered from 0.12 to 0.06 on measurement: 0.12 cost a process question on
`eval_set` (30/32 → 29/32) and bought nothing back on the workflow set.

**Answer shape.** When a diagram is in the retrieved context, the prompt asks for a narrative
walk — a lead-in, numbered steps, `A decision is made:` with one sub-bullet per branch, an
explicit ending, and a trailing `Returns and loops:` list. The source document is cited **once**,
on its own line at the end.

The returns line is not cosmetic. Measured on three diagrams, routes represented against
`graph.edges`: **14/36 without it, 28/36 with it**. It also stopped the model inventing a
threshold — on the rental-equipment flow it had been manufacturing a decision at `Finance Review`
using condition labels borrowed from a different step.

## 7. Evaluation

```bash
python evals/eval_retrieval.py --eval evals/eval_set_v2.json --top-k 3
python evals/eval_retrieval.py --eval evals/eval_set.json --top-k 3
python evals/eval_retrieval.py --eval evals/eval_set_workflow.json --top-k 6
```

Current, against the production corpus:

| Eval set | Questions | Score |
|---|---|---|
| `eval_set_v2.json` | 92 answerable | **86/92** top-3 |
| `eval_set.json` | 32 answerable | **27/32** top-3 |
| `eval_set_workflow.json` | 18 | **18/18** top-6, 16/18 top-3, 6/18 top-1 |

`eval_retrieval.py` now calls `chatbot.load_corpus()` and **always scores against the corpus
production actually serves**. It used to merge workflow chunks only when an eval set's gold
documents needed them, which meant `eval_set_v2` was never once scored against the real index —
88/92 against the process-only index it built, 86/92 against the real merged one. Two questions
were regressing where nothing in the repo could see it.

`evals/make_workflow_eval.py` generates the workflow eval set from the diagrams' own graphs.

## 8. What works, and what does not

### Works, measured

- **Retrieval.** 86/92 and 27/32 top-3 on the process sets, 18/18 top-6 on workflow. Dense and
  BM25 are at parity on graph-rendered text (17/18 each at top-6); fusion beats both.
- **Arabic.** Question translated to English for retrieval, answer generated in Arabic.
- **Follow-up resolution.** "who signs after that?" resolves against the previous turn — the
  gate fires on `back-reference (that)` and the rewriter binds the pronoun to the step the
  previous *answer* named.
- **The workflow answer format**, including loops and send-backs (see §6).
- **Extraction is reproducible.** Byte-identical across all 110 workflow PDFs.

### Does not work, or is not verified

- **~30% of workflow graphs are structurally defective.** 78 of 137 chunks are structurally
  clean. The rest have duplicate node labels (the extractor emits `x` and `x_2` for one box),
  self-loops, or a node connected to nothing.
- **18 files have an orphaned node** — usually `End`, sometimes `Auto-Creation`. On 17 of them
  the arrow is **absent from the PDF's vector layer**: the only thing near the box is a 2-point
  stub, or the nearest connector endpoint is 75–149 pt away. No extractor change recovers an
  arrow that was never drawn as one. This is now **said out loud**: an answer built from such a
  diagram carries a caveat naming the steps that connect to nothing (`chatbot.workflow_gaps`,
  rendered by `app.reliability_notes`). The gap is not closed — it is no longer silent.
- **`Variation Order Flow Chart.pdf` is in the wrong pipeline.** 36 of its nodes are whole
  sentences ("Contract Department reviews the Merit of VO … within 7 days"). It is a text-heavy
  process document being parsed as a box-and-arrow flowchart.
- **Threshold text is garbled on some diagrams.** Condition strings are concatenated arrow
  labels: `"Below 3M SAR Above 3M SAR"` collapses both branches of a decision into one string,
  so even a correctly retrieved document cannot tell you which side an approver is on.
- **Near-duplicate documents make eval scores softer than they look.** Of the 18 workflow
  questions, only 2 have a unique correct answer in the corpus; 16 have another diagram carrying
  the identical predecessor → successor edge. Document-level scoring is the wrong metric for
  this population.
- **A sibling-document mix-up reached a user.** Asked about the general subcontract amendment
  process, the bot answered with the **rental-equipment variation-order** threshold
  (3M EGP → VP Approval). The general process turns on 5M/10M/20M EGP with different approvers,
  and the KSA variant is in SAR. Every workflow chunk now carries `family` and `variant`
  (see §9), and those fields are now used: when the retrieved chunks span two documents of one
  family, the context labels each chunk with its version, a prompt rule requires the answer to
  come from one of them and to name it, and the UI lists the versions it did *not* use
  (`chatbot.variant_conflicts`). Measured on the case that produced this failure — "subcontract
  preparation approval steps", which returns V1 at 1.036 and V3 at 0.960 — the answer now opens
  by naming V1 and closes by listing V3, instead of merging them.

  **Nothing is dropped**, deliberately: `workflow@1` is 6/18, so collapsing to the top-scoring
  variant would trade a visible ambiguity for a silent wrong pick. It also does not try to
  separate superseding *versions* (`Subcontract Preparation V1/V3/V4`) from distinct *scopes*
  (`IIR – Manholes / Pipes / Valve Chambers`); the filenames do not carry enough to tell them
  apart, and both want the same handling.
- **The Streamlit UI has never been visually verified** by the automated checks — only the
  record fields behind it.

## 9. The repaired corpus, and the metadata on it

`data/workflow_chunks_fixed.json` is **live** — re-extracted with the `merge_fragments` fix and
then duplicate-label collapsed. Measured against the raw extraction:

| | current | fixed |
|---|---|---|
| structurally clean | 78/137 (56.9%) | **96/137 (70.1%)** |
| duplicate-label chunks | 34 | **4** |
| self-loops | 25 | **0** |
| route coverage on a deterministic walk | 78.1% | **83.6%** |
| retrieval (all four measures) | — | **identical** |

Text is re-rendered with the extractor's own `render_prose()`, and the threshold tail is spliced
from the original verbatim — thresholds live on a page-level `escalation` list that chunks do not
store. That splice was gated: with the graph left unmodified, re-render + splice had to equal the
stored text **byte for byte** on all 137 chunks before anything was written.

**Still open:** three files lose 5 edges each (`RFI - RME to Client V2`, `S-C Invoice MEP`,
`S-C Invoice Operation`) against 13 gained elsewhere. Net positive, and retrieval is unchanged
on all four eval measures, but that cluster has not been explained.

### Metadata on workflow chunks

| Field | Coverage | Where it comes from |
|---|---|---|
| `family` | 136/136 | Filename before `" - "` — e.g. `Subcontract Amendments` |
| `variant` | 133/136 | Filename after `" - "` — e.g. `Rental Equipment Variation Order WF` |
| `doc_code` | 6 chunks / 3 documents | A code actually printed in the filename or on the page |

**Only 3 of 110 workflow documents carry a real document code**, so `doc_code` cannot be
populated for the rest and is left `None`. It was not invented: minting plausible-looking RME
document numbers into a compliance corpus is how a fabricated identifier ends up cited to a user.
Mapping each diagram onto its process document automatically was tried and rejected on
measurement — only 3 of 59 families matched confidently, and the near-misses were wrong in
exactly the dangerous way (`Internal Head Office NCR` → `Head Office Dress Code Policy`).

`doc_code_prefix()` now strips an optional `F-`, because `F-P-VMO-01-07` is a form in the
`P-VMO` family. Those three documents route with their process family; before, **no workflow
chunk could be routed at all**.

### The extractor bug that was fixed

`merge_fragments()` joins connector fragments that share an endpoint. On the NOD Auto sheet two
arrows leave `Contracts Department` from the same point on its edge — one routed above the page,
one below — and both arrive at `End`. Union-find saw a single path
`End(bottom) → Contracts Department → End(top)`; every interior vertex had degree 2, so the
fan-out guard did not fire. Merged into one polyline it began and ended on the *same* box, and
`direct_edges` discarded it at `ia == ib`. Both arrows were lost.

The fix splits a merged chain wherever it passes **through** a box — but only when the merge is
about to be useless, i.e. both free ends snap to the same box. A first version split every chain
near a box and measured badly: 223 edges lost against 134 gained. The narrow version fixes one
file and regresses none.

## 10. Repository conventions

- `.gitattributes` pins line endings (LF in the repo) and marks every binary type, because the
  repo is worked on from Windows and tracks 400+ PDFs.
- Machine-produced records are tagged `"review": "machine"` — in `eval_set_ar.json`,
  `eval_set_workflow.json` and every workflow chunk — and become `"human"` only after a person
  has checked them.
- Regenerated files are listed in `.gitignore` with the command that writes them.
- Expected runtime failures raise a `PipelineError` subclass carrying a user-facing sentence;
  anything else is a bug and is allowed to propagate, so the two are never flattened together.
- `_verify/` holds the scripts and recorded output behind the numbers in §6–§9. Read-only
  diagnostics; nothing in the pipeline imports them.

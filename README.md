# RME Process Chatbot

A retrieval-augmented chatbot over RME's ISO process documents and approval workflow diagrams.
Ask a question in **English or Arabic**, get an answer built only from the corpus, with the
source documents cited and every form number checked against what actually occurs in the
documents.

Everything runs locally — PDF extraction, retrieval, and generation through Ollama. Nothing
leaves the machine.

> This file is the repository-level map: what is here, how the pieces connect, what regenerates.
> **[files/README_HANDOFF.md](files/README_HANDOFF.md) is the operating manual** — install, model
> choice, latency, Arabic path, and the measured reasoning behind each design decision. Read that
> one to *run* the thing. Note that it predates the `Workflows/` corpus and still describes a
> 9-document `processes_pdf/`, so where the two disagree on corpus size, this file is current.

---

## 1. Layout

```
chatbot_handoff/
  files/                  code, notebook, eval sets, committed extraction artifacts
  processes_pdf/          71 ISO process documents  ->  adaptive_chunker.py
    other/                64 forms, matrices, annexes — parked, in neither pipeline
  Workflows/              110 approval flowcharts   ->  workflow_extractor.py
    overseas/             29 more, KSA / UAE / Côte d'Ivoire variants
  workflow_audit_full.json    per-page graphs + warnings from the full 131-document run
  workflow_audit_new.json     the same, for a 5-document follow-up run
```

`files/` resolves the corpus as `../processes_pdf` and `../Workflows`, so the folders have to
stay siblings.

The PDFs are **deliberately tracked** right now (see the note in `.gitignore`) so a clone has the
deduplicated corpus and the extraction can be re-run on a machine with the vision model pulled.

## 2. The two extraction paths

The corpus holds two kinds of document that need completely different readers, and sending one
to the other's reader fails *silently* in both directions. [`document_router.py`](files/document_router.py)
decides which is which — on ISO section headings and page count, two orthogonal signals, with
disagreements reported rather than resolved.

```
                       document_router.classify()
                        /                      \
          process document                   workflow diagram
                 |                                  |
        adaptive_chunker.py                 workflow_extractor.py
        layout-aware sectioning             vision model reads the topology
                 |                                  |
            chunks.json (152)              workflow_chunks.json (175)
                 \                                  /
                  \------> retriever.py <----------/
                        BM25 + MiniLM/FAISS hybrid
                                  |
                            chatbot.py  ->  app.py / demo_chatbot.ipynb
```

Why the split matters: a flowchart's meaning is in the **edges** — which arrow points where,
which diamond loops back on "Not Approved", which red monetary label decides who signs. A text
extractor returns the labels with every relationship destroyed, which reads like content and
answers nothing.

```bash
python document_router.py
```

## 3. Modules

| File | What it does |
|---|---|
| [`document_router.py`](files/document_router.py) | Routes each PDF to its extractor; reports files where the two signals disagree |
| [`adaptive_chunker.py`](files/adaptive_chunker.py) | Process PDFs → `chunks.json`. Finds section boundaries from layout, labels them against a canonical taxonomy. Audits itself on every run |
| [`workflow_extractor.py`](files/workflow_extractor.py) | Workflow PDFs → `workflow_chunks.json`, one chunk per lane. VLM emits a structured graph; the embedded prose is rendered deterministically from it |
| [`extract_pipeline.py`](files/extract_pipeline.py) | Independent OCR-health audit — which pages carry no embedded text |
| [`retriever.py`](files/retriever.py) | Hybrid BM25 + MiniLM/FAISS retrieval, doc-code routing, embeddings cached in `.embed_cache/` |
| [`contextualize.py`](files/contextualize.py) | Follow-up resolution — rewrites "in the self execution process" into a standalone query, only when a cheap gate fires |
| [`translate.py`](files/translate.py) | Arabic in (Egyptian dialect), MSA out. Form numbers masked before translation, restored after |
| [`validator.py`](files/validator.py) | Hallucination proxy: extracts form numbers from each answer, checks them against those occurring in the corpus. Flags, does not block |
| [`chatbot.py`](files/chatbot.py) | The pipeline itself — one copy of the prompt text, the reasoning-strip regex, the validation order |
| [`app.py`](files/app.py) | Streamlit front end |
| [`demo_chatbot.ipynb`](files/demo_chatbot.ipynb) | The explanation — rebuilds the pipeline end to end and shows the reasoning |

## 4. Quick start

```bash
pip install -r files/requirements.txt
```

```bash
ollama pull qwen3:14b
```

```bash
streamlit run files/app.py
```

Full install, model choice, latency figures and the Arabic evaluation are in
[files/README_HANDOFF.md](files/README_HANDOFF.md).

## 5. Rebuilding the artifacts

Both chunk files ship prebuilt, so **you can skip this**.

```bash
python adaptive_chunker.py --audit audit.json          # ../processes_pdf -> chunks.json
```

```bash
python workflow_extractor.py --src ../Workflows --audit workflow_audit_full.json --resume
```

The workflow run is the expensive one: one vision-model call per page over 133 pages. It
checkpoints the audit file after **every document**, and `--resume` skips what is already in
there — a run of this length will be interrupted. To re-render the prose without touching the
model:

```bash
python workflow_extractor.py --from-audit workflow_audit_full.json
```

**The workflow chunks are the only part of the corpus that is not ground truth.** They carry
`source_type: "vlm_description"` and `review: "machine"`, they are excluded from the validator's
form whitelist, and the app labels any answer built from them. 71 of 133 pages extracted with no
audit warning at all; the rest are tracked in `workflow_warnings_report.pdf`, and the failure is
concentrated and understood — see below.

## 6. Known issue: dense diagrams are under-extracted

60 of 131 workflow documents carry at least one audit warning, and the distribution is not
uniform. Grouping pages by the size of the PDF canvas:

| longest page edge | pages | clean | mean warnings |
|---|---|---|---|
| under 1700 pt (A4/A3) | 85 | 58 | 0.4 |
| 1700–3000 pt | 27 | 12 | 0.8 |
| over 3000 pt | 21 | **1** | **8.9** |

These are Visio exports on canvases up to 7082 × 4642 pt — roughly 66 × 64 inches. Rendered at
`RENDER_DPI = 140` and then downscaled by the vision model's own preprocessing, a 16 pt label
lands at a couple of pixels tall. The model does not hallucinate on these pages so much as it
stops seeing them: on one 399-line diagram it returned 16 nodes and 21 edges, and 136 printed
labels plus 27 monetary thresholds never made it into the graph.

**Two fixes are implemented, both built on the fact that the missing content is already available
exactly**: 130 of 133 pages carry a native text layer with word-level coordinates, and the
extractor previously used it only as an after-the-fact audit channel.

1. **Tiling** (`plan_tiles`, `render_tiles`). What matters is not render DPI but the ratio of
   label height to page extent, because the model's input size is fixed regardless of what it is
   sent. So an oversized page is cut into tiles small enough that a label survives that
   downscale. Cuts are placed in **gutters** — empty runs no box or line crosses — so a boundary
   never bisects a node; tiles then overlap, because an *arrow* crossing a boundary is
   unavoidable and the overlap keeps both endpoints visible somewhere. `merge_graphs` rejoins
   the tile graphs on **normalised label text**, not on ids, which the model slugs per call.
2. **Checklist seeding** (`build_prompt`). The text-layer strings for each tile are handed to the
   model up front as a list it must place — turning the audit's post-hoc complaint into a
   constraint. It is explicitly *not* a transcription source: the text layer carries labels with
   every relationship destroyed, so it can say what is on the page and never where an arrow
   points.

Planning across all 133 pages: 92 stay single-tile, and 61 of the 71 previously-clean pages are
untouched. The worst offenders split into 6–12. Total vision calls go 133 → 230.

```bash
python workflow_extractor.py --src ../Workflows --no-tile   # the pre-tiling behaviour, for A/B
```

**Measured on one document so far** — `Request for Quotation - Purchasing RFQ WF - FC in
Damietta.pdf`, a 4046 pt page that split into 3 tiles:

| | nodes | edges | thresholds | warnings |
|---|---|---|---|---|
| before | 14 | 22 | 5 | 10 |
| after | 68 | 112 | 9 | 7 |

Roughly five times the topology recovered, and the warning count fell rather than rose — the
extra nodes are not extra noise. It cost 1775 s for the page, all of it CPU (see 6b).

**Not yet measured across the corpus.** Until a full re-run is compared against
`workflow_audit_full.json`, treat any answer sourced from a document listed in
`workflow_warnings_report.pdf` as unverified — the warnings are advisory, and a partially-read
graph still becomes chunks.

One quality issue surfaced by that run, since **fixed**: the merged graph kept a lane named after
the whole page ("Construction Company Approval Routing Flowchart") alongside the six real flows.
`resolve_shared_lanes` drops such a pseudo-lane by comparing its name to `graph["title"]`, and
that comparison cannot fire when the model returns a null title — which it does on tiled pages.
It now also matches the filename stem, and, more importantly, treats **full absorption alone** as
sufficient when the model reported no title: a lane whose every node was pulled into sibling
lanes has no content that is not already elsewhere, so dropping it cannot lose anything. Replayed
over all 133 existing pages the change is inert (175 lanes before and after), so it carries no
regression risk for the untiled corpus.

Two further options, not implemented: closing the loop (re-prompt with exactly the labels
`audit()` reports missing, merge, re-audit), and skipping the VLM entirely on native pages by
reconstructing the graph from the vector layer — rects → nodes, polylines → edges, arrowhead
triangles → direction. The latter would make these chunks ground truth rather than
`review: "machine"`.

## 6b. Running on an Intel Arc GPU

Ollama ships a Vulkan backend (`lib/ollama/vulkan/ggml-vulkan.dll`) but gates it behind two
environment variables, and skips integrated GPUs by default even when a backend is present:

```bash
setx OLLAMA_VULKAN 1 && setx OLLAMA_IGPU_ENABLE 1
```

Restart the Ollama server fully afterwards — `setx` only affects new processes. Confirm with
`/api/ps`, which is the only reliable check: `torch` cannot tell you, because Ollama ships its
own runtime.

Measured on an Arc 140T (integrated, shared system memory): `qwen3:14b` loads at
`vram=9.6GB / total=9.6GB`, fully resident. An iGPU shares memory bandwidth with the CPU and
token generation is bandwidth-bound, so this is a real speedup but not a discrete-GPU speedup —
do not calibrate against the GPU column of the latency table in `README_HANDOFF.md`, which was
written for a dedicated card.

**The vision model cannot use it.** `qwen3.6:27b` loads at 94% on the GPU and then dies partway
through every request:

```
slot process_mtmd: id 0 | task 0 | encoding mtmd batch from idx = 4
[GIN] 500 | POST "/api/generate"
```

The LLM half runs fine; Vulkan cannot execute the multimodal projector. llama.cpp has
`--no-mmproj-offload` for exactly this and the flag is present inside the Ollama binary, but no
`OLLAMA_*` variable exposes it, so the projector cannot be placed separately. `workflow_extractor.py`
therefore pins **only the vision requests** to the CPU with `VISION_NUM_GPU = 0` — set per
request, because `OLLAMA_VULKAN` is global and the interactive path wants it. Set it to `None`
on a CUDA machine.

Consequence for the extraction: it stays a CPU job, at roughly **9 minutes per tile**. Tiling
takes the corpus from 133 calls to 230, so a full re-run is on the order of a day and a half.
That cost is the strongest argument for the vector-layer extraction described above, which needs
no model at all.

`retriever.py` passes `device=None` and lets sentence-transformers choose; `Retriever(device="xpu")`
pins MiniLM to the Arc, which is worth doing only for full corpus re-embeds.

## 7. Evaluation

`eval_set_v2.json` — 100 questions over 68 of the 71 process documents. Every question was
built from chunk text and every `must_include` string verified to occur verbatim in its gold
document; that check runs *inside* the harness, because an eval set that has drifted from the
corpus reports a low score rather than an error. All records are `review: "machine"`.

```bash
python eval_retrieval.py     # retrieval only, no Ollama, ~13 ms/question
```

```bash
python eval_generation.py    # end to end, needs Ollama, ~47 min on an Arc iGPU
```

| | before router fix | after |
|---|---|---|
| retrieval, gold doc in top-3 | 79/92 = 85.9% | **87/92 = 94.6%** |
| answer, all `must_include` found | 75/92 = 81.5% | not re-run |
| refusal on unanswerable | 8/8 = 100% | not re-run |

Do not compare against the old `model_eval_results.json` (100% on 36 questions over 9 documents).
Top-3 of 9 is a ~33% chance baseline; this is 68 documents.

> `generation_eval_results.json` was produced **before** the router fix below. Its retrieval
> column is superseded; re-run `eval_generation.py` (~47 min) for current end-to-end numbers.

### What the first run found

The 17 misses attributed cleanly, which is the point of scoring retrieval and generation apart:
**11 retrieval**, **2 chunk-granularity** (right document, wrong chunk, model correctly refused),
**4 generation**.

**8 of the 11 retrieval failures were one bug.** `route_prefixes` hard-restricted the search to a
doc-code family chosen from `PREFIX_HINTS`, a keyword table covering 6 families. The 9-document
corpus had 6; the full corpus has 19. So a keyword that used to be unambiguous now excludes most
of the index — `"handover"` routed to PTN and dropped PEN/PFW/PLO, `"subcontract"` routed to PCN
and dropped POP. Explicit codes were broken too: `"what is P-OP-02"` intersected with the stale
six-family list and routed nowhere.

Fixed by making hard routing fire **only on an explicit doc code** in the query, which cannot be
ambiguous, and leaving keyword hints to `soft_prefixes`, which boosts without excluding. Prefix
validity is now checked against the loaded corpus rather than a module constant, so the list
cannot silently go stale again. Retrieval went 87.0% → 94.6%, and `numeric` — the weakest type —
went 81% → 94%.

The remaining open item is unchanged and is now measured: **no chunk carries a step number**, so
a duration sits inside one large `PROCESS OPERATION` chunk with no term distinctive to its
document. That is what the 2 chunk-granularity misses are, and step-level chunking is the fix.

### The validator's limit, with evidence

On `PVMO01-1` the model answered `F-P-TN-02-04`; the correct form is `F-P-TN-02-05`. Both are
real, so the whitelist passed it as `clean`.

`validate_answer` now takes the retrieved chunks and reports a third verdict, **`ungrounded`** —
a form that exists in the corpus but was not in the context the answer was built from, which the
prompt forbids. That is strictly stronger than the whitelist.

**It does not catch `PVMO01-1`**, and it is worth being precise about why. The source lists the
two forms two lines apart:

```
Tender sourcing strategy   F-P-TN-02-04
Project sourcing strategy  F-P-TN-02-05
```

Both were in the retrieved context, so the model picked the adjacent, confusable entry out of
material it was genuinely shown. No form-number check can close that gap. **Read a `clean`
verdict as "cited nothing impossible", never as "correct"** — and note that `policy="block"`
would not have caught this either.

## 8. Repository conventions

- `.gitattributes` pins line endings (LF in the repo) and marks every binary type, because the
  repo is worked on from Windows and tracks 400+ PDFs.
- Machine-produced records are tagged `"review": "machine"` — in `eval_set_ar.json` and in every
  workflow chunk — and become `"human"` only after a person has checked them.
- Regenerated files are listed in `.gitignore` with the command that writes them.

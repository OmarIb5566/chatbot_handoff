# Status audit — 3 September 2026

An independent re-measurement of this repository at `1b83cba`. Every number below was produced by
running the code today, not read off [README.md](README.md). Where the two disagree, the
disagreement is the finding.

**Overall.** The retrieval half of this system is finished and honestly measured. The delivery half
is not: the Arabic answer path failed on the first live question put to it, 29 workflow diagrams
sitting in the repository are never indexed, and answers take 90–125 seconds. This is a strong
internal pilot for a warned audience, not something to put in front of a thousand colleagues.

---

## 1. Confirmed working

**Retrieval reproduces its published numbers exactly.** All three eval sets, run against
`chatbot.load_corpus()`, land on the README's figures to the question, at ~15 ms/query.

```
eval_set_v2.json        86/92 top-3  (93.5%)   77/92 top-1
eval_set.json           27/32 top-3  (84.4%)   20/32 top-1
eval_set_workflow.json  18/18 top-6  (100%)     6/18 top-1
```

**The test suite passes and is genuinely hermetic.** 48 tests from 43 test functions, under a
second, no model and no network. It ran clean on the first attempt with no environment setup.

**The version-collision guard works on the case that caused it.** Asked *"Who approves a subcontract
amendment?"*, the pipeline named the version it answered from and listed the sibling it did not use:

```
chosen: Subcontract Amendments V1 (USD in Damietta Project)
others: Rental Equipment Variation Order WF
answer opens: "This describes the ... V1(USD in Damietta Project) version"
```

**The corpus repair is real, and understated.** Recomputed directly from the two JSON files.
Self-loops are gone, duplicate labels nearly so, and by a strict definition of "structurally clean"
— no duplicate node labels, no self-loops, no node with neither an inbound nor an outbound edge —
the repaired corpus scores **114/137**, better than README §9's own claim of 96/137. Which
definition is authoritative is worth pinning down.

**The UI renders.** README §8 lists the Streamlit front end as never visually verified. It was
loaded in a browser at 1280×900: header, corpus counts ("71 process documents and 110 workflow
diagrams"), four example prompts and the chat input all render correctly. The idle state is now
verified; an answer-rendering state still is not.

---

## 2. Broken, or claimed but unverified

Ordered by how much damage each does to the stated goal.

### Arabic questions come back in English

The largest gap between what the repository claims and what it does. Asked
`ما هي خطوات طلب امر تغيير للمقاول؟` — the language was detected, the translation into English for
retrieval was correct, retrieval worked, and the answer came back entirely in English.

```
lang: ar
question_en: "What are the steps to request a contractor change order?"
answer:      "The workflow begins with the identification of the source
              of variation. 1. TOM identifies the affected items..."
```

The cause is structural, not incidental. Arabic output is *requested* by appending
`ANSWER_IN_ARABIC` to the prompt (`chatbot.py:504`) and nothing checks the result. The only call to
`to_arabic()` is on the blocked-refusal path (`chatbot.py:735`), which is English text produced by
`validator.py` rather than by the model. So when the model ignores the instruction — as it did here,
plausibly because the English `WORKFLOW_FORMAT` rules dominate the prompt — the failure is silent.
`evals/eval_arabic.py` scores *retrieval* on translated questions only; no eval has ever checked
that an answer comes back in Arabic.

### 29 workflow diagrams in the repository are never indexed

`Workflows/overseas/` holds 29 tracked PDFs — the UAE, KSA and Côte d'Ivoire variants. The extractor
globs non-recursively (`workflow_vector.py:878`) so it reads 110 files and never sees them; the
router globs recursively (`document_router.py:178`) and counts them. Nothing errors and nothing
appears in an audit. The corpus is 21% smaller than the repository's own contents, and the two
READMEs disagree about it: [README.md](README.md) §1 says 110 diagrams and never mentions
`overseas/`, while [README_HANDOFF.md](README_HANDOFF.md) says 139, which is exactly 110 + 29.

A user asking about the KSA subcontract process gets the Egyptian answer with no indication the KSA
document exists — the same failure mode as the sibling mix-up the variant guard was built to fix.

### Legend boxes are being read as workflow steps

README §8 flags `Variation Order Flow Chart.pdf` as being in the wrong pipeline. Live, it is worse
than documented: the diagram's *legend* — the key explaining what a diamond means — was parsed into
graph nodes and emitted into the answer's "Returns and loops" section as process content.

```
Returns and loops:
- "Legend Start/End A Close No Process Contract Department
   Document (Input) Document (Output)" -> ...
```

A missing edge is an omission; this is fabricated structure in the visible answer text. The caveat
machinery did fire alongside it — 13 steps reported with no route in or out — so the answer is
labelled unreliable. It is still unreadable at that point.

### 90 to 125 seconds per answer

Measured on `qwen3:14b`: 90 s for the English question, 124 s for the Arabic one, which pays for an
extra translation call. The README is upfront about this, but it is worth stating plainly against
the goal — a thousand colleagues will not wait two minutes, and there is no queueing, no streaming
and no concurrency story in the app.

### Answer quality has no current measurement

Both generation eval records in `eval_results/` date from 23 August. The served corpus
(`workflow_chunks_fixed.json`) was rebuilt on 1 September. Every generation number in the repository
therefore describes a corpus that is no longer loaded. Retrieval was re-scored after the rebuild;
generation was not.

### Top-1 retrieval is weak where the stakes are highest

Workflow questions hit 18/18 at top-6 but only 6/18 at top-1. Because the model sees all six chunks,
a plausible wrong sibling is almost always in the context window. The variant guard mitigates this;
it does not remove it. README §8 is right that document-level scoring flatters this population — 16
of 18 questions have another diagram carrying the same edge.

### Tests cover the coordination layer, not the engine

The suite imports only `chatbot`, `config` and `errors`. That leaves `retriever.py`,
`contextualize.py`, `validator.py`, `workflow_vector.py`, `adaptive_chunker.py`, `translate.py` and
`document_router.py` — roughly 6,300 of 6,880 backend lines — with no unit tests at all. The
non-recursive glob above is exactly the kind of bug a two-line test would have caught.

### 19 chunks still contain a step connected to nothing

Across 17 distinct files, mostly the `End` node, concentrated in the IIR family. README §8's
analysis holds — on most of them the arrow was never drawn in the PDF's vector layer, so no
extractor change recovers it. It is disclosed to the user rather than fixed, which is the right
call.

---

## 3. Claimed against measured

| Claim in README | Stated | Measured | Verdict |
|---|---|---|---|
| `eval_set_v2` top-3 | 86/92 | 86/92 | matches |
| `eval_set` top-3 | 27/32 | 27/32 | matches |
| workflow top-6 | 18/18 | 18/18 | matches |
| test count | 48 | 48 pass | matches |
| self-loops after repair | 0 | 0 | matches |
| duplicate-label chunks | 34 → 4 | 34 → 4 | matches |
| structurally clean chunks | 96/137 | 114/137 | **understated** |
| workflow diagrams indexed | 110 | 110 of 139 present | **29 excluded** |
| Arabic answer generated in Arabic | works | English out | **failed** |
| UI visually verified | never | idle state OK | improved |

---

## 4. Distance to the goal

The stated goal: a locally-run bilingual assistant over RME's ISO corpus, for roughly a thousand
colleagues, answering only from the documents with sources cited and unverifiable form numbers
flagged.

| Area | State |
|---|---|
| Retrieval | done, measured |
| Grounding and citation | works, caveats fire |
| Corpus completeness | 110 of 139 diagrams |
| Diagram fidelity | legend text leaking into answers |
| Arabic — the second half of "bilingual" | retrieval only; generation unverified and failing |
| Serving a thousand people | 90–125 s, single session, no queueing |

---

## 5. What to fix first

1. **Check the answer's language and fall back.** One call to `is_arabic()` on the generated answer,
   routing through `to_arabic()` when it fails. Both are already imported in `chatbot.py`. This
   turns a silent failure into a slower correct answer, and it is the difference between the product
   being bilingual and claiming to be.
2. **Decide what `Workflows/overseas/` is.** Either index the 29 diagrams — one character, `rglob` —
   or move them beside `processes_pdf/other/` and say in the README that they are parked. Right now
   they are neither, and the router and the extractor disagree in code.
3. **Re-run the generation evals against the September corpus.** The harnesses exist. Their recorded
   output describes a corpus that has since been rebuilt, so the repository currently has no honest
   claim about answer quality.
4. **Filter legend geometry out of the graph.** Legend boxes are visually separable — isolated,
   unconnected, clustered in a page corner. Dropping them stops fabricated structure from reaching
   the answer text, which matters more than the missing `End` arrows.
5. **Put a test around corpus scope.** Assert that the number of PDFs the extractor sees equals the
   number the router classifies. That single assertion would have caught the 29 missing diagrams the
   day the folder appeared.

---

## 6. Housekeeping

- **Documentation.** [README.md](README.md) is unusually good — specific, self-critical, and it
  survived independent checking on almost every number. [README_HANDOFF.md](README_HANDOFF.md) marks
  its own staleness at the top, which is honest, but it now describes a corpus that has not existed
  for weeks.
- **Stray artifacts.** `workflow_audit_full.json` (866 KB) and `workflow_audit_new.json` sit at the
  repository root, are referenced by nothing outside `_verify/`, and are absent from `paths.py`
  while every other audit lives in `data/`.
- **Scratch layer.** `_verify/` is 60 tracked files and 2,066 lines — a fifth of the Python here.
  README §10 defends it as the evidence behind the published numbers, which is defensible, but it
  should probably be pruned to the scripts that still reproduce something.
- **Missing constant.** `paths.py` has no entry for `evals/eval_set_workflow.json`, though README §7
  documents running it.

---

## Method

Tests run with `pytest` from the repository root. Evals via `evals/eval_retrieval.py` against
`chatbot.load_corpus()`. Graph health recomputed directly from `data/workflow_chunks.json` and
`data/workflow_chunks_fixed.json`. Two live questions through `Chatbot.ask()` on `qwen3:14b` with
Ollama running locally. UI loaded in a browser at 1280×900.

Not checked: `demo_chatbot.ipynb`, the extraction pipeline end to end, and answer quality at scale.

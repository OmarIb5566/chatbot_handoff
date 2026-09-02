"""Shared fixtures. Nothing here loads a model or touches the network.

WHY THE TESTS ARE SHAPED THIS WAY
---------------------------------
The expensive parts of this pipeline - MiniLM, the FAISS index, Ollama - are
exactly the parts an automated test cannot assert much about: retrieval
quality is measured by the harnesses in evals/, against real question sets,
and answer quality needs a model. What is left is the logic BETWEEN those
pieces, and it is where every regression found so far actually lived:

  * drop_unreadable matched a page-scoped warning as if it were chunk-scoped
    and deleted three good chunks to remove one bad one
  * the coverage caveat printed the same page warning once per chunk
  * fit_context has to keep the top hit even when it alone busts the budget

None of those need a model to catch, and all three are one assertion each. So
these tests run in under a second on synthetic chunks, which means they can be
run on every change rather than once a week.

`chunk()` and `wf_chunk()` build the two chunk shapes the pipeline deals with.
They deliberately mirror the real fields rather than a simplified version: a
test corpus that has drifted from the real one tests nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))


def chunk(filename="P-HR-01 Leave Policy.pdf", text="Annual leave is 21 days.",
          section="Leave", **kw):
    """A process-text chunk: verbatim PDF text, no graph."""
    c = {"filename": filename, "section": section, "text": text,
         "source_type": "pdf_text", "page": 1, "doc_code": "P-HR-01"}
    c.update(kw)
    return c


def wf_chunk(filename="Subcontracts - Subcontract Preparation V1.pdf",
             family="Subcontracts", variant="Subcontract Preparation V1",
             nodes=("start", "review", "end"),
             edges=(("start", "review"), ("review", "end")),
             coverage=1.0, warnings=(), page=1, section="Approval flow", **kw):
    """A workflow chunk with a real graph.

    `nodes` are ids; the label is derived from the id so a test can assert on
    readable text. `edges` are (from, to) pairs - any node absent from them is
    a disconnected step, which is the condition workflow_gaps reports.
    """
    c = {"filename": filename, "section": section, "page": page,
         "text": f"Approval workflow: {section}.", "source_type": "pdf_vector",
         "family": family, "variant": variant, "coverage": coverage,
         "audit_warnings": list(warnings), "review": "machine",
         "graph": {"name": section,
                   "nodes": [{"id": n, "label": n.replace("_", " ").title(),
                              "kind": "step"} for n in nodes],
                   "edges": [{"from": a, "to": b} for a, b in edges]}}
    c.update(kw)
    return c


@pytest.fixture
def real_corpus():
    """The actual workflow corpus, or skip.

    Generated, so it is not in the repo and a fresh clone will not have it.
    Skipping is right: these assertions are about the corpus, and absent
    corpus means the assertion has nothing to say, not that it failed.
    """
    import json

    from paths import WORKFLOW_CHUNKS_ACTIVE
    if not WORKFLOW_CHUNKS_ACTIVE.exists():
        pytest.skip("workflow corpus not built on this checkout")
    return json.loads(WORKFLOW_CHUNKS_ACTIVE.read_text(encoding="utf-8"))

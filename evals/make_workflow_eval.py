"""
Build eval_set_workflow.json: gold questions whose answer is a DIAGRAM.

    python make_workflow_eval.py
    python make_workflow_eval.py --n 20 --out eval_set_workflow.json

WHY THIS EXISTS
---------------
eval_set_v2.json has 100 questions and not one of them has a workflow PDF as
its gold document - every `doc` points into processes_pdf/. So the corpus is
1063 process chunks plus 137 workflow chunks, and the measured half is the
first one only.

That matters more than a missing test usually does, because the workflow half
is where the recall complaint came from. Without this file, "diagrams retrieve
badly" and "diagrams retrieve fine" are equally unfalsifiable, and any fix -
including the boost and the floor added to retriever.py - is tuned against
whichever handful of questions someone typed by hand that afternoon. That is
the same premature-calibration trap the diagram classifier is already being
kept away from.

WHAT IS AND IS NOT GUARANTEED
-----------------------------
Same convention as the other machine-generated sets: every question is built
FROM chunk text, and every `must_include` string is checked to occur verbatim
in the gold document's own chunks before the question is kept. So the answers
are grounded, not invented.

What is NOT guaranteed is that the question is fair or that the gold document
is the only one that answers it. Approval flows repeat across the corpus - a
dozen Request-for-Quotation variants share most of their node names - so a
retriever can return a *different* RFQ diagram that is arguably just as
correct and score as a miss. Read the number as a floor, and promote questions
to review="human" as they are checked.

The generator also refuses to build questions from degenerate chunks (empty
node labels, e.g. "Steps: ; ."). Those exist - workflow_vector.py could not
read some pages - and a question generated from one would be unanswerable by
construction, which would look like a retrieval failure forever.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from paths import EVALS, WORKFLOW_CHUNKS_JSON  # noqa: E402

ORDER_RE = re.compile(r"^\s*(\d+)\.\s+(.+?)\s*$", re.M)
STEPS_RE = re.compile(r"^Steps:\s*(.+?)$", re.M)
ROUTE_RE = re.compile(r"^\s*-\s+(.+?)\s+->\s+(.+?)(?:\s+\[.*\])?\s*$", re.M)

# A node label that carries no information. Questions built on these are
# unanswerable by construction.
JUNK = {"", "end", "start", "auto-creation", "creation", "yes", "no"}


def clean(label: str) -> str:
    return re.sub(r"\s+", " ", label).strip(" .;")


def ordered_steps(text: str) -> list[str]:
    return [clean(m.group(2)) for m in ORDER_RE.finditer(text)]


def usable(label: str) -> bool:
    return bool(label) and label.lower() not in JUNK and len(label) > 2


def questions_for(chunk: dict) -> list[dict]:
    """Every question this one chunk can support. May be empty."""
    text = chunk["text"]
    doc = chunk["filename"]
    # Strip the " - WF" / " WF V1" tail so the question names the PROCESS the
    # way a person would, not the filename. Asking "in the 'Bonds Request -
    # Bonds Request WF.pdf' workflow" would leak the gold document into the
    # query and measure string matching instead of retrieval.
    topic = clean(re.split(r"\s+-\s+", chunk.get("title") or doc)[0])
    out: list[dict] = []

    steps = [s for s in ordered_steps(text) if usable(s)]
    if len(steps) >= 2:
        out.append({
            "question": f"In the {topic} approval workflow, who signs after {steps[0]}?",
            "must_include": [steps[1]],
            "type": "workflow_order",
        })
    if len(steps) >= 3:
        out.append({
            "question": f"What is the order of approval in the {topic} workflow?",
            "must_include": steps[:3],
            "type": "workflow_order",
        })

    routes = [(clean(a), clean(b)) for a, b in ROUTE_RE.findall(text)]
    routes = [(a, b) for a, b in routes if usable(a) and usable(b)]
    if routes:
        a, b = routes[0]
        out.append({
            "question": f"In the {topic} workflow, where does the request go after {a}?",
            "must_include": [b],
            "type": "workflow_routing",
        })

    m = STEPS_RE.search(text)
    if m:
        roles = [clean(s) for s in m.group(1).split(";")]
        roles = [r for r in roles if usable(r)]
        if len(roles) >= 3:
            out.append({
                "question": f"Which roles take part in the {topic} approval workflow?",
                "must_include": roles[:2],
                "type": "workflow_roles",
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--out", default=str(EVALS / "eval_set_workflow.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    chunks = json.load(open(WORKFLOW_CHUNKS_JSON, encoding="utf-8"))
    by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        by_doc.setdefault(c["filename"], []).append(c)

    degenerate = sum(1 for c in chunks if not [s for s in ordered_steps(c["text"]) if usable(s)]
                     and not ROUTE_RE.search(c["text"]))
    print(f"{len(chunks)} workflow chunks over {len(by_doc)} documents "
          f"({degenerate} with no readable graph)")

    candidates: list[dict] = []
    for doc, cs in by_doc.items():
        # One question per DOCUMENT, not per chunk. Several chunks of the same
        # diagram would otherwise produce near-identical questions and inflate
        # the score with what is really one measurement repeated.
        pool = [q for c in cs for q in questions_for(c)]
        if not pool:
            continue
        blob = "\n".join(c["text"] for c in cs)
        pool = [q for q in pool if all(inc in blob for inc in q["must_include"])]
        if pool:
            candidates.append({**pool[0], "doc": doc})

    random.seed(args.seed)
    random.shuffle(candidates)
    picked = candidates[:args.n]
    for i, q in enumerate(picked, 1):
        q["id"] = f"WF-{i:02d}"
        q["review"] = "machine"

    order = ["id", "doc", "question", "must_include", "type", "review"]
    picked = [{k: q[k] for k in order} for q in picked]

    Path(args.out).write_text(json.dumps(picked, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"wrote {len(picked)} questions -> {args.out}")
    print(f"  (from {len(candidates)} documents that could support one)")


if __name__ == "__main__":
    main()
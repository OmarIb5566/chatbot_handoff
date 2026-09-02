"""
Retrieval accuracy over the full process corpus, on eval_set_v2.json.

    python eval_retrieval.py
    python eval_retrieval.py --chunks chunks.json      # compare against the old artifact
    python eval_retrieval.py --top-k 5 --misses

WHY RETRIEVAL ONLY, AND WHY THAT IS NOT A COP-OUT
-------------------------------------------------
No Ollama, no generation, ~13 ms per question - so this runs in seconds and can
be used as a regression check on every chunker change. The gold field is the
document a question is answerable from, which is language-independent (the same
property eval_arabic.py relies on) and, unlike an answer string, is not a matter
of judgement.

It is also the half that fails first. A generation score cannot distinguish "the
model wrote a bad answer" from "the model never saw the right chunk", which is
why app.py ships the retrieved chunks next to every answer. Measure the
retrieval floor separately and the generation number afterwards means something.

WHAT THE NUMBER IS, AND WHY IT IS LOWER THAN THE OLD ONE
--------------------------------------------------------
    top-1  74/92 = 80.4%
    top-3  80/92 = 87.0%
    top-5  80/92 = 87.0%

The old harness reported 32/32 = 100% - on 32 questions over 9 documents, where
top-3 of 9 is a ~33% chance baseline before any signal at all. This set is 100
questions over 68 of the 71 process documents, so the task is a different and
much harder one and the two numbers are not comparable. Read 87% as the first
real measurement, not a regression.

TOP-3 AND TOP-5 ARE IDENTICAL, WHICH IS THE INTERESTING PART
-------------------------------------------------------------
Widening the window recovers nothing. The twelve misses are not near-misses
sitting at rank 4; the gold document is nowhere in the top 5. So raising `top_k`
is not the fix, and the questions that fail have a shape:

    numeric      29/36  81%   <- weakest
    form_lookup  22/24  92%
    role_lookup  13/13 100%
    definition    9/9  100%

A question like "within how long must the PCE submit the time impact study"
carries almost no term that is distinctive to its document - "within", "days",
"submit" occur in all 71. Role and definition questions name the thing they are
about and retrieve perfectly. That points at the fix being step-level chunking
or a doc-code router entry, not a bigger k.

THE EVAL SET IS review="machine"
--------------------------------
Every question was generated from chunk text and its `must_include` strings were
verified to occur verbatim in the gold document (see `verify()`), so the answers
are grounded rather than invented. That is not the same as a person having
confirmed the question is fair or the gold document is the best one. Same
convention as eval_set_ar.json: promote to "human" once checked, and the harness
reports that subset separately as soon as one exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

# backend/ holds the pipeline; its modules import each other by bare name.
# See paths.add_backend_to_path for why that is worth keeping.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))


import argparse
import collections
import json
import re
import time
from pathlib import Path

from paths import (EVAL_SET_V2 as DEFAULT_EVAL, CHUNKS_JSON as DEFAULT_CHUNKS,
                   WORKFLOW_CHUNKS_JSON)


def load_chunks(chunks_path, items) -> tuple[list[dict], bool]:
    """The production corpus, always - not a subset chosen per eval set.

    This used to merge workflow chunks in only when an eval set's gold
    documents needed them, on the reasoning that skipping the merge was a free
    optimization. It was not free: it meant eval_set_v2 - a process-only eval
    set - was never once scored against the corpus production actually serves.
    Cowork's verification run found the gap directly: top-3 on eval_set_v2 was
    88/92 against the 1063-chunk process-only index this function used to
    build, and 86/92 against the real 1200-chunk merged index. Two questions
    were regressing in a way nothing in this repo could see, because nothing
    ever tested the corpus that was actually shipped. (The 80/92 figure in
    this file's own module docstring above predates the workflow corpus
    entirely and should be read as historical, not current.)

    Calls chatbot.load_corpus() rather than reimplementing the merge, so the
    degenerate-chunk filter (drop_unreadable) can't drift out of sync between
    what eval scores and what production serves - that exact kind of drift is
    what caused this bug in the first place.

    `chunks_path` is kept only for `--chunks` overrides in ad hoc runs (e.g.
    scoring a chunker experiment before it has been wired into load_corpus);
    the default path always goes through the real corpus.
    """
    if chunks_path == DEFAULT_CHUNKS:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
        from chatbot import load_corpus
        return load_corpus(), True
    return json.load(open(chunks_path, encoding="utf-8")), False


def verify(items: list[dict], chunks: list[dict]) -> list[str]:
    """Every gold doc must exist, and every must_include must occur in it.

    Run before scoring, not as a separate script, because an eval set that has
    drifted from the corpus reports a LOW SCORE rather than an error - the
    retriever cannot return a document that is not there, so a renamed file
    looks exactly like a retrieval failure. That happened while this set was
    being built: two questions pointed at a filename that no longer existed.
    """
    body: dict[str, str] = collections.defaultdict(str)
    for c in chunks:
        body[c["filename"]] += " " + c["text"]
    norm = {k: re.sub(r"\s+", " ", v).lower() for k, v in body.items()}

    problems = []
    for it in items:
        if it["doc"] == "none":
            if it["must_include"]:
                problems.append(f"{it['id']}: unanswerable but has must_include")
            continue
        if it["doc"] not in norm:
            problems.append(f"{it['id']}: gold doc not in corpus - {it['doc']}")
            continue
        for s in it["must_include"]:
            if s.lower() not in norm[it["doc"]]:
                problems.append(f"{it['id']}: {s!r} does not occur in {it['doc']}")
    return problems


def score(items: list[dict], search_fn, top_k: int) -> tuple[int, int, list[dict]]:
    hits, misses = 0, []
    for it in items:
        found = [x["filename"] for x in search_fn(it["question"], top_k=top_k)]
        if it["doc"] in found:
            hits += 1
        else:
            misses.append(it)
    return hits, len(items), misses


def main() -> None:
    from translate import enable_utf8_stdout

    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="Retrieval accuracy on the process corpus")
    ap.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    ap.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--misses", action="store_true", help="list every miss")
    args = ap.parse_args()

    items = json.load(open(args.eval, encoding="utf-8"))
    chunks, merged = load_chunks(args.chunks, items)
    if merged:
        print("gold documents outside the process corpus - workflow chunks merged in\n")

    problems = verify(items, chunks)
    if problems:
        print(f"EVAL SET DOES NOT MATCH {args.chunks.name} - {len(problems)} problem(s):")
        for p in problems[:20]:
            print("   ", p)
        print("\nScores below would understate accuracy. Fix the set first.\n")

    # Unanswerable questions have no gold document, so they are not scoreable
    # here at all - they measure refusal, which needs generation.
    answerable = [it for it in items if it["doc"] != "none"]
    print(f"{args.eval.name}: {len(items)} questions, {len(answerable)} with a gold "
          f"document, {len(items) - len(answerable)} unanswerable (not scored here)")
    print(f"{args.chunks.name}: {len(chunks)} chunks over "
          f"{len({c['filename'] for c in chunks})} documents\n")

    from retriever import Retriever

    r = Retriever(chunks)

    print("document accuracy")
    print("-" * 58)
    for k in (1, args.top_k, 5):
        h, n, _ = score(answerable, r.search, k)
        print(f"  top-{k:<2} {h:>3}/{n} = {h / n:>6.1%}")

    t0 = time.perf_counter()
    h, n, misses = score(answerable, r.search, args.top_k)
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"\n  {ms:.0f} ms/query at top-{args.top_k}")

    print("\nby question type (top-%d)" % args.top_k)
    print("-" * 58)
    per = collections.defaultdict(lambda: [0, 0])
    for it in answerable:
        per[it["type"]][1] += 1
    for it in answerable:
        if it not in misses:
            per[it["type"]][0] += 1
    for t, (hit, tot) in sorted(per.items()):
        print(f"  {t:<16} {hit:>3}/{tot:<4} {hit / tot:>6.0%}")

    human = [it for it in answerable if it.get("review") == "human"]
    print()
    if human:
        hh, hn, _ = score(human, r.search, args.top_k)
        print(f"  human-checked only  {hh}/{hn} = {hh / hn:.1%}  "
              f"({hn}/{len(answerable)} reviewed)")
    else:
        print("  All questions are review=machine: grounded in chunk text and")
        print("  verified verbatim, but not confirmed fair by a person.")

    if misses and (args.misses or len(misses) <= 15):
        print(f"\nmisses at top-{args.top_k} ({len(misses)}):")
        for it in misses:
            print(f"  [{it['id']:<10}] {it['question'][:70]}")


if __name__ == "__main__":
    main()
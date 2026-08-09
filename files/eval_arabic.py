"""
Does the Arabic path retrieve the same documents as the English path?

Retrieval hit only, deliberately. The generation score in the old harness was
English substring matching against `must_include`, so an Arabic answer would
score ~0 for reasons that have nothing to do with whether the pipeline works.
Retrieval is the honest measurable: the gold `doc` per question is
language-independent, so the Arabic number sits on exactly the same scale as
the English one and the two can be compared directly.

Three rows:

    English (baseline)    recomputed here, same process and same retriever, so
                          the comparison isn't against a remembered number
    Arabic -> English     the real pipeline: translate, then retrieve
    Arabic (raw, no MT)   straight into the retriever, untranslated

Measured on this corpus:

    English (baseline)    32/32 = 100.0%
    Arabic -> English     31/32 =  96.9%
    Arabic (raw, no MT)   18/32 =  56.2%

The third row is the control, and it is more interesting than expected. The
prediction going in was "near zero": BM25 tokenises `[a-z0-9]+`, so an Arabic
query yields no tokens at all, and PREFIX_HINTS is an English keyword table
that never fires. 56% is not near zero.

Two things explain it, and neither is the retriever working. With every BM25
score identical at zero, min-max normalisation gives that channel a constant
value, so ranking falls entirely to the dense channel - and MiniLM, despite
being an English model, retains enough cross-lingual signal to beat chance.
Chance here is not low either: top-3 of 9 documents is ~33% before any signal
at all. So the honest reading of 56% is "somewhat better than guessing", and
the 41-point gap to the translated row is what shows the translation step is
carrying the result. Without this row, "Arabic works" would be an assertion.

The single miss is worth reading rather than rounding away. PTN01-3 asks about
the *Commercial dept*; the Arabic says `قسم التجارة`, which comes back as
"trade department". That drops the word "commercial" - which is both a BM25
term and a PREFIX_HINTS routing keyword for PCM. It is a translation failure,
not a retrieval failure, and it is exactly the silent-miss mode the notebook
warns about: nothing in the output says a domain term was lost.

Needs Ollama up (translation runs through qwen3:14b) and eval_set_ar.json,
so run make_arabic_eval.py first.

    python make_arabic_eval.py
    python eval_arabic.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from make_arabic_eval import _gen
from retriever import Retriever
from translate import enable_utf8_stdout, is_arabic, to_english

HERE = Path(__file__).resolve().parent
EVAL_EN = HERE / "eval_set.json"
EVAL_AR = HERE / "eval_set_ar.json"
CHUNKS = HERE / "chunks.json"


def retrieval_hit(item: dict, question: str, search_fn, top_k: int = 3) -> bool | None:
    """None for the unanswerable questions - no gold doc, so not scoreable."""
    if item["doc"] == "none":
        return None
    docs = [x["filename"] for x in search_fn(question, top_k=top_k)]
    return item["doc"] in docs


def score(name: str, items: list[dict], questions: list[str], search_fn,
          top_k: int = 3, show_misses: bool = True) -> float:
    t0 = time.perf_counter()
    hits = [retrieval_hit(it, q, search_fn, top_k) for it, q in zip(items, questions)]
    dt = (time.perf_counter() - t0) / len(items) * 1000
    scored = [h for h in hits if h is not None]
    acc = sum(scored) / len(scored) if scored else 0.0
    print(f"{name:<24} {sum(scored):>2}/{len(scored)} = {acc:>6.1%}   {dt:>6.1f} ms/q")
    if show_misses:
        for it, q, h in zip(items, questions, hits):
            if h is False:
                print(f"      MISS [{it['id']}] {q[:66]}")
    return acc


def main() -> None:
    enable_utf8_stdout()
    if not EVAL_AR.exists():
        raise SystemExit(f"{EVAL_AR.name} not found - run `python make_arabic_eval.py` first.")

    eval_en = json.load(open(EVAL_EN, encoding="utf-8"))
    by_id = {it["id"]: it for it in json.load(open(EVAL_AR, encoding="utf-8"))}
    # Iterate the English set so ordering and the denominator match exactly.
    items = [it for it in eval_en if it["id"] in by_id]
    en_qs = [it["question"] for it in items]
    ar_qs = [by_id[it["id"]]["question"] for it in items]

    chunks = json.load(open(CHUNKS, encoding="utf-8"))
    r = Retriever(chunks)

    # Translate up front so the per-query timings below measure retrieval
    # rather than MT, and so the translations can be eyeballed at the end.
    print(f"translating {len(ar_qs)} questions through the same path ask() uses ...")
    t0 = time.perf_counter()
    translated = [to_english(q, _gen) if is_arabic(q) else q for q in ar_qs]
    print(f"  {time.perf_counter() - t0:.0f}s total\n")

    print("top-3 document accuracy")
    print("-" * 68)
    score("English (baseline)", items, en_qs, r.search)
    score("Arabic -> English", items, translated, r.search)
    score("Arabic (raw, no MT)", items, ar_qs, r.search, show_misses=False)
    print("-" * 68)

    human = [i for i, it in enumerate(items) if by_id[it["id"]].get("review") == "human"]
    if human:
        score("  human-checked only",
              [items[i] for i in human], [translated[i] for i in human],
              r.search, show_misses=False)
        print(f"  ({len(human)}/{len(items)} questions reviewed by a human)")
    else:
        print("All questions are review=machine: they were generated by the same")
        print("model that translates them back, so this is an UPPER BOUND, not a")
        print("user-facing accuracy claim. Naturalise them and set review=human.")

    print("\nsample translations (Arabic -> English):")
    for it, ar, en in list(zip(items, ar_qs, translated))[:5]:
        print(f"  [{it['id']}] {ar}\n        -> {en}")


if __name__ == "__main__":
    main()

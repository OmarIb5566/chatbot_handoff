"""
End-to-end generation accuracy on eval_set_v2.json. Needs Ollama.

    python eval_generation.py                 # all 100
    python eval_generation.py --limit 5       # time it before committing
    python eval_generation.py --out run.json  # keep every record for inspection

Slow by nature - one generation call per question - so it writes its output file
after every answer. A run of this length gets interrupted, and losing 80 answers
to a Ctrl-C at question 81 is avoidable.

WHAT IS SCORED, AND WHY EACH NUMBER IS SEPARATE
-----------------------------------------------
    retrieval   was the gold document among the retrieved chunks?
    answer      did the answer contain every `must_include` string?
    refusal     did an unanswerable question get refused?
    validator   did the answer cite a form number the corpus does not contain?

These are reported apart rather than folded into one accuracy, because a single
number cannot tell you which half broke - the point app.py makes by shipping the
retrieved chunks next to every answer. The interesting cell is the one this
prints as `answered but retrieval MISSED`: the model produced the expected
string without the gold document, which means either a second document also
supports the answer or the keyword match got lucky. Both make a headline
accuracy less trustworthy than it looks.

`answer` is keyword matching, exactly as the old harness did it, and it is a
weak proxy in both directions: it cannot see a right answer phrased differently,
and it scores "Use F-P-CM-01-01, or possibly F-P-CM-03-07" as correct. That
second failure is what `validator` exists to catch, and why the two columns are
never combined.

REFUSAL IS MATCHED ON THE PROMPT'S OWN STRING
---------------------------------------------
`chatbot.REFUSAL_EN` is what the prompt instructs the model to say, so the check
tests the contract the prompt actually sets rather than a list of hedging
phrases someone guessed at. A looser "sounds like a refusal" heuristic would
score paraphrases as passes, and a paraphrase is a real failure here: the app
and the notebook both key off this exact sentence.
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_EVAL = HERE / "eval_set_v2.json"


def scored(rec: dict, item: dict, refusal: str) -> dict:
    """One answer -> the four independent verdicts."""
    ans = (rec["answer"] or "").lower()
    docs = [h["filename"] for h in rec["hits"]]
    unanswerable = item["doc"] == "none"

    return {
        "id": item["id"],
        "type": item["type"],
        "doc": item["doc"],
        "question": item["question"],
        "answer": rec["answer"],
        "retrieved": docs,
        "retrieval_hit": None if unanswerable else item["doc"] in docs,
        "answer_hit": None if unanswerable else all(
            s.lower() in ans for s in item["must_include"]),
        "missing": None if unanswerable else [
            s for s in item["must_include"] if s.lower() not in ans],
        "refused": refusal.lower()[:28] in ans if unanswerable else None,
        "validator": rec["validator"]["verdict"],
        "unknown_forms": rec["validator"]["unknown"],
        "cited_forms": rec["validator"]["cited"],
        "vlm_sources": [v["filename"] for v in rec["vlm_sources"]],
        "generation_s": rec["generation_s"],
        "total_s": rec["total_s"],
    }


def report(rows: list[dict]) -> None:
    ans = [r for r in rows if r["answer_hit"] is not None]
    una = [r for r in rows if r["refused"] is not None]

    print("\n" + "=" * 62)
    print(f"GENERATION EVAL - {len(rows)} questions answered")
    print("=" * 62)

    if ans:
        ret = sum(r["retrieval_hit"] for r in ans)
        cor = sum(r["answer_hit"] for r in ans)
        print(f"  retrieval (gold doc retrieved)  {ret:>3}/{len(ans)} = {ret / len(ans):>6.1%}")
        print(f"  answer    (must_include found)  {cor:>3}/{len(ans)} = {cor / len(ans):>6.1%}")

        # The diagnostic cell: right string, wrong (or absent) source.
        lucky = [r for r in ans if r["answer_hit"] and not r["retrieval_hit"]]
        blocked = [r for r in ans if r["retrieval_hit"] and not r["answer_hit"]]
        print(f"\n  answered but retrieval MISSED   {len(lucky):>3}   "
              "(right string without the gold document)")
        print(f"  retrieved but answer MISSED     {len(blocked):>3}   "
              "(had the source, still did not say it)")

    if una:
        ref = sum(r["refused"] for r in una)
        print(f"\n  refusal   (unanswerable)        {ref:>3}/{len(una)} = {ref / len(una):>6.1%}")
        leaked = [r for r in una if r["cited_forms"]]
        if leaked:
            print(f"  !! cited a form on an unanswerable question: "
                  f"{[r['id'] for r in leaked]}")

    flagged = [r for r in rows if r["validator"] != "clean"]
    print(f"\n  validator flagged               {len(flagged):>3}/{len(rows)}")
    if flagged:
        unknown = sorted({f for r in flagged for f in r["unknown_forms"]})
        print(f"    unknown forms cited: {unknown}")
        for r in flagged[:8]:
            print(f"    [{r['id']}] {r['unknown_forms']}")

    if ans:
        print("\n  by type (answer accuracy)")
        per = collections.defaultdict(lambda: [0, 0])
        for r in ans:
            per[r["type"]][0] += r["answer_hit"]
            per[r["type"]][1] += 1
        for t, (h, n) in sorted(per.items()):
            print(f"    {t:<16} {h:>3}/{n:<4} {h / n:>6.0%}")

    vlm = [r for r in rows if r["vlm_sources"]]
    if vlm:
        print(f"\n  answers built partly on machine-read diagrams: {len(vlm)}")

    t = [r["total_s"] for r in rows]
    t.sort()
    print(f"\n  latency: median {t[len(t) // 2]:.1f}s, max {t[-1]:.1f}s, "
          f"total {sum(t) / 60:.0f} min")

    miss = [r for r in ans if not r["answer_hit"]]
    if miss:
        print(f"\n  answer misses ({len(miss)}):")
        for r in miss:
            tag = "ret-ok " if r["retrieval_hit"] else "ret-MISS"
            print(f"    [{r['id']:<10}] {tag} missing={r['missing']}")


def main() -> None:
    from translate import enable_utf8_stdout

    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="Generation accuracy, end to end")
    ap.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    ap.add_argument("--out", type=Path, default=HERE / "generation_eval_results.json")
    ap.add_argument("--limit", type=int, default=None, help="first N questions only")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    from chatbot import DEFAULT_MODEL, REFUSAL_EN, Chatbot, ollama_placement, ollama_up

    if not ollama_up():
        raise SystemExit("Ollama is not reachable - start it with `ollama serve`.")

    items = json.load(open(args.eval, encoding="utf-8"))
    if args.limit:
        items = items[:args.limit]

    bot = Chatbot(model=args.model or DEFAULT_MODEL)
    print(f"{len(bot.chunks)} chunks ({bot.n_workflow_chunks} machine-read), "
          f"{len(bot.form_whitelist)} known forms")
    # Latency is uninterpretable without this - see README_HANDOFF section 5.
    # Takes the HOST, not the model: /api/ps reports whatever is resident.
    print(f"placement: {ollama_placement(bot.host)}")
    print(f"running {len(items)} questions through {bot.model} ...\n")

    rows = []
    t0 = time.perf_counter()
    for i, it in enumerate(items, 1):
        rec = bot.ask(it["question"], top_k=args.top_k)
        row = scored(rec, it, REFUSAL_EN)
        rows.append(row)

        if row["answer_hit"] is None:
            mark = "REFUSED" if row["refused"] else "answered"
        else:
            mark = "hit " if row["answer_hit"] else "MISS"
        eta = (time.perf_counter() - t0) / i * (len(items) - i) / 60
        # flush: a run this long is normally watched through a redirected file
        # or a pipe, and Python block-buffers those - without this the progress
        # line only appears once the whole run has finished, which is when it is
        # no longer progress. The checkpoint file is still the source of truth.
        print(f"  [{i:>3}/{len(items)}] {row['id']:<11} {mark:<8} "
              f"{row['total_s']:>5.1f}s   eta {eta:>4.0f} min", flush=True)

        # Checkpoint every answer, not at the end.
        args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                            encoding="utf-8")

    report(rows)
    print(f"\nWrote {args.out.name}")


if __name__ == "__main__":
    main()

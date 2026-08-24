"""
Run eval_set_app.json through the APP's pipeline - Chatbot.ask(), not the
retriever on its own.

    python eval_app.py --retrieval-only        # ~10s, no generation
    python eval_app.py                         # English, ~20 generations
    python eval_app.py --lang both             # English + Arabic, ~45 min
    python eval_app.py --lang ar --limit 5

WHY THIS EXISTS ALONGSIDE eval_retrieval.py AND eval_arabic.py
---------------------------------------------------------------
Those two score retrieval, which is the right call for them: the gold `doc` is
language-independent, so English and Arabic sit on one scale. But four things
the app does are invisible to a retrieval score, and three of them were added
after those harnesses were written:

  * spelling repair - APP-17 retrieved the correct document and still answered
    "Not specified in these process documents", because repair ran inside
    search() and the generator was handed the raw typo. Retrieval was 100% for
    that question, and the answer was still wrong. A retrieval-only harness
    would have called it a pass.
  * follow-up rewriting - APP-19's second turn cannot be scored on its own; it
    only means anything after the turn before it.
  * continuation replay - APP-20 does no retrieval at all, so retrieval
    accuracy is undefined for it by construction.
  * refusal - APP-15/16 are correct when the answer is empty, which no
    substring check rewards.

So this harness carries a session, sends setup turns where a question has one,
and scores the ANSWER as well as the retrieval.

SCORING, AND WHAT IT REFUSES TO SCORE
--------------------------------------
`must_include` strings are split by whether they survive translation:

    codes and figures  F-P-CM-01-01, 160, 10%, 14 days   checked in BOTH languages
    prose              "Individual Development Plan"      English only

An Arabic answer fails English substring matching for reasons that have nothing
to do with the pipeline - the same argument eval_arabic.py makes for scoring
retrieval only. Rather than pretend otherwise, prose targets are reported as
`n/a` in Arabic and left out of the denominator.

APP-20 is `scoring: manual` and is always reported as such. Whether a
continuation "added what it missed" is not a substring question, and a harness
that scored it anyway would be lying about what it measured.
"""
from __future__ import annotations

import sys
from pathlib import Path

# backend/ holds the pipeline; its modules import each other by bare name.
# See paths.add_backend_to_path for why that is worth keeping.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))


import argparse
import json
import re
import time
from pathlib import Path

from chatbot import DEFAULT_MODEL, Chatbot
from translate import enable_utf8_stdout

from paths import EVAL_SET_APP as EVAL_APP, APP_EVAL_RESULTS

# A target that survives translation: a form/doc code, or anything with a digit
# in it. Everything else is prose and is English-only.
_CODE_RE = re.compile(r"^[A-Z]-[A-Z]-[A-Z]{2}-\d|^\d|\d")


def language_independent(target: str) -> bool:
    return bool(_CODE_RE.search(target))


# A form code survives translation whole; a figure survives only as its number.
# "5 days" is a correct Arabic answer as "5 أيام", so matching the English
# string would fail it for the one reason this harness exists not to.
_FULL_CODE_RE = re.compile(r"^[A-Za-z]-[A-Za-z]-[A-Za-z]{2}-[\d-]+$")
_NUMERIC_RE = re.compile(r"\d+(?:[.,]\d+)?%?")


def arabic_targets(target: str) -> list[str]:
    """What of an English target may legitimately be demanded of Arabic."""
    if _FULL_CODE_RE.match(target.strip()):
        return [target.strip()]
    return _NUMERIC_RE.findall(target)


def check(answer: str, targets: list[str], lang: str) -> tuple[int, int, list[str]]:
    """Returns (hit, scored, missed). Prose targets are skipped in Arabic, and
    figures are reduced to the figure itself."""
    hit = scored = 0
    missed = []
    for t in targets:
        wanted = [t]
        if lang == "ar":
            wanted = arabic_targets(t)
            if not wanted:
                continue      # prose-only target: not scoreable in Arabic
        scored += 1
        if all(w.lower() in answer.lower() for w in wanted):
            hit += 1
        else:
            missed.append(t if lang == "en" else f"{t} (as {'+'.join(wanted)})")
    return hit, scored, missed


def refused(answer: str) -> bool:
    """Did the model decline rather than invent? Matches both refusal strings
    from translate.py plus the shapes the model actually produces."""
    a = answer.lower()
    return any(s in a for s in (
        "not specified", "not covered", "does not", "no information",
        "isn't in the context", "not in the context", "لا يوجد", "غير محدد",
        "لم يتم", "غير مذكور",
    ))


def run_one(bot: Chatbot, item: dict, lang: str, top_k: int, use_typo: bool) -> dict:
    """One question, its own session so history never leaks between items."""
    sid = f"{item['id']}-{lang}-{'typo' if use_typo else 'clean'}"
    key = "question_ar" if lang == "ar" else "question_en"

    setup_key = "setup_ar" if lang == "ar" else "setup_en"
    if item.get(setup_key):
        bot.ask(item[setup_key], top_k=top_k, session_id=sid)

    q = item[key]
    if use_typo and lang == "en" and item.get("question_en_typo"):
        q = item["question_en_typo"]

    rec = bot.ask(q, top_k=top_k, session_id=sid)
    docs = [h["filename"] for h in rec["hits"]]
    ans = rec["answer"]

    out = {
        "id": item["id"], "lang": lang, "typo": use_typo, "question": q,
        "mode": rec.get("mode"), "repairs": rec.get("spelling_repairs") or [],
        "retrieved": docs, "answer": ans,
        "verdict": rec["validator"]["verdict"], "seconds": rec["total_s"],
    }

    if item["doc"] == "none":
        out["outcome"] = "ok" if refused(ans) else "INVENTED"
    elif item.get("scoring") == "manual":
        out["outcome"] = "manual"
        out["doc_hit"] = item["doc"] in docs
    else:
        # A continuation deliberately does not retrieve, so a doc check on it
        # would be measuring the previous turn.
        out["doc_hit"] = None if rec.get("reused_sources") else item["doc"] in docs
        hit, scored, missed = check(ans, item["must_include"], lang)
        out["missed"] = missed
        if scored == 0:
            out["outcome"] = "n/a"        # prose-only target in Arabic
        elif hit == scored:
            out["outcome"] = "ok"
        else:
            out["outcome"] = "WRONG"
    return out


def main(argv: list[str] | None = None) -> None:
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--lang", choices=("en", "ar", "both"), default="en")
    ap.add_argument("--top-k", type=int, default=5,
                    help="chunks retrieved (default 5; the app defaults to 3, "
                         "which drops the round-trip qualifier in APP-17)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--typo", action="store_true",
                    help="use question_en_typo where the set provides one")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="skip generation: score the gold document only")
    ap.add_argument("--out", type=Path, default=APP_EVAL_RESULTS)
    args = ap.parse_args(argv)

    data = json.load(open(EVAL_APP, encoding="utf-8"))
    items = data["questions"][:args.limit or None]
    langs = ("en", "ar") if args.lang == "both" else (args.lang,)

    if args.retrieval_only:
        # No model, no session: just ask whether the gold document is reachable.
        from retriever import Retriever

        from chatbot import load_corpus
        r = Retriever(load_corpus())
        print(f"retrieval only, top-{args.top_k}\n" + "-" * 72)
        tot = hit = 0
        for it in items:
            if it["doc"] == "none" or it.get("setup_en"):
                continue
            for lang in langs:
                q = it["question_ar"] if lang == "ar" else it["question_en"]
                if args.typo and lang == "en" and it.get("question_en_typo"):
                    q = it["question_en_typo"]
                docs = [h["filename"] for h in r.search(q, top_k=args.top_k)]
                ok = it["doc"] in docs
                tot += 1
                hit += ok
                print(f"  {'ok  ' if ok else 'MISS'} [{it['id']} {lang}] {q[:56]}")
        print("-" * 72)
        print(f"  {hit}/{tot} = {hit/tot:.1%}   (Arabic goes in raw here - the app "
              "translates first, so this row is a floor, not the app's number)")
        return

    bot = Chatbot(model=args.model)
    results, t0 = [], time.perf_counter()
    for it in items:
        for lang in langs:
            res = run_one(bot, it, lang, args.top_k, args.typo)
            results.append(res)
            mark = {"ok": "ok  ", "WRONG": "WRONG", "INVENTED": "INVENT",
                    "manual": "man ", "n/a": "n/a "}[res["outcome"]]
            dh = {True: "doc✓", False: "doc✗", None: "doc–"}[res.get("doc_hit")]
            rep = f" repaired={res['repairs']}" if res["repairs"] else ""
            print(f"  {mark} [{res['id']} {lang}] {dh} {res['mode']:<12} "
                  f"{res['seconds']:>5.1f}s{rep}")
            print(f"        {res['answer'][:96]}")
            if res.get("missed"):
                print(f"        missing: {res['missed']}")

    print("\n" + "=" * 72)
    for lang in langs:
        rows = [r for r in results if r["lang"] == lang]
        scored = [r for r in rows if r["outcome"] in ("ok", "WRONG", "INVENTED")]
        ok = sum(r["outcome"] == "ok" for r in scored)
        docs = [r for r in rows if r.get("doc_hit") is not None]
        dh = sum(r["doc_hit"] for r in docs)
        print(f"{lang}: answer {ok}/{len(scored)} = {ok/len(scored):.0%}   "
              f"gold doc retrieved {dh}/{len(docs)} = {dh/len(docs):.0%}   "
              f"({sum(r['outcome'] == 'n/a' for r in rows)} prose targets n/a, "
              f"{sum(r['outcome'] == 'manual' for r in rows)} manual)")
    print(f"total {time.perf_counter() - t0:.0f}s")

    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out.name}")
    manual = [r for r in results if r["outcome"] == "manual"]
    if manual:
        print("\nMANUAL CHECKS - a substring test cannot judge these:")
        for r in manual:
            print(f"  [{r['id']} {r['lang']}] mode={r['mode']}  {r['answer'][:120]}")


if __name__ == "__main__":
    main()

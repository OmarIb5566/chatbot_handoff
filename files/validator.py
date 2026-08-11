"""
Deterministic post-generation validator for form numbers and doc codes.

The scoring in the eval harness is keyword-match: it asks "did the expected
string appear?". That says nothing about what *else* the model said. A model
that answers "Use F-P-CM-01-01, or alternatively F-P-CM-03-07" scores as
correct while inventing a form number that does not exist. This module is
the independent check: pull every form-shaped string out of the answer and
test it against the set of form numbers that actually occur in the corpus.

It is deliberately dumb and deterministic. No model judges another model.

--- Two design decisions worth knowing about ---

1. The whitelist is harvested from ALL chunks, not just RELATED DOCUMENTED
   INFORMATION sections. RDI is where forms are *catalogued*, but scanning
   the 9-doc corpus finds 67 distinct forms of which 4 (F-P-HR-02-09,
   F-P-QD-08-04, F-P-QP-06-02, F-P-VMO-02-01) appear only in step bodies.
   An RDI-only whitelist would flag those four real forms as hallucinations.

2. FLAG, don't BLOCK - at least until the corpus is complete. The corpus
   cites forms belonging to process families that are not in it (FW, HR, OP,
   PU, QP prefixes). At 9 docs the whitelist is a subset of reality, so a
   "not in corpus" verdict means "unverifiable here", not "fabricated". Hard
   blocking now would suppress correct answers. Once all 500-600 docs are
   indexed the whitelist approaches complete and BLOCK becomes safe -
   `policy="block"` switches it.

--- Scale, and why the regexes were widened ---

At 71 process documents the whitelist holds 370 forms rather than 67. That is
most of the way to the condition point 2 sets for BLOCK, so the regexes now have
to be right: at 67 forms an over-narrow pattern was one missing entry among many
unverifiable ones, and at 370 it is a correct answer being called a
hallucination. Both the dash class and the F-M-* family below were found by
counting what the corpus contains against what these patterns could match, and
both were silent - a whitelist that is too small produces no error, only
confident false flags.
"""
from __future__ import annotations

import re

# Every dash the corpus actually uses. Two documents (P-PC-03 and the Quality
# Manual) write their form numbers with U+2010 HYPHEN rather than ASCII
# hyphen-minus - 18 references in total. Matching only ASCII kept those forms
# out of the whitelist entirely, so a model that cited "F-P-PC-03-02" correctly,
# with an ordinary hyphen, was flagged as having hallucinated it. A false
# hallucination flag is worse than no flag: it trains the reader to ignore them.
DASH = r"[-‐‑‒–—−]"

# Loose on purpose. A tight regex that only matches well-formed real codes
# would let a malformed hallucination ("F-P-CM-1-1", "F-P-ZZZ-99-99") through
# unnoticed, which is precisely the failure we are trying to detect.
#
# The second letter is [A-Z], not a literal P. It was a literal P while the
# corpus was 9 process documents, and every form in those is F-P-*. The full
# corpus has 60 references to F-M-* forms (the M-HR-* HR policies: resignation,
# promotion, local transfer, casual-labour hiring) plus one F-F-*, which is 37
# distinct real forms the whitelist could not see. Same consequence as the
# hyphen, at twice the scale.
FORM_RE = re.compile(
    rf"\bF\s*{DASH}\s*([A-Z])\s*{DASH}\s*([A-Z]{{1,5}})\s*{DASH}\s*(\d{{1,3}})\s*{DASH}\s*(\d{{1,3}})\b",
    re.I)

# Doc codes: P-CM-01, P-VMO-02, M-HR-08, ...
#
# Deliberately [PM] rather than [A-Z]. A form number is anchored by its leading
# "F-", so widening its second letter cannot make it match much else; a doc code
# has no such anchor, and LETTER-LETTERS-DIGITS is a common enough shape that
# [A-Z] would start reporting spurious unknown doc codes. P and M are the two
# families that exist (Process and Manual): 69 codes in the corpus, of which 20
# are M-HR-*.
DOC_CODE_RE = re.compile(rf"\b([PM])\s*{DASH}\s*([A-Z]{{1,5}})\s*{DASH}\s*(\d{{1,3}})\b", re.I)

_DASH_CHARS = "‐‑‒–—−"


def _norm(s: str) -> str:
    """Whitespace out, dashes folded to ASCII, upper-cased.

    Folding the dashes matters as much as matching them: without it a corpus
    form written with U+2010 and the same form cited with an ASCII hyphen
    normalise to two different strings, so the whitelist would contain the code
    and still not recognise it.
    """
    s = re.sub(r"\s+", "", s).upper()
    return s.translate({ord(d): "-" for d in _DASH_CHARS})


def build_form_whitelist(chunks: list[dict]) -> set[str]:
    """Every form number that actually appears anywhere in the corpus."""
    known = set()
    for c in chunks:
        for m in FORM_RE.finditer(c.get("text", "")):
            known.add(_norm(m.group(0)))
    return known


def build_doc_code_whitelist(chunks: list[dict]) -> set[str]:
    known = set()
    for c in chunks:
        if c.get("doc_code"):
            known.add(_norm(c["doc_code"]))
    return known


def extract_forms(text: str) -> list[str]:
    """All form-shaped strings in a model answer, normalised and deduped."""
    seen, out = set(), []
    for m in FORM_RE.finditer(text or ""):
        f = _norm(m.group(0))
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def context_forms(chunks: list[dict]) -> set[str]:
    """Form numbers occurring in the chunks one answer was actually built from.

    The whitelist answers "does this form exist?". This answers "was it in
    front of the model?", which is a different and stricter question - and the
    one the prompt actually sets, since it says to answer from the context only.
    """
    known = set()
    for c in chunks or []:
        for m in FORM_RE.finditer(c.get("text", "")):
            known.add(_norm(m.group(0)))
    return known


def validate_answer(
    answer: str,
    form_whitelist: set[str],
    doc_whitelist: set[str] | None = None,
    policy: str = "flag",
    retrieved: list[dict] | None = None,
) -> dict:
    """Check an answer's cited form numbers against the corpus and its context.

    Returns:
        cited          - form numbers the answer mentions
        unknown        - those not present in the corpus at all
        ungrounded     - those that exist, but not in the retrieved chunks
        ok             - True if every cited form is known (or none cited)
        verdict        - "clean" | "ungrounded" | "flagged" | "blocked"
        display_answer - the answer, or a refusal if policy == "block"

    WHY `ungrounded` EXISTS, AND WHAT IT STILL DOES NOT CATCH
    ---------------------------------------------------------
    The whitelist answers "does this form exist anywhere in the corpus", which
    is a weaker question than the prompt asks of the model - the prompt says to
    answer from the context only. A form cited that was not in the context is a
    contract violation regardless of whether it exists, so it is worth naming.

    Be clear about the limit, because the case that motivated this check is NOT
    caught by it. In the first generation run over eval_set_v2.json, asked which
    form is the Project Sourcing Strategy, the model answered

        "The form for the Project Sourcing Strategy is F-P-TN-02-04."

    The answer is F-P-TN-02-05. Both forms exist, so `unknown` was empty. Both
    were ALSO in the retrieved chunks, because the source lists them two lines
    apart:

        Tender sourcing strategy   F-P-TN-02-04
        Project sourcing strategy  F-P-TN-02-05

    so `ungrounded` is empty too, and the verdict is "clean". The model picked
    the adjacent, confusable entry out of material it was genuinely shown.

    That is the honest boundary: this check catches a citation pulled from
    OUTSIDE the context, and cannot catch the wrong one chosen from INSIDE it -
    which, on a corpus that names forms in near-identical pairs, is the more
    likely error. No form-number check can close that gap; it needs the answer
    compared against the specific line, which is a different kind of check.
    Treat a "clean" verdict as "cited nothing impossible", never as "correct".

    Reported separately from `unknown` rather than merged, because the two mean
    different things to a reader: `unknown` says the corpus has no such form,
    `ungrounded` says the answer went outside what it was shown.
    """
    cited = extract_forms(answer)
    unknown = [f for f in cited if f not in form_whitelist]

    # Only meaningful when the caller passed the chunks. `None` means "not
    # checked" and must not read as "all grounded", so the default is no check
    # rather than an empty set.
    ungrounded = []
    if retrieved is not None:
        seen = context_forms(retrieved)
        ungrounded = [f for f in cited if f not in unknown and f not in seen]

    unknown_docs = []
    if doc_whitelist is not None:
        for m in DOC_CODE_RE.finditer(answer or ""):
            code = _norm(m.group(0))
            # Skip doc codes that are really the prefix of a cited form number.
            if any(code in f for f in cited):
                continue
            if code not in doc_whitelist:
                unknown_docs.append(code)

    # `ok` stays keyed to `unknown` alone, so the existing meaning - and every
    # caller that reads it - is unchanged. An ungrounded citation is surfaced
    # through `verdict`, and never blocks: the form is real, and the model may
    # have had a good reason the retrieval window did not show.
    ok = not unknown
    if unknown and policy == "block":
        verdict = "blocked"
        display = ("Answer withheld: it cited form number(s) "
                   f"{', '.join(unknown)} that do not appear in the process "
                   "corpus. Please check the source document directly.")
    elif unknown:
        verdict = "flagged"
        display = answer
    elif ungrounded:
        verdict = "ungrounded"
        display = answer
    else:
        verdict = "clean"
        display = answer

    return {
        "cited": cited,
        "unknown": unknown,
        "ungrounded": ungrounded,
        "unknown_doc_codes": sorted(set(unknown_docs)),
        "ok": ok,
        "verdict": verdict,
        "display_answer": display,
    }


def validator_report(results: list[dict], form_whitelist: set[str]) -> dict:
    """Aggregate hallucination-proxy stats over a run_eval() result list.

    `caught_rate` is the headline: the share of answers that cited a form
    number the corpus does not contain. It is independent of the keyword
    scoring, so a model can be "accurate" and still score badly here.
    """
    n = len(results)
    checked = [validate_answer(r.get("got", ""), form_whitelist) for r in results]
    caught = [c for c in checked if not c["ok"]]
    cited_any = [c for c in checked if c["cited"]]

    # Answers to unanswerable questions that nonetheless cite a form number
    # are the worst case: the model invented a citation out of nothing.
    fabricated_on_unanswerable = [
        r["id"] for r, c in zip(results, checked)
        if r.get("type") == "unanswerable" and c["cited"]
    ]

    return {
        "n": n,
        "answers_citing_a_form": len(cited_any),
        "answers_with_unknown_form": len(caught),
        "caught_rate": len(caught) / n if n else 0.0,
        "unknown_forms": sorted({f for c in caught for f in c["unknown"]}),
        "fabricated_on_unanswerable": fabricated_on_unanswerable,
        "per_item": checked,
    }


if __name__ == "__main__":
    import json
    from pathlib import Path

    # Read the pipeline artifact rather than re-chunking: chunks.json is now
    # produced by adaptive_chunker.py, which reads the PDFs directly.
    chunks = json.load(open(Path(__file__).resolve().parent / "chunks.json",
                            encoding="utf-8"))
    wl = build_form_whitelist(chunks)
    dwl = build_doc_code_whitelist(chunks)
    print(f"{len(wl)} known form numbers, {len(dwl)} known doc codes")

    for probe in [
        "The Customer Satisfaction Survey uses form F-P-CM-01-01.",
        "Use form F-P-CM-03-07 to request a company car.",   # invented
        "Submit F-P-VMO-01-01 and also F-P-ZZ-99-99 to procurement.",
        "Not specified in these process documents.",
    ]:
        v = validate_answer(probe, wl, dwl)
        print(f"  {v['verdict']:<8} cited={v['cited']} unknown={v['unknown']}  <- {probe[:55]}")

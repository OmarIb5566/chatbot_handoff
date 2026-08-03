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
"""
from __future__ import annotations

import re

# Loose on purpose. A tight regex that only matches well-formed real codes
# would let a malformed hallucination ("F-P-CM-1-1", "F-P-ZZZ-99-99") through
# unnoticed, which is precisely the failure we are trying to detect.
FORM_RE = re.compile(r"\bF\s*-\s*P\s*-\s*([A-Z]{1,5})\s*-\s*(\d{1,3})\s*-\s*(\d{1,3})\b", re.I)

# Doc codes: P-CM-01, P-VMO-02, ...
DOC_CODE_RE = re.compile(r"\bP\s*-\s*([A-Z]{1,5})\s*-\s*(\d{1,3})\b", re.I)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s).upper()


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


def validate_answer(
    answer: str,
    form_whitelist: set[str],
    doc_whitelist: set[str] | None = None,
    policy: str = "flag",
) -> dict:
    """Check an answer's cited form numbers against the corpus.

    Returns:
        cited          - form numbers the answer mentions
        unknown        - those not present in the corpus
        ok             - True if every cited form is known (or none cited)
        verdict        - "clean" | "flagged" | "blocked"
        display_answer - the answer, or a refusal if policy == "block"
    """
    cited = extract_forms(answer)
    unknown = [f for f in cited if f not in form_whitelist]

    unknown_docs = []
    if doc_whitelist is not None:
        for m in DOC_CODE_RE.finditer(answer or ""):
            code = _norm(m.group(0))
            # Skip doc codes that are really the prefix of a cited form number.
            if any(code in f for f in cited):
                continue
            if code not in doc_whitelist:
                unknown_docs.append(code)

    ok = not unknown
    if ok:
        verdict = "clean"
        display = answer
    elif policy == "block":
        verdict = "blocked"
        display = ("Answer withheld: it cited form number(s) "
                   f"{', '.join(unknown)} that do not appear in the process "
                   "corpus. Please check the source document directly.")
    else:
        verdict = "flagged"
        display = answer

    return {
        "cited": cited,
        "unknown": unknown,
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
    from chunker import chunk_all

    chunks = chunk_all()
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

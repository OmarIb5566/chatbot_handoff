"""RME's own vocabulary, pinned so translation can't paraphrase it away.

`translate.to_english()` sends a question through a generic MT prompt before
it reaches the retriever, and generic MT does not know that "Commercial" is
this corpus's department name rather than a synonym for "trade" - see
`قسم التجارة` -> "trade department" (README_HANDOFF.md, PTN01-3), or "PM"
rendered as "Prime Minister" (make_arabic_eval.py). Both are honest
translations of the Arabic; both are wrong for this domain, because BM25 and
`retriever.PREFIX_HINTS` key on RME's exact words, not their synonyms.

The fix mirrors `translate.mask_codes`/`unmask_codes`: a term the model never
sees cannot be mistranslated. This only works because the vocabulary is
closed and small - RME's own department and role names, extracted from the
corpus itself (data/chunks.json, data/workflow_chunks_fixed.json), not
open-ended text. Two tiers:

  * GLOSSARY - unambiguous in the corpus (checked against every document that
    mentions each term): hard-masked, like a form code.
  * AMBIGUOUS_TERMS - abbreviations the corpus itself uses for more than one
    role (e.g. "TM" is Technical Affairs Manager in one document and
    Technical Manager in another). Pinning these to one meaning would trade
    one mistranslation for another, so they only get a soft prompt hint:
    "don't translate it, leave it in Latin script."
"""
from __future__ import annotations

import re

GLOSSARY: dict[str, list[str]] = {
    # canonical English term (exact corpus spelling) -> known Arabic surface
    # forms (formal renderings plus the specific bad-translation phrasings
    # already observed in eval data). Order doesn't matter: mask_glossary
    # sorts by length before matching.
    "Commercial Dept.": [
        "قسم التجارة", "الإدارة التجارية", "القسم التجاري",
        "قسم العملاء والتجارة", "إدارة العملاء والتجارة",
    ],
    "Formwork": ["الشدة الخشبية", "قسم الشدات", "الشدات", "أعمال الشدات"],
    "PM": ["مدير المشروع"],
    "PsM": ["مدير المشاريع"],
    "HoD": ["رئيس القسم"],
    "CM": ["مدير الإنشاءات", "مدير البناء"],
    "TOM": ["مدير المكتب الفني"],
    "VMO": ["مكتب إدارة الموردين", "إدارة الموردين"],
    "HR": ["الموارد البشرية"],
    "Procurement": ["المشتريات", "إدارة المشتريات"],
    "Purchasing": ["الشراء", "قسم المشتريات"],
    "Tendering": ["المناقصات"],
    "Subcontracts": ["العقود من الباطن", "المقاولين من الباطن"],
    "Self-Execution": ["التنفيذ الذاتي"],
    "Quality": ["الجودة"],
    "Finance": ["المالية", "قسم المالية"],
    "Equipment": ["المعدات", "قسم المعدات"],
    "Logistics": ["اللوجستيات", "الخدمات اللوجستية"],
    "Engineering": ["الهندسة", "القسم الهندسي"],
    "Planning & Cost Control": ["التخطيط ومراقبة التكاليف"],
    "COO": ["الرئيس التنفيذي للعمليات"],
    "CEO": ["الرئيس التنفيذي"],
    "VP": ["نائب الرئيس"],
}

# Abbreviations the corpus itself uses for more than one role (found by
# grepping every "TERMS AND DEFINITION(S)" section across all 61 PDFs, not
# guessed). Hard-masking these to a single canonical term would silently pick
# the wrong sense for whichever document the other sense belongs to, so they
# are surfaced as a "don't translate, leave verbatim" hint instead - the same
# trade-off _KEEP_VERBATIM makes for identifiers the mask regex cannot cover.
AMBIGUOUS_TERMS = {
    "TM": "Technical Affairs Manager in some processes, Technical Manager in others",
    "MS": "Method Statement in some processes, Material Submittal in others",
    "IIR": "Internal Inspection Request in process docs, Infrastructure Inspection Request in workflow diagrams",
}

_PLACEHOLDER = "GZ{}GZ"
_PLACEHOLDER_RE = re.compile(r"GZ\s*(\d+)\s*GZ", re.I)

# Longest surface form first, so "قسم التجارة" doesn't get pre-empted by a
# shorter substring of itself from another entry.
_SURFACE_FORMS: list[tuple[str, str]] = sorted(
    ((term, canonical) for canonical, terms in GLOSSARY.items() for term in terms),
    key=lambda pair: len(pair[0]),
    reverse=True,
)
_GLOSSARY_RE = re.compile(
    "|".join(re.escape(term) for term, _ in _SURFACE_FORMS)
) if _SURFACE_FORMS else None


def mask_glossary(text: str) -> tuple[str, list[str]]:
    """Replace known Arabic surface forms with placeholders.

    Returns (masked_text, canonical_terms) - canonical_terms holds the
    ENGLISH term to restore, not the Arabic that was matched, so the
    translation model never gets a chance to render it as anything else.
    """
    if _GLOSSARY_RE is None or not text:
        return text, []

    canon_by_surface = dict(_SURFACE_FORMS)
    terms: list[str] = []

    def _sub(m: re.Match) -> str:
        terms.append(canon_by_surface[m.group(0)])
        return _PLACEHOLDER.format(len(terms) - 1)

    return _GLOSSARY_RE.sub(_sub, text), terms


def unmask_glossary(text: str, terms: list[str]) -> str:
    """Restore placeholders as their canonical English term.

    Same "append what the model dropped" fallback as unmask_codes: losing the
    term's position is a formatting wrinkle, losing the term itself is a
    retrieval miss.
    """
    seen: set[int] = set()

    def _sub(m: re.Match) -> str:
        i = int(m.group(1))
        if i >= len(terms):
            return m.group(0)
        seen.add(i)
        return terms[i]

    out = _PLACEHOLDER_RE.sub(_sub, text)
    missing = [t for i, t in enumerate(terms) if i not in seen]
    if missing:
        out = f"{out.rstrip()} ({' '.join(missing)})"
    return out


# For the EN->AR leg (to_arabic): the first Arabic surface form listed for
# each canonical term is the preferred/formal rendering, used as the pinned
# translation rather than whatever qwen3 would produce live - see
# make_arabic_eval.py's documented "PM" -> "Prime Minister" defect, which
# happens on this leg when building the Arabic eval set.
_PREFERRED_ARABIC = {canonical: terms[0] for canonical, terms in GLOSSARY.items()}

# Longest canonical term first, and boundary-guarded so "PM" doesn't fire
# inside "PMS" or similar. Plain \b breaks on a term like "Commercial Dept."
# that ends in punctuation: \b requires a word/non-word transition, and
# "." followed by a space is non-word-to-non-word, so \b never matches there.
# (?<!\w)/(?!\w) check the character itself rather than a transition, so a
# term ending in punctuation still gets a boundary at the space that follows.
_EN_TERM_RE = re.compile(
    "|".join(
        r"(?<!\w)" + re.escape(term) + r"(?!\w)"
        for term in sorted(GLOSSARY, key=len, reverse=True)
    )
) if GLOSSARY else None


def mask_english_terms(text: str) -> tuple[str, list[str]]:
    """Replace canonical English glossary terms with placeholders.

    The EN->AR counterpart of mask_glossary: canonical_terms holds the
    ARABIC rendering to restore, so the model never gets a chance to
    translate "PM" into anything other than the term RME actually uses.
    """
    if _EN_TERM_RE is None or not text:
        return text, []

    terms: list[str] = []

    def _sub(m: re.Match) -> str:
        terms.append(_PREFERRED_ARABIC[m.group(0)])
        return _PLACEHOLDER.format(len(terms) - 1)

    return _EN_TERM_RE.sub(_sub, text), terms


def glossary_hint_block() -> str:
    """Prompt text covering vocabulary the surface-form regex won't catch.

    Two jobs: tell the model to prefer RME's exact terms for anything it
    still has to translate live (dialectal phrasing not in GLOSSARY's known
    surface forms), and tell it to leave the ambiguous abbreviations alone
    entirely rather than guessing a sense.
    """
    canonical_terms = ", ".join(sorted(GLOSSARY.keys()))
    ambiguous = ", ".join(sorted(AMBIGUOUS_TERMS))
    return (
        f"If the text refers to any of these RME department or role names, use "
        f"exactly this English term, not a synonym: {canonical_terms}.\n"
        f"These abbreviations mean different things in different RME documents "
        f"({ambiguous}) - if you see one, leave it in Latin script rather than "
        f"guessing which sense is meant."
    )

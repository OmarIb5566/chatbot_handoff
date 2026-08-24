"""
Arabic support for the process chatbot: translate at the edges, not in the index.

The corpus is English. So are the chunk embeddings, the BM25 tokens, the
routing keyword table in retriever.py, and the form-number regex in validator.py.
Making any of that multilingual is a rebuild of the whole retrieval stack for
nine documents. Instead an Arabic question is translated to English before it
is embedded, the existing pipeline runs untouched, and the English answer is
translated back to Arabic before it is printed.

Translation is done by the same local model that answers the question
(qwen3:14b over Ollama), so this adds no dependency and nothing leaves the
machine. `to_english`/`to_arabic` take a `generate_fn(prompt) -> str` callable
rather than talking to Ollama themselves, which keeps this module free of HTTP
and lets the caller pass a generator that already strips reasoning traces.

Two things the prompts fight for, because both are silent failures:

  * Form numbers, doc codes and PDF filenames must survive verbatim in Latin
    script. A model that helpfully renders 'F-P-CM-01-01' in Arabic-Indic
    digits produces a citation nobody can look up.
  * No preamble. "Sure, here is the translation:" becomes part of the answer
    otherwise, and on the AR->EN leg it becomes part of the embedded query.

The first of those is no longer left to the prompt alone. Codes are masked out
before the text reaches the model and substituted back afterwards
(`mask_codes`/`unmask_codes`), so a code cannot be corrupted by a translation
that never saw it. The `_KEEP_VERBATIM` instruction stays as a second line of
defence for identifiers the regex does not cover, like PDF filenames.

Belt and braces is warranted here specifically: `validator.validate_answer`
decides "hallucinated citation" by regex, so one mangled digit turns a correct
answer into a false hallucination flag, and on the AR->EN leg it silently
breaks doc-code routing in retriever.route_prefixes.
"""
from __future__ import annotations

import re

# Arabic, Arabic Supplement/Extended, and the presentation forms blocks.
AR_RE = re.compile(
    r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]"
)

# Preambles the model prepends despite being told not to.
_PREAMBLE_RE = re.compile(
    r"^\s*(?:the\s+)?(?:translation|translated\s+text|english|arabic|"
    r"الترجمة|الإنجليزية|"
    r"العربية)\s*(?:is)?\s*[:\-–]\s*",
    re.I,
)


def is_arabic(text: str, threshold: float = 0.2) -> bool:
    """True if the text is an Arabic query.

    The denominator is alphabetic characters only, not the whole string. A
    query like 'ما هو نموذج F-P-CM-01-01' carries Latin letters and digits from
    the form code it is asking about, and counting those would push a genuinely
    Arabic question under the threshold. Pure English scores exactly 0, so the
    exact threshold matters far less than the denominator does.
    """
    letters = [ch for ch in (text or "") if ch.isalpha()]
    if not letters:
        return False
    arabic = sum(1 for ch in letters if AR_RE.match(ch))
    return arabic / len(letters) >= threshold


# --- code masking ------------------------------------------------------------
# Form numbers ('F-P-CN-01-11') and doc codes ('P-VMO-01'). Matches the shapes
# validator.FORM_RE and validator.DOC_CODE_RE recognise, minus the whitespace
# tolerance - a code that already has spaces in it is not one we can safely
# reinsert verbatim.
MASK_RE = re.compile(r"\bF-P-[A-Z]{1,5}-\d{1,3}-\d{1,3}\b|\bP-[A-Z]{1,5}-\d{1,3}\b", re.I)

# Short, pure-ASCII, no internal punctuation: models copy this through far more
# reliably than they copy the codes themselves. Anything still lost is appended
# rather than dropped (see unmask_codes).
_PLACEHOLDER = "ZQ{}ZQ"
_PLACEHOLDER_RE = re.compile(r"ZQ\s*(\d+)\s*ZQ", re.I)


def mask_codes(text: str) -> tuple[str, list[str]]:
    """Replace form/doc codes with placeholders. Returns (masked, codes)."""
    codes: list[str] = []

    def _sub(m: re.Match) -> str:
        codes.append(m.group(0))
        return _PLACEHOLDER.format(len(codes) - 1)

    return MASK_RE.sub(_sub, text or ""), codes


def unmask_codes(text: str, codes: list[str]) -> str:
    """Restore codes in place. Codes the model dropped are appended instead.

    Appending rather than giving up is deliberate. On the AR->EN leg the result
    is fed to BM25 (a bag of words) and to route_prefixes (which scans the whole
    string), so a code recovered at the wrong position still does its full job.
    On the EN->AR leg a trailing code is ugly but auditable, which beats a
    silently missing citation.
    """
    seen: set[int] = set()

    def _sub(m: re.Match) -> str:
        i = int(m.group(1))
        if i >= len(codes):
            return m.group(0)
        seen.add(i)
        return codes[i]

    out = _PLACEHOLDER_RE.sub(_sub, text)
    missing = [c for i, c in enumerate(codes) if i not in seen]
    if missing:
        out = f"{out.rstrip()} ({' '.join(missing)})"
    return out


def _clean(out: str) -> str:
    """Strip the wrapping a model adds around a translation it was asked for bare."""
    s = (out or "").strip()
    s = _PREAMBLE_RE.sub("", s).strip()
    # Whole-string quoting: "..." or «...» around the entire translation.
    if len(s) >= 2 and s[0] in "\"'«“" and s[-1] in "\"'»”":
        s = s[1:-1].strip()
    return s


_KEEP_VERBATIM = (
    "Keep every form number (like F-P-CM-01-01), document code (like P-VMO-02), "
    "PDF filename, and Latin-script identifier EXACTLY as it appears - same "
    "characters, same digits, do not transliterate or convert the digits."
)


# Answers go out in MSA. Users write Egyptian dialect, and the INPUT side
# handles that - but formal Arabic is the right register for a reply quoting
# ISO process documents, and it is the one qwen3 produces reliably.
MSA = "Modern Standard Arabic"

# Only used for generating the Arabic eval set, where the questions stand in
# for what a user would type. qwen3 largely ignores it and returns MSA anyway;
# see make_arabic_eval.py.
EGYPTIAN = "Egyptian Arabic (اللهجة المصرية)"


def to_english(text: str, generate_fn) -> str:
    """Arabic -> English, for the query on its way into the retriever."""
    masked, codes = mask_codes(text)
    prompt = (
        "Translate the following Arabic text into English. The text may be in "
        "Egyptian dialect rather than Modern Standard Arabic.\n"
        f"{_KEEP_VERBATIM}\n"
        "Leave any ZQ<number>ZQ marker exactly as it is - it is a placeholder.\n"
        "Output ONLY the English translation. No preamble, no explanation, no quotes.\n\n"
        f"Arabic text:\n{masked}\n\nEnglish translation:"
    )
    return unmask_codes(_clean(generate_fn(prompt)), codes)


def to_arabic(text: str, generate_fn, dialect: str = MSA) -> str:
    """English -> Arabic.

    No longer on the main answer path - the generator is asked for Arabic
    directly instead (see the notebook's `ask`), which removes a whole
    round trip and with it a chance to corrupt a citation. This remains for
    the two places a genuine translation is still needed: the validator's
    English refusal text under `policy="block"`, and building the Arabic
    eval set in make_arabic_eval.py.
    """
    masked, codes = mask_codes(text)
    prompt = (
        f"Translate the following English text into {dialect}.\n"
        f"{_KEEP_VERBATIM}\n"
        "Leave any ZQ<number>ZQ marker exactly as it is - it is a placeholder.\n"
        "Output ONLY the Arabic translation. No preamble, no explanation, no quotes.\n\n"
        f"English text:\n{masked}\n\nArabic translation:"
    )
    return unmask_codes(_clean(generate_fn(prompt)), codes)


# Appended to the answer prompt when the question came in Arabic, so the
# generator writes Arabic straight from the English context instead of writing
# English that then has to be translated back.
ANSWER_IN_ARABIC = (
    "\n\nIMPORTANT: The context above is in English, but the user asked in "
    f"Arabic. Write your ANSWER in {MSA} (العربية الفصحى), not English. The "
    "question may be in Egyptian dialect; answer in formal Arabic regardless. "
    "Keep document filenames, document codes (like P-VMO-02) and form numbers "
    "(like F-P-CM-01-01) EXACTLY as they appear in the context: Latin script, "
    "Western digits, never transliterated."
)

# The refusal in build_prompt, in Arabic. Pinned as a constant rather than
# translated at runtime so the "no answer" path stays deterministic and free.
REFUSAL_AR = "غير محدد في وثائق العمليات هذه."


def enable_utf8_stdout() -> None:
    """Make printing Arabic work on a Windows console.

    Python defaults stdout to the ANSI codepage there (cp1252), which cannot
    encode Arabic at all - so a script dies with UnicodeEncodeError before it
    prints anything useful. Jupyter is already UTF-8; this is for the CLIs.
    """
    import sys

    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


if __name__ == "__main__":
    # Smoke test against a live Ollama, so this path is checkable without Jupyter.
    import sys

    import requests

    enable_utf8_stdout()

    OLLAMA_HOST = "http://localhost:11434"
    MODEL = "qwen3:14b"
    THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)

    def gen(prompt: str) -> str:
        payload = {
            "model": MODEL, "prompt": prompt, "stream": False,
            # Matches the notebook's generate(): no output cap, explicit context
            # window. Arabic is token-denser than English, so a capped budget
            # clips a translation before it clips the answer it came from.
            "options": {"temperature": 0, "num_predict": -1, "num_ctx": 8192, "seed": 0},
            "think": False,
        }
        r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=1800)
        if r.status_code == 400:          # older Ollama, or the model rejects `think`
            payload.pop("think")
            r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=1800)
        r.raise_for_status()
        return THINK_RE.sub("", r.json()["response"]).strip()

    print("--- detection ---")
    for probe in [
        "What form is used for the Customer Satisfaction Survey?",
        "ما هو النموذج المستخدم لاستبيان رضا العملاء؟",
        "ما هو نموذج F-P-CM-01-01؟",   # mixed script: still Arabic
        "P-VMO-02",                                                                     # no letters at all
    ]:
        print(f"  is_arabic={str(is_arabic(probe)):<5} {probe}")

    print("\n--- round trip (needs ollama) ---")
    ar = ("ما هو النموذج "
          "المستخدم لاستبيان "
          "رضا العملاء؟")
    en = to_english(ar, gen)
    print(f"  AR in   : {ar}")
    print(f"  -> EN   : {en}")

    src = ("The Customer Satisfaction Survey uses form F-P-CM-01-01, per "
           "P-CM-01 Customer Satisfaction Process.pdf.")
    back = to_arabic(src, gen)
    print(f"\n  EN in   : {src}")
    print(f"  -> AR   : {back}")
    print(f"  form number survived verbatim: {'F-P-CM-01-01' in back}")
    print(f"  doc code survived verbatim   : {'P-CM-01' in back}")

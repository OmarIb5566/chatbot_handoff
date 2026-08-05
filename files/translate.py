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


def to_english(text: str, generate_fn) -> str:
    """Arabic -> English, for the query on its way into the retriever."""
    prompt = (
        "Translate the following Arabic text into English.\n"
        f"{_KEEP_VERBATIM}\n"
        "Output ONLY the English translation. No preamble, no explanation, no quotes.\n\n"
        f"Arabic text:\n{text}\n\nEnglish translation:"
    )
    return _clean(generate_fn(prompt))


def to_arabic(text: str, generate_fn) -> str:
    """English -> Arabic, for the answer on its way to the user."""
    prompt = (
        "Translate the following English text into Modern Standard Arabic.\n"
        f"{_KEEP_VERBATIM}\n"
        "Output ONLY the Arabic translation. No preamble, no explanation, no quotes.\n\n"
        f"English text:\n{text}\n\nArabic translation:"
    )
    return _clean(generate_fn(prompt))


if __name__ == "__main__":
    # Smoke test against a live Ollama, so this path is checkable without Jupyter.
    import sys

    import requests

    # A Windows console defaults to cp1252 and cannot encode Arabic at all.
    # Jupyter is UTF-8 already; this is only for `python translate.py`.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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

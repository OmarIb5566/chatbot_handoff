"""Every tunable in one place, overridable by environment variable.

WHY THIS EXISTS
---------------
The host, the two model names, the retrieval width, the context budget and
four different HTTP timeouts were literals spread across chatbot.py,
contextualize.py and app.py - `OLLAMA_HOST` defined identically in two
modules, `qwen3:14b` written in three. That is fine on the machine it was
written on and nowhere else: moving Ollama to another host, or running the
eval harness against a second model, meant editing source in several files and
hoping none was missed.

Every value here answers to an environment variable, so the same checkout runs
against a different host or model without a diff:

    RME_OLLAMA_HOST=http://gpu-box:11434 streamlit run app.py
    RME_MODEL=qwen3.6:27b python evals/eval_generation.py

The DEFAULTS ARE THE TUNED ONES. They are not arbitrary and several are load
bearing - `TOP_K = 6` and `CONTEXT_BUDGET_CHARS = 18000` were measured against
the eval sets, and `NUM_CTX = 8192` exists because Ollama silently truncates
the prompt at its 4096 default, dropping the instruction at the front rather
than erroring. Changing one via the environment is a deliberate act; changing
one here changes what everybody gets.

Reading the environment happens once, at import. A process therefore has one
consistent configuration for its lifetime, and `settings_summary()` can be
logged at startup to record which one it was.
"""
from __future__ import annotations

import os

BAD_ENV: list[tuple[str, str, int]] = []


def _str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v.strip() if v and v.strip() else default


def _int(name: str, default: int) -> int:
    """Bad value -> the default, not a crash.

    A typo in an environment variable should not take the app down at import,
    before any logging exists to say why. The invalid value is remembered in
    BAD_ENV so startup can report it instead of silently ignoring it.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        BAD_ENV.append((name, raw, default))
        return default


# --- where the model server is --------------------------------------------
OLLAMA_HOST = _str("RME_OLLAMA_HOST", "http://localhost:11434")

# --- which models ----------------------------------------------------------
# Answer generation and translation. 14B is the smallest size tested that
# holds the workflow answer format and does not substitute a subject when
# rewriting - see contextualize.py.
MODEL = _str("RME_MODEL", "qwen3:14b")
# Follow-up rewriting. Split from MODEL so a small model can be tried on the
# short structured task without touching answer quality; currently the same.
REWRITE_MODEL = _str("RME_REWRITE_MODEL", MODEL)
# Sentence-transformers encoder for the dense channel. Changing this
# invalidates the embedding cache by fingerprint, which is the intended
# behaviour - a stale embedding is worse than a slow first load.
EMBED_MODEL = _str("RME_EMBED_MODEL", "all-MiniLM-L6-v2")

# --- retrieval and prompt shape (tuned; see evals/) ------------------------
TOP_K = _int("RME_TOP_K", 6)
CONTEXT_BUDGET_CHARS = _int("RME_CONTEXT_BUDGET_CHARS", 18_000)
NUM_CTX = _int("RME_NUM_CTX", 8192)

# --- HTTP timeouts, in seconds --------------------------------------------
# Generation is minutes, not seconds, on CPU: a long workflow answer at
# num_predict=-1 legitimately runs past ten minutes, and a timeout shorter
# than the work turns a slow answer into a failed one. The probes are short
# because they are liveness checks - a slow /api/tags means "treat as down".
GEN_TIMEOUT = _int("RME_GEN_TIMEOUT", 1800)
PROBE_TIMEOUT = _int("RME_PROBE_TIMEOUT", 5)
TAGS_TIMEOUT = _int("RME_TAGS_TIMEOUT", 10)

# --- validator -------------------------------------------------------------
# "flag" warns on an unverifiable form number; "block" withholds the answer.
# Default flag: the whitelist is incomplete at 71 documents, so "unknown"
# means unverifiable rather than fabricated, and blocking on it withholds
# correct answers.
POLICY = _str("RME_POLICY", "flag")

# --- logging ---------------------------------------------------------------
LOG_LEVEL = _str("RME_LOG_LEVEL", "INFO").upper()
LOG_FILE = _str("RME_LOG_FILE", "")          # empty: stderr only
# Questions and answers are user content. Operational logs record shape and
# timing only unless this is switched on deliberately - a log of what everyone
# asked is a different artifact with different handling requirements.
LOG_CONTENT = _str("RME_LOG_CONTENT", "0") not in ("0", "", "false", "False", "no")


def settings_summary() -> dict:
    """What this process is actually running, for the startup log line."""
    return {"host": OLLAMA_HOST, "model": MODEL, "rewrite_model": REWRITE_MODEL,
            "embed_model": EMBED_MODEL, "top_k": TOP_K,
            "context_budget": CONTEXT_BUDGET_CHARS, "policy": POLICY,
            "gen_timeout_s": GEN_TIMEOUT, "log_content": LOG_CONTENT}

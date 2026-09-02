"""One logging setup, called once by whichever process is the entry point.

WHY THIS EXISTS
---------------
There was no logging. The pipeline's diagnostics went into the answer record
and were rendered in the developer view, which means they existed only while
someone had the browser open and vanished with the Streamlit session. When a
colleague reported a bad answer an hour later there was nothing on disk to
look at: not the model, not the latency, not whether the validator flagged it,
not whether a version collision was in play.

`setup()` is deliberately idempotent and only ever touches the "rme" logger,
never the root: Streamlit, urllib3 and sentence-transformers all configure
logging themselves, and reconfiguring root either duplicates their output or
loses ours behind their handlers.

CONTENT IS NOT LOGGED BY DEFAULT. `log_answer` records the SHAPE of a turn -
timings, hit count, verdict, how many caveats fired - and not the question or
the answer text. A log of what a thousand colleagues asked their compliance
assistant is a different artifact with different retention and access rules,
so it is opt-in through RME_LOG_CONTENT rather than a side effect of turning
logging on.
"""
from __future__ import annotations

import logging
import sys

import config

_ready = False


def setup() -> logging.Logger:
    """Configure the `rme` logger once; safe to call from every entry point."""
    global _ready
    log = logging.getLogger("rme")
    if _ready:
        return log
    log.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
    log.propagate = False           # do not hand our records to root's handlers
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    log.addHandler(stream)
    if config.LOG_FILE:
        try:
            fh = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
            fh.setFormatter(fmt)
            log.addHandler(fh)
        except OSError as e:
            # An unwritable log path must not stop the app from answering.
            log.warning("log file %s unusable (%s); logging to stderr only",
                        config.LOG_FILE, e)
    for name, raw, default in config.BAD_ENV:
        log.warning("ignoring %s=%r (not an integer); using %s", name, raw, default)
    log.info("config %s", config.settings_summary())
    _ready = True
    return log


def get(name: str = "rme") -> logging.Logger:
    """A child logger. Does not configure anything - setup() owns that."""
    return logging.getLogger(name if name.startswith("rme") else f"rme.{name}")


def log_answer(rec: dict, log: logging.Logger | None = None) -> None:
    """One line per answered question: shape and timing, not content.

    Every field here is one someone asks about when an answer is disputed -
    which model, how long, what the validator said, whether the answer was
    built from an incomplete diagram or from one of several versions of the
    same process. `question_chars` stands in for the question itself so the
    line still distinguishes turns without recording what was asked.
    """
    log = log or get()
    fields = {
        "lang": rec.get("lang"), "model": rec.get("model"),
        "mode": rec.get("mode"),
        "question_chars": len(rec.get("question") or ""),
        "hits": len(rec.get("hits") or []),
        "wf_hits": rec.get("n_workflow_hits"),
        "verdict": (rec.get("validator") or {}).get("verdict"),
        "incomplete_wf": len(rec.get("incomplete_workflows") or []),
        "variant_conflicts": len(rec.get("variant_conflicts") or []),
        "retrieval_ms": round(1000 * (rec.get("retrieval_s") or 0)),
        "generation_s": round(rec.get("generation_s") or 0, 1),
        "total_s": round(rec.get("total_s") or 0, 1),
    }
    log.info("answered " + " ".join(f"{k}={v}" for k, v in fields.items()))
    if config.LOG_CONTENT:
        log.info("content question=%r answer=%r",
                 rec.get("question"), rec.get("answer"))

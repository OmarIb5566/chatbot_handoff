"""What can go wrong at runtime, as types the caller can act on differently.

WHY THIS EXISTS
---------------
`generate()` used to let `requests` exceptions and `raise_for_status()` out
raw, and app.py caught bare `Exception` and printed "Something went wrong
answering that." That sentence is the same whether Ollama died mid-answer, the
configured model was never pulled, or the answer took longer than the timeout
- three problems with three different fixes, two of which the user can act on
themselves. Losing that distinction at the boundary is why the failure looked
like a bug in the assistant rather than a missing `ollama pull`.

The split is by WHAT THE READER SHOULD DO, not by what threw:

    ModelUnavailable   the server is not answering        -> start/reach Ollama
    ModelNotFound      the server is up, model not pulled -> ollama pull X
    ModelTimeout       it is working, just not in time    -> retry, or raise
                                                             RME_GEN_TIMEOUT
    CorpusMissing      the index has not been built       -> run the pipeline
    GenerationFailed   anything else from the model API   -> read the message

All carry a user-facing sentence in `.user_message`, so the UI renders one
string and does not re-derive the wording from the exception type. Everything
inherits from `PipelineError`, so a caller that genuinely wants "any of ours"
can still catch one thing - and a bug in this codebase, which is not a
PipelineError, keeps propagating instead of being flattened into a polite
message.
"""
from __future__ import annotations


class PipelineError(Exception):
    """Base for every expected runtime failure. Bugs must NOT use this."""

    user_message = "Something went wrong answering that. Please try again."


class ModelUnavailable(PipelineError):
    user_message = ("The assistant is not available right now - the model "
                    "server is not responding. If you are running this "
                    "yourself, start it with `ollama serve` and reload.")


class ModelNotFound(PipelineError):
    user_message = ("The configured model is not installed on the model "
                    "server. Pull it with `ollama pull <model>` and reload.")


class ModelTimeout(PipelineError):
    user_message = ("That answer took longer than the time allowed and was "
                    "stopped. Try a narrower question, or raise "
                    "RME_GEN_TIMEOUT if long answers are expected.")


class GenerationFailed(PipelineError):
    user_message = ("The model server rejected that request. The details are "
                    "in the application log.")


class CorpusMissing(PipelineError):
    user_message = ("The document index has not been built yet, so there is "
                    "nothing to search. Build it with the extraction pipeline "
                    "in backend/ before starting the app.")

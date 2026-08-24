"""Every path in the project, in one place.

WHY THIS EXISTS
---------------
Each module used to carry its own `HERE = Path(__file__).resolve().parent` and
build data paths off it - twelve copies of the same idea, each assuming the
code and the JSON it reads live in the same directory. That held only while
everything sat in one folder, so splitting the repo into backend/, evals/,
eval_results/ and data/ broke all of them at once.

Anchoring on the repo ROOT instead of on each file's own directory means a
module can move between folders without its data paths changing. This is the
last time a reorganisation should require touching path constants.

    ROOT/
      app.py                 the Streamlit UI, deliberately standalone
      backend/               the pipeline: chunking, retrieval, generation
      evals/                 harnesses AND the question sets they run
      eval_results/          generated scores - regenerable, safe to delete
      data/                  chunks and extraction audits - regenerable
      processes_pdf/         source documents  (input, not generated)
      Workflows/             source flowcharts (input, not generated)

Nothing in data/ or eval_results/ is authored by hand: both are outputs of a
script in backend/ or evals/, which is what makes them safe to wipe and
rebuild. processes_pdf/ and Workflows/ are the only irreplaceable directories.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BACKEND = ROOT / "backend"
EVALS = ROOT / "evals"
EVAL_RESULTS = ROOT / "eval_results"
DATA = ROOT / "data"

# --- inputs: the only directories that cannot be regenerated ---------------
PROCESSES_PDF = ROOT / "processes_pdf"
WORKFLOWS = ROOT / "Workflows"

# --- generated corpus ------------------------------------------------------
CHUNKS_JSON = DATA / "chunks.json"                     # adaptive_chunker.py
WORKFLOW_CHUNKS_JSON = DATA / "workflow_chunks.json"   # workflow_vector.py
WORKFLOW_CHUNKS_SCANNED = DATA / "workflow_chunks_scanned.json"
WORKFLOW_AUDIT = DATA / "workflow_audit.json"
WORKFLOW_VECTOR_AUDIT = DATA / "workflow_vector_audit.json"
WORKFLOW_SCANNED_AUDIT = DATA / "workflow_scanned_audit.json"
EXTRACTED_RAW = DATA / "extracted_raw.json"
OCR_FALLBACK_LOG = DATA / "ocr_fallback_log.json"

# Written next to the corpus it describes, and keyed by a fingerprint of that
# corpus - see Retriever._corpus_fingerprint. Re-chunking therefore misses the
# cache and re-encodes, which is the intended behaviour: a stale embedding is
# worse than a slow first load.
EMBED_CACHE = DATA / ".embed_cache"

# --- eval sets (inputs to the harnesses) -----------------------------------
EVAL_SET = EVALS / "eval_set.json"
EVAL_SET_AR = EVALS / "eval_set_ar.json"
EVAL_SET_V2 = EVALS / "eval_set_v2.json"
EVAL_SET_V2_AR = EVALS / "eval_set_v2_ar.json"
EVAL_SET_APP = EVALS / "eval_set_app.json"

# --- eval results (outputs) ------------------------------------------------
APP_EVAL_RESULTS = EVAL_RESULTS / "app_eval_results.json"
GENERATION_EVAL_RESULTS = EVAL_RESULTS / "generation_eval_results.json"
MODEL_EVAL_RESULTS = EVAL_RESULTS / "model_eval_results.json"


def add_backend_to_path() -> None:
    """Make `import retriever` work from a script outside backend/.

    The backend modules import each other by bare name (`import validator`,
    `from retriever import Retriever`). That is worth keeping - it reads well
    and it is what the notebook and every existing docstring assume - but it
    only resolves when backend/ is on sys.path. Entry points outside that
    directory (app.py at the root, the harnesses in evals/) call this first.
    """
    import sys

    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))

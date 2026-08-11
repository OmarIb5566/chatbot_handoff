"""
Hybrid retrieval over the chunked RME process docs: BM25 + dense embeddings.

Why hybrid rather than either alone:

  * BM25 alone nails exact tokens but gets dragged around by cross-document
    term overlap. These 9 docs share almost all of their vocabulary
    ("process", "manager", "approval", "form", "within days"), so a query
    like "form number for the Request for Quotation (RFQ)" scores well on
    chunks from the *wrong* doc that happen to repeat those words. That is
    exactly the PVMO01-1 baseline miss.

  * Dense alone handles paraphrase ("who signs off on" ~ "approved by") but
    is structurally bad at form numbers. 'F-P-CN-01-11' and 'F-P-VMO-01-01'
    land in nearly the same place in MiniLM space - the model never learned
    that the middle token is the only part that matters.

So we keep both signals and fuse them. BM25 stays the exact-match channel;
embeddings supply the semantic channel.

Both fusion methods are implemented and both were measured (see the notebook):

    BM25 only (baseline)        31/32 =  96.9%
    Dense only (MiniLM)         30/32 =  93.8%
    Hybrid RRF                  31/32 =  96.9%
    Hybrid weighted (w=0.3-0.5) 32/32 = 100.0%
    Hybrid weighted + routing   32/32 = 100.0%

Default is weighted-sum at dense_weight=0.4. That is the empirical result,
but read it honestly: the gap is ONE question out of 32. The eval set is
saturated - it cannot resolve differences smaller than 3.1%, so this is not
evidence that weighted-sum beats RRF in general. RRF was the a-priori
choice (it fuses ranks, so it needs no retuning as the corpus grows 60x,
whereas a weighted sum mixes an unbounded BM25 score with a bounded cosine
and can need re-tuning). If accuracy on a larger eval set ever ties, prefer
RRF for that reason. `fusion="rrf"` is one keyword away.

The 100% is stable across dense_weight 0.3-0.5 (a plateau, not a lucky
point), and prefix routing reaches 100% at every weight including w=0.0.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

# Use the embedding model already cached on this machine and never reach out to
# the Hub. Set before sentence_transformers is imported (it reads these at
# import time). `setdefault` so an explicit env var from the caller still wins.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from rank_bm25 import BM25Okapi

HERE = Path(__file__).resolve().parent
CHUNKS_JSON = HERE / "chunks.json"   # written by adaptive_chunker.py
CACHE_DIR = HERE / ".embed_cache"

DEFAULT_MODEL = "all-MiniLM-L6-v2"
RRF_K = 60  # standard RRF damping constant


# --- doc-code prefix routing -------------------------------------------------
# Pre-retrieval narrowing for the 500-600 doc corpus: if a query clearly names
# a process family, search only that family instead of the whole index.
#
# This lived in chunker.py until that module was retired. It never belonged
# there - routing is a retrieval-time decision about a *query*, not a chunking
# decision about a document - so it moved here, to its only consumer.
DOC_PREFIXES = ["PSE", "PQD", "PCM", "PTN", "PVMO", "PCN"]

# Maps a doc-code prefix to the words that should route a query to it.
PREFIX_HINTS = {
    "PSE":  ["self-execution", "self execution", "sem", "selfexecution"],
    "PQD":  ["quality", "inspection", "testing", "itp", "pqp", "qc", "hold point"],
    "PCM":  ["customer", "satisfaction", "branding", "marketing", "survey", "commercial"],
    "PTN":  ["tender", "tendering", "initiation", "launching", "bid submission", "loa", "handover"],
    "PVMO": ["vendor", "procurement", "purchase", "rfq", "quotation", "sourcing", "supplier"],
    "PCN":  ["subcontract", "subcontractor", "contract", "agreement", "btb", "back-to-back"],
}


def doc_code_prefix(chunk: dict) -> str | None:
    """'P-VMO-01' -> 'PVMO', 'M-HR-08' -> 'MHR'.

    Falls back to the filename when doc_code is missing. The M family was added
    with the full corpus: the 20 M-HR-* HR policies are a fifth of the documents
    and matched nothing while this required a literal P.
    """
    code = chunk.get("doc_code") or chunk.get("filename", "")
    m = re.match(r"([PM])-?([A-Z]{2,3})-?\d", code.upper())
    return f"{m.group(1)}{m.group(2)}" if m else None


def route_prefixes(query: str) -> list[str]:
    """Doc-code prefixes to restrict the search to. Explicit codes ONLY.

    Empty list means "no confident route, search everything" - the safe
    default, since a wrong route is an unrecoverable miss while a missing route
    only costs latency.

    KEYWORD HINTS NO LONGER HARD-ROUTE, AND THAT IS A BUG FIX, NOT A RETREAT
    ------------------------------------------------------------------------
    PREFIX_HINTS covers 6 families. The 9-document corpus had 6. The full corpus
    has 19, so a keyword table that was exhaustive became a table that knows
    about a third of the index - and hard routing on it does not degrade
    gracefully, it *excludes*. Measured on eval_set_v2.json: routing fired on 19
    of 92 questions and sent 8 of them away from their own gold document, which
    is 8 of the 11 retrieval failures in the generation run. Every one is a
    keyword that used to be unambiguous and no longer is:

        "handover"    -> PTN, excluding PEN / PFW / PLO
        "subcontract" -> PCN, excluding POP (Subcontractor Management,
                         Variation Order, Earthmoving)
        "vendor"      -> PVMO, excluding PPR (Evaluation of External Providers)

    Extending the table to 19 families would buy back the same failure at the
    next document intake, and is the mistake adaptive_chunker.py already
    documents making with its literal heading list. An explicit code in the
    query is different in kind: "what is P-OP-02" cannot mean anything else, so
    that stays a hard route.

    Keyword hints keep their real job - `soft_prefixes` boosts on them, which
    nudges ranking without excluding anything and cannot cause this failure.
    """
    # "what is P-VMO-01", "F-P-CN-01-11", "M-HR-08".
    #
    # No intersection with DOC_PREFIXES any more: that list is the original six,
    # so "what is P-OP-02 about" intersected to nothing and did not route at all.
    # Validation moved to _candidates(), which checks the prefix against the
    # corpus actually loaded rather than a hard-coded list that goes stale.
    explicit = {f"{fam}{sub}" for fam, sub
                in re.findall(r"\bF?-?([PM])-([A-Z]{2,3})-\d", query.upper())}
    return sorted(explicit)


def soft_prefixes(query: str) -> list[str]:
    """Every doc-code family the query names. No ambiguity gate.

    The soft-boost counterpart to route_prefixes(). Same keyword table, but
    without the "exactly one family" rule, because the two are protecting
    against different things: a hard route that picks wrong excludes the answer
    permanently, whereas boosting two families just nudges both and lets the
    scores decide.

    That rule actively broke the boost. A rewritten follow-up naturally mentions
    two topics - "What form is used for the Customer Satisfaction Survey in the
    self-execution process?" hits PCM and PSE - so route_prefixes returned [],
    and the family signal was empty for precisely the queries contextualisation
    exists to produce.
    """
    q = query.lower()
    explicit = route_prefixes(query)
    if explicit:
        return explicit
    return [p for p, words in PREFIX_HINTS.items() if any(w in q for w in words)]


def _singular(word: str) -> str:
    """Crude plural strip, applied identically to corpus and query.

    BM25 does no stemming, so "stakeholders" and "stakeholder" were entirely
    different terms. That is not a theoretical concern: "who are the liable
    stakeholder..." answered correctly while "who are the liable
    stakeholders..." returned nothing useful, purely on the trailing s.

    Deliberately a suffix rule and not a real stemmer. A Porter stemmer would
    also fold "processing"/"processes"/"process" together, and "process" is the
    single most common word in this corpus - collapsing its forms costs more
    discrimination than it buys. The `ss`/`us`/`is` guard keeps "process",
    "status" and "analysis" intact, and the length guard keeps short tokens and
    the pieces of a split form code ("f", "p", "cm", "01") untouched.
    """
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    return [_singular(w) for w in re.findall(r"[a-z0-9]+", text.lower())]


def embed_text(chunk: dict) -> str:
    """What actually gets embedded.

    A bare step body ("Within 7 days the VMO shall prepare...") carries no
    hint of which document it came from, and the eval scores documents. So
    we prepend title + section: it costs nothing and gives the dense channel
    the document-level context BM25 gets for free from term overlap.
    """
    return f"{chunk.get('title') or ''} | {chunk.get('section') or ''}\n{chunk['text']}"


class Retriever:
    """Hybrid BM25 + dense retriever.

    Drop-in for the old BM25-only class: `search()` keeps the same signature
    and returns the same dicts (score + chunk fields), so the existing
    notebook harness and `retrieval_hit` work unchanged.
    """

    def __init__(
        self,
        chunks: list[dict],
        model_name: str = DEFAULT_MODEL,
        fusion: str = "weighted",
        dense_weight: float = 0.4,
        use_cache: bool = True,
        device: str | None = None,
    ):
        self.chunks = chunks
        self.fusion = fusion
        self.dense_weight = dense_weight
        self.model_name = model_name

        # --- sparse channel ---
        # Indexed over embed_text (title + section + body), not the body alone,
        # for the same reason the dense channel is - and one more.
        #
        # The section LABEL is canonical; the body text is whatever the PDF
        # says. PVMO01's stakeholder heading is misspelled in the source as
        # "2. STAKHOLDER", so the body contains no token a question about
        # stakeholders can match. adaptive_chunker's fuzzy labeller already
        # recovered the correct name into `section`, but BM25 could not see it,
        # and the 94-character body is too short to win on dense similarity.
        # The result was that every plural stakeholder question returned
        # OBJECTIVES instead. Indexing the label fixes that class of miss for
        # any short, list-shaped section identified mainly by its heading.
        self.corpus_tokens = [tokenize(embed_text(c)) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

        # --- dense channel ---
        from sentence_transformers import SentenceTransformer

        # local_files_only: use the already-installed HF cache
        # (~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2)
        # instead of hitting the network to check for updates.
        self.model = SentenceTransformer(model_name, device=device, local_files_only=True)
        self.embeddings = self._get_embeddings(use_cache)
        self._build_faiss()

        # Precompute prefix routing labels once (see doc_code_prefix above).
        self.prefixes = [doc_code_prefix(c) for c in chunks]
        self._present_prefixes = {p for p in self.prefixes if p}

    # ------------------------------------------------------------------ index
    def _corpus_fingerprint(self) -> str:
        h = hashlib.sha256(self.model_name.encode())
        for c in self.chunks:
            h.update(embed_text(c).encode("utf-8", errors="ignore"))
        return h.hexdigest()[:16]

    def _get_embeddings(self, use_cache: bool) -> np.ndarray:
        """Encode all chunks, caching to disk keyed by corpus+model hash.

        At 600 docs this is ~7k chunks; re-encoding on every notebook restart
        is minutes of wall time for no reason. The fingerprint means the cache
        invalidates itself the moment a chunk or the model changes.
        """
        if not use_cache:
            return self._encode_corpus()

        CACHE_DIR.mkdir(exist_ok=True)
        cache_file = CACHE_DIR / f"emb_{self._corpus_fingerprint()}.npy"
        if cache_file.exists():
            return np.load(cache_file)
        emb = self._encode_corpus()
        np.save(cache_file, emb)
        return emb

    def _encode_corpus(self) -> np.ndarray:
        texts = [embed_text(c) for c in self.chunks]
        emb = self.model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,   # so inner product == cosine
            show_progress_bar=len(texts) > 500,
        )
        return emb.astype("float32")

    def _build_faiss(self):
        import faiss

        dim = self.embeddings.shape[1]
        # Flat index: exact search. At 7k chunks (600 docs) this is still
        # sub-millisecond and exact; only move to IVF/HNSW past ~100k chunks,
        # where the recall loss starts buying real speed.
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    # --------------------------------------------------------------- channels
    def _bm25_ranking(self, query: str, candidates: np.ndarray) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        pairs = [(int(i), float(scores[i])) for i in candidates]
        return sorted(pairs, key=lambda x: -x[1])

    def _dense_ranking(self, query: str, candidates: np.ndarray) -> list[tuple[int, float]]:
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        q = q.astype("float32")
        if len(candidates) == len(self.chunks):
            scores, idx = self.index.search(q, len(self.chunks))
            return [(int(i), float(s)) for i, s in zip(idx[0], scores[0])]
        # Routed subset: score directly against the candidate rows.
        sub = self.embeddings[candidates]
        sims = (sub @ q[0])
        order = np.argsort(-sims)
        return [(int(candidates[j]), float(sims[j])) for j in order]

    def _candidates(self, query: str, route: bool) -> np.ndarray:
        """Pre-retrieval doc-code routing. Returns row indices to consider."""
        if not route:
            return np.arange(len(self.chunks))
        prefixes = route_prefixes(query)
        # A prefix no chunk carries is not a narrowing, it is an empty index.
        # Checked against the corpus that is actually loaded rather than a
        # module-level list, because the module-level list is what went stale:
        # DOC_PREFIXES still names the six families the 9-document corpus had,
        # and the index now holds 19.
        prefixes = [p for p in prefixes if p in self._present_prefixes]
        if not prefixes:
            return np.arange(len(self.chunks))
        # `p is None` means the chunk carries no doc code to route on - the
        # scanned workflow diagrams (workflow_extractor.py) are the case that
        # exists today. Routing can only ever say "this chunk belongs to the
        # wrong family"; about a chunk with no family it knows nothing, so
        # dropping it is a silent recall loss rather than a safe narrowing.
        # A question like "who approves procurement over 500 K" routes to PVMO
        # and would otherwise exclude the signature matrix that answers it.
        # Every one of the 114 process chunks has a prefix, so keeping the
        # unroutable ones cannot change the measured process retrieval.
        keep = np.array(
            [i for i, p in enumerate(self.prefixes) if p is None or p in prefixes],
            dtype=int,
        )
        # Never let routing empty the candidate pool - fall back to full search.
        return keep if len(keep) else np.arange(len(self.chunks))

    # ----------------------------------------------------------------- search
    def search(self, query: str, top_k: int = 5, route: bool = True,
               boost_prefixes: list[str] | None = None,
               boost: float = 0.15) -> list[dict]:
        """Hybrid search. Same return shape as the old BM25-only version.

        `route=True` applies doc-code prefix narrowing first. On this eval set
        routing fires on 22/36 questions with 0 unsafe routes (it never
        excluded the gold document). Caveat for scale: PREFIX_HINTS above is a
        hand-written keyword table. That is fine for 6 families / 9 docs; at
        500-600 docs it should be derived from document titles rather than
        maintained by hand, or routing becomes the thing that silently loses
        recall.
        """
        cand = self._candidates(query, route)
        bm = self._bm25_ranking(query, cand)
        dn = self._dense_ranking(query, cand)

        if self.fusion == "rrf":
            fused: dict[int, float] = {}
            for ranking in (bm, dn):
                for rank, (i, _) in enumerate(ranking):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        elif self.fusion == "weighted":
            # Min-max normalise each channel before mixing, since BM25 scores
            # are unbounded and cosine is not.
            fused = {}
            for ranking, w in ((bm, 1 - self.dense_weight), (dn, self.dense_weight)):
                vals = [s for _, s in ranking]
                lo, hi = min(vals), max(vals)
                span = (hi - lo) or 1.0
                for i, s in ranking:
                    fused[i] = fused.get(i, 0.0) + w * (s - lo) / span
        else:
            raise ValueError(f"unknown fusion: {self.fusion!r}")

        # --- soft family boost -------------------------------------------
        # A nudge, not a filter: nothing is excluded, and a strong match in
        # another family still outranks a weak match in the boosted one. Used
        # by contextualize.py to pass "the user said self execution process"
        # into ranking without turning it into a router.
        #
        # Scaled by the observed score range rather than added as a raw
        # constant, because the two fusion modes live on different scales -
        # weighted sums land near 0..1 while RRF scores are ~1/60. A fixed
        # +0.15 would be a gentle nudge in one and total domination in the
        # other.
        if boost_prefixes and boost:
            vals = fused.values()
            span = (max(vals) - min(vals)) or 1.0
            bump = boost * span
            wanted = set(boost_prefixes)
            for i in list(fused):
                if self.prefixes[i] in wanted:
                    fused[i] += bump

        ranked = sorted(fused.items(), key=lambda x: -x[1])[:top_k]
        return [{"score": float(s), **self.chunks[i]} for i, s in ranked]

    # --- single-channel variants, kept for ablation/eval comparison ---
    def search_bm25(self, query: str, top_k: int = 5) -> list[dict]:
        cand = np.arange(len(self.chunks))
        ranked = self._bm25_ranking(query, cand)[:top_k]
        return [{"score": s, **self.chunks[i]} for i, s in ranked]

    def search_dense(self, query: str, top_k: int = 5) -> list[dict]:
        cand = np.arange(len(self.chunks))
        ranked = self._dense_ranking(query, cand)[:top_k]
        return [{"score": s, **self.chunks[i]} for i, s in ranked]


if __name__ == "__main__":
    import sys

    # Chunk text carries typographic artifacts from the PDFs (U+0336 bullets,
    # curly quotes) that a cp1252 Windows console cannot encode. Only affects
    # this demo print, but an encoding crash here looks like a broken retriever.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    chunks = json.load(open(CHUNKS_JSON, encoding="utf-8"))
    r = Retriever(chunks)
    q = "what form is used for the customer satisfaction survey"
    for res in r.search(q, top_k=3):
        print(f"[{res['score']:.4f}] {res['filename']} | {res['section'][:25]} | "
              f"{res['text'][:80].replace(chr(10), ' ')}")

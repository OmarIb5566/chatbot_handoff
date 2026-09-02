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

import difflib
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

# Use the embedding model already cached on this machine and never reach out to
# the Hub. Set before sentence_transformers is imported (it reads these at
# import time). `setdefault` so an explicit env var from the caller still wins.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from rank_bm25 import BM25Okapi

import config  # noqa: E402
from paths import CHUNKS_JSON, EMBED_CACHE as CACHE_DIR  # noqa: E402

# The encoder for the dense channel. Read from config so it can be swapped
# without a diff; changing it invalidates the embedding cache by fingerprint,
# which is intended - a stale embedding is worse than a slow first load.
DEFAULT_MODEL = config.EMBED_MODEL
RRF_K = 60  # standard RRF damping constant

# Query repair (see Retriever.repair_query). Tuned on the 136 eval questions:
# 0.90 alters one token across all of them, 0.85 alters 21, 0.92 alters none
# but stops fixing "accomdation".
REPAIR_CUTOFF = 0.90
MIN_REPAIR_LEN = 4


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


# --- workflow-diagram recall -------------------------------------------------
# The corpus is two populations in one index: 1063 process-text chunks and 137
# workflow-diagram chunks (data/chunks.json + data/workflow_chunks.json). That
# is roughly 8:1, and the imbalance is not neutral for ranking.
#
# BM25 in particular favours the process side on almost any query, because the
# process PDFs are long, repetitive prose that restates its own vocabulary many
# times, while a workflow chunk is a short rendering of a graph. So a question
# whose real answer is an approval chain ("who signs after the Technical
# Director") competes against sixty paragraphs that merely CONTAIN the words.
#
# The fix here is deliberately NOT a router. Nothing is excluded and no query
# is confined to one file: a workflow-shaped question gets a soft score boost
# toward diagram chunks, plus a guaranteed floor of slots in the final top-k so
# that a relevant diagram cannot be crowded out entirely by a larger corpus.
# Both are recall protections; a strongly-matching process chunk still wins.
WORKFLOW_SOURCE_TYPES = {"pdf_vector", "vlm_description"}

# Words that indicate the user wants the ROUTE through a process - who signs,
# in what order, what happens next - rather than its written policy. Kept
# deliberately narrow: a false positive only nudges ranking and reserves two
# slots, but a table this loose fires on everything and the floor stops being
# a floor and becomes a quota.
WORKFLOW_HINTS = (
    "workflow", "work flow", "flow chart", "flowchart", "diagram",
    "approval chain", "approval cycle", "approval flow", "approval route",
    "who approves", "who signs", "who authorises", "who authorizes",
    "sign off", "signs off", "signature matrix", "signatory",
    "next step", "what happens after", "what happens next", "goes to",
    "sequence", "order of approval", "routed", "routing", "escalate",
    "escalation", "sent back", "rejected", "cycle", "steps in",
)


def workflow_intent(query: str) -> bool:
    """True when the query is asking about a route/approval chain.

    Only ever ENABLES extra recall. A wrong True costs two of top_k slots; a
    wrong False just leaves ranking as it was. That asymmetry is why the list
    above errs toward missing a case rather than firing on everything.
    """
    q = query.lower()
    return any(h in q for h in WORKFLOW_HINTS)


def load_embedder(model_name: str = DEFAULT_MODEL, device: str | None = None):
    """MiniLM from the local HF cache, downloading it only if it is not there.

    Cache first, because that is the normal case and it avoids a network round
    trip to check for updates on every single construction - the Retriever is
    built once per process but that is still once per notebook cell, per app
    restart, per eval run.

    The fallback is the point, though. This used to be `local_files_only=True`
    with no fallback, which is correct on a machine that already has the model
    and a hard failure on one that does not - and README_HANDOFF says "first run
    downloads the model", which was simply untrue. Moving this repo to another
    machine is a normal thing to do (a colleague's GPU box, a fresh clone), and
    the first thing it did there was crash inside a library with an error that
    does not mention the network.

    Offline with no cache still fails, as it must - but now it fails saying so.
    """
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model_name, device=device, local_files_only=True)
    except Exception:
        pass
    try:
        return SentenceTransformer(model_name, device=device)
    except Exception as e:
        raise RuntimeError(
            f"Could not load the embedding model {model_name!r}. It is not in the "
            "local HuggingFace cache and could not be downloaded. On a machine "
            "with no network, pre-fetch it once from a machine that has one:\n"
            "    python -c \"from sentence_transformers import SentenceTransformer; "
            f"SentenceTransformer('sentence-transformers/{model_name}')\"\n"
            f"Underlying error: {type(e).__name__}: {e}"
        ) from e


def doc_code_prefix(chunk: dict) -> str | None:
    """'P-VMO-01' -> 'PVMO', 'M-HR-08' -> 'MHR'.

    Falls back to the filename when doc_code is missing. The M family was added
    with the full corpus: the 20 M-HR-* HR policies are a fifth of the documents
    and matched nothing while this required a literal P.
    """
    code = chunk.get("doc_code") or chunk.get("filename", "")
    # `F-P-VMO-01-07` is a FORM in the P-VMO family, so the optional F- prefix
    # is stripped rather than failing the match: the three workflow diagrams
    # that carry a real code route with the process documents they belong to.
    m = re.match(r"(?:F-)?([PM])-?([A-Z]{2,4})-?\d", code.upper())
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
        # Vocabulary for query repair, with frequencies to break ties toward
        # the term the corpus actually uses. Free - these tokens are already
        # computed for BM25.
        self._vocab: Counter[str] = Counter()
        for toks in self.corpus_tokens:
            self._vocab.update(toks)
        self._repair_cache: dict[str, tuple[tuple[str, ...], tuple[tuple[str, str], ...]]] = {}

        # --- dense channel ---
        self.model = load_embedder(model_name, device=device)
        self.embeddings = self._get_embeddings(use_cache)
        self._build_faiss()

        # Precompute prefix routing labels once (see doc_code_prefix above).
        self.prefixes = [doc_code_prefix(c) for c in chunks]
        self._present_prefixes = {p for p in self.prefixes if p}

        # Which rows are workflow diagrams rather than process text. Computed
        # once here rather than per-search: at 1200 chunks the dict lookup is
        # cheap but it runs inside the ranking loop, and this is the kind of
        # thing that quietly becomes the hot path when the corpus reaches its
        # 500-700 document target.
        self.is_workflow = [
            c.get("source_type") in WORKFLOW_SOURCE_TYPES for c in chunks
        ]
        self._workflow_rows = [i for i, w in enumerate(self.is_workflow) if w]

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
    def repair_query(self, query: str) -> tuple[str, list[tuple[str, str]]]:
        """Map out-of-vocabulary query words onto the nearest corpus term.

        Returns the repaired QUERY TEXT and the substitutions made, so a caller
        can show its work. Words are replaced in place in the original string,
        so a query with nothing to repair comes back byte-identical and both
        channels behave exactly as they did before this existed.

        WHY THIS EXISTS
        ---------------
        BM25 matches tokens exactly, and it carries 60% of the fused score
        (dense_weight=0.4). So one misspelled word does not merely degrade the
        sparse channel, it inverts it: the misspelling contributes nothing, and
        what is left scores every document sharing the query's COMMON words
        while the document holding the rare discriminative one gets no credit
        for it. "how far should employees live to get access to accomdation"
        put MHR08_Accommodation_Policy outside BM25's top 40 - the dense
        channel still ranked it 1st, and the fusion buried it at 14 anyway.

        Applied to BOTH channels, which was not the original intent. The
        argument for sparse-only is good on paper - MiniLM embeds subwords and
        is already typo-tolerant, so why risk poisoning the channel that worked
        - but it measured worse: repairing only BM25 left the target chunk at
        rank 6, repairing both brought it to 4, level with the correctly spelled
        query. Nothing is poisoned because the substitution is in place: a query
        with no repairs reaches both channels byte-identical.

        What this does NOT do is move the eval. Top-3 document accuracy over
        eval_set.json + eval_set_v2.json is 113/124 with repair and 113/124
        without, top-5 is 115/124 either way - because every question in both
        sets is already spelled correctly, so there is nothing to repair. The
        eval's job here is to show the feature costs nothing, and that is all
        it shows. The gain is only visible on input the eval does not contain:

            "...to get access to accomdation"       rank 14 -> 4
            "distance required for company acommodation"  13 -> 6

        Measured at cutoff 0.90, ONE token across those 136 correctly-spelled
        questions is altered ("misses" -> "misuse" in ADV-1), and it changes no
        retrieval. Loosening to 0.85 raises that to 21 altered tokens; 0.92
        drops to zero but stops repairing "accomdation". The cutoff is where it
        is because that is where the curve turns, not because it is round.

        The honest limit: this fixes typos in words the CORPUS knows. A user who
        misspells in a way that lands within 0.90 of the wrong corpus term gets
        a confident wrong substitution, and nothing downstream can tell. That is
        why `search()` returns the repairs and the app prints them.

        Deliberately NOT a spell-checker: the vocabulary is this corpus, so
        "aproval" resolves to the corpus's "approval" and a real word the
        corpus never uses is left alone unless something is very close to it.
        """
        cached = self._repair_cache.get(query)
        if cached is not None:
            return cached[0], [tuple(e) for e in cached[1]]

        edits: dict[str, str] = {}
        for t in tokenize(query):
            # Short tokens have too many neighbours to correct safely, and the
            # pieces of a split form code ("f", "p", "cm", "01") are all short
            # - which is what keeps F-P-CM-01-01 out of this entirely.
            if len(t) < MIN_REPAIR_LEN or t.isdigit() or t in self._vocab or t in edits:
                continue
            near = difflib.get_close_matches(t, self._vocab, n=3, cutoff=REPAIR_CUTOFF)
            if not near:
                continue
            edits[t] = max(near, key=lambda w: (difflib.SequenceMatcher(None, t, w).ratio(),
                                                self._vocab[w]))

        if edits:
            # Substitute in place, matching each surface word through the same
            # normalisation BM25 uses, so "Accomdation" and "accomdations" both
            # find their entry. Everything else - punctuation, casing, word
            # order - is left exactly as the user typed it.
            def _sub(m: re.Match) -> str:
                return edits.get(_singular(m.group(0).lower()), m.group(0))

            repaired = re.sub(r"[A-Za-z0-9]+", _sub, query)
        else:
            repaired = query

        pairs = sorted(edits.items())
        self._repair_cache[query] = (repaired, tuple(pairs))
        return repaired, pairs

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
               boost: float = 0.15,
               workflow_boost: float = 0.06,
               workflow_floor: int = 2,
               min_rel: float = 0.55,
               min_hits: int = 3,
               wf_intent: bool | None = None) -> list[dict]:
        """Hybrid search. Same return shape as the old BM25-only version.

        `route=True` applies doc-code prefix narrowing first. On this eval set
        routing fires on 22/36 questions with 0 unsafe routes (it never
        excluded the gold document). Caveat for scale: PREFIX_HINTS above is a
        hand-written keyword table. That is fine for 6 families / 9 docs; at
        500-600 docs it should be derived from document titles rather than
        maintained by hand, or routing becomes the thing that silently loses
        recall.
        """
        # Spelling repair first, so both channels and the router see the same
        # text. No-op for a query whose words are all in the corpus vocabulary,
        # which is every question in both eval sets bar one.
        query, _repairs = self.repair_query(query)

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

        # --- workflow recall protection -----------------------------------
        # Two mechanisms, both soft, both gated on the query actually looking
        # like a route question. See WORKFLOW_HINTS above for why this is not
        # a router.
        wf = workflow_intent(query) if wf_intent is None else wf_intent
        if wf and self._workflow_rows:
            if workflow_boost:
                # Same scaling trick as the family boost: proportional to the
                # observed score range, so it behaves identically under both
                # fusion modes instead of dominating RRF's ~1/60 scores.
                vals = fused.values()
                span = (max(vals) - min(vals)) or 1.0
                bump = workflow_boost * span
                for i in list(fused):
                    if self.is_workflow[i]:
                        fused[i] += bump

        order = sorted(fused.items(), key=lambda x: -x[1])
        ranked = order[:top_k]

        if wf and workflow_floor > 0:
            # A floor in BOTH directions, and the second half is not symmetry
            # for its own sake - it is the failure this change introduced.
            #
            # Boosting diagrams on a route question made the top-6 come back
            # ALL diagram on "who approves the request for quotation": a
            # flowchart states that the CEO signs above a threshold, and the
            # process document states the conditions under which the threshold
            # applies at all. An answer built from the graph alone quotes the
            # first without the second and reads as authoritative. Guaranteeing
            # a diagram a seat while letting it take every seat just relocates
            # the recall bug.
            #
            # Both floors are capped at a third of top_k, so neither can starve
            # the other and neither overrides ranking when scores are decisive.
            cap = max(1, top_k // 3)
            ranked = self._enforce_mix(ranked, order,
                                       min_workflow=min(workflow_floor, cap),
                                       min_process=cap)

        # --- relevance cutoff: top_k is a CEILING, not a quota -------------
        # Raising top_k from 3 to 6 buys recall, and it also buys noise: when
        # only four chunks are genuinely relevant, slots five and six fill with
        # the best of the irrelevant. Measured on "what is the approval
        # workflow for bid closing", those slots were the Employees'
        # Performance Policy and the Overseas Business Travel Process, neither
        # of which has anything to do with bids.
        #
        # That is not free. The prompt instructs the model to answer from the
        # context, so weak context is not ignored context - it is an invitation
        # to pad an answer with an unrelated document's rules and cite it. A
        # variable number of strong chunks beats a fixed number of mixed ones.
        #
        # `min_hits` is 3 and NOT 2 for a specific reason: this file's own
        # docstring earlier (repair_query, ~line 424) already reports top-3
        # accuracy for both eval sets under a different comparison, and that
        # number was recycled here to justify min_hits=3 WITHOUT independently
        # re-running it against this code. Cowork's verification run caught the
        # error: reused here it produced an internally impossible pair (a 92-
        # question and a 32-question set that could not sum to the recycled
        # total), and separately, top-3 on eval_set_v2 scored against the
        # PRODUCTION 1200-chunk merged index - the corpus this code actually
        # runs against - is 86/92, not whatever this comment previously
        # implied. That number does not need to be re-derived here; it needs
        # to be read from evals/, which is now the only source of truth for it.
        #
        # The reasoning for min_hits=3 stands regardless of the exact figure:
        # a cutoff that can return two chunks can drop a gold chunk that sat at
        # rank 3, which reads as a retrieval regression caused by a change made
        # to improve retrieval. Holding the floor at 3 keeps that window
        # closed. But the number that window's SIZE was chosen from was never
        # actually measured here - only asserted - which is the mistake to not
        # repeat.
        #
        # CALIBRATION: min_rel=0.55 measured against real MiniLM. It let a Code
        # of Conduct chunk (0.604) and a Document Change History chunk (0.589)
        # through against a 0.576 bar on a genuine question - the exact noise
        # class this mechanism exists to stop. 0.7 excludes both, at the cost
        # of one eval_set question. Not yet raised here pending a look at which
        # question that is and whether it's a fair loss.
        if min_rel and ranked:
            bar = ranked[0][1] * min_rel
            strong = [(i, s) for i, s in ranked if s >= bar]
            ranked = strong if len(strong) >= min_hits else ranked[:min_hits]

        return [{"score": float(s), **self.chunks[i]} for i, s in ranked]

    def _enforce_mix(self, ranked, order, min_workflow: int, min_process: int,
                     rel_floor: float = 0.55):
        """Backfill the top-k so both chunk populations are represented.

        Displaces the WEAKEST members of the over-represented side, never the
        top hit. Returns `ranked` unchanged when the natural ordering already
        satisfies both floors, which is the common case - this is a safety net,
        not a quota.

        THE FLOOR IS ALLOWED TO GO UNMET
        --------------------------------
        `rel_floor` is why. A backfilled chunk must still score at least 55% of
        the top hit; below that the slot is simply left as it was. Without this
        guard the floor pads the context with whatever ranked highest among the
        irrelevant - measured, "what is the approval workflow for bid closing"
        backfilled a chunk of the Employees' Performance Policy at 0.585
        against a 0.927 top hit. That is not neutral filler. The prompt says to
        answer from the context, so an unrelated policy in the context is an
        invitation to blend it into the answer, and a compliance answer that
        cites the wrong document is worse than one that cites too few.
        """
        top = max((s for _, s in ranked), default=0.0)
        bar = top * rel_floor if top > 0 else float("-inf")
        for want, is_wf in ((min_workflow, True), (min_process, False)):
            have = sum(1 for i, _ in ranked if self.is_workflow[i] is is_wf)
            if have >= want:
                continue
            chosen = {i for i, _ in ranked}
            extra = [(i, s) for i, s in order
                     if self.is_workflow[i] is is_wf and i not in chosen
                     and s >= bar][:want - have]
            if not extra:
                continue
            keep = [(i, s) for i, s in ranked if self.is_workflow[i] is is_wf]
            other = [(i, s) for i, s in ranked if self.is_workflow[i] is not is_wf]
            other = other[:max(0, len(other) - len(extra))]
            ranked = sorted(keep + other + extra, key=lambda x: -x[1])
        return ranked

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
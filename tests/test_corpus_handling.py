"""Chunk selection and the context budget - the logic between retrieval and
the model, where a mistake is invisible in the answer."""
from __future__ import annotations

from conftest import chunk, wf_chunk

from chatbot import (CONTEXT_BUDGET_CHARS, drop_unreadable, fit_context,
                     ground_truth_chunks)


def test_budget_trims_the_lowest_ranked():
    # A fifth of the budget each, so four fit and the rest do not. Sized off
    # CONTEXT_BUDGET_CHARS rather than a literal, so raising the budget in
    # config does not silently turn this into a test of nothing.
    size = CONTEXT_BUDGET_CHARS // 5
    big = [chunk(filename=f"{i}.pdf", text="x" * size) for i in range(8)]
    kept = fit_context(big)
    assert len(kept) == 4
    assert kept == big[:4]          # order preserved, weakest dropped


def test_the_top_hit_survives_even_when_it_alone_busts_the_budget():
    """A single oversized chunk is a CHUNKING problem. Returning nothing would
    hide it behind an answer that says the context is empty."""
    huge = chunk(text="x" * (CONTEXT_BUDGET_CHARS * 2))
    assert fit_context([huge, chunk(filename="b.pdf")]) == [huge]


def test_nothing_is_dropped_when_it_all_fits():
    hits = [chunk(filename=f"{i}.pdf") for i in range(6)]
    assert fit_context(hits) == hits


def test_a_chunk_whose_every_node_is_blank_is_dropped():
    blank = wf_chunk(nodes=("", ""), edges=())
    for n in blank["graph"]["nodes"]:
        n["label"] = ""
    assert drop_unreadable([blank]) == []


def test_a_chunk_with_some_labels_is_kept():
    """ROUND 2 REGRESSION.

    The first version of drop_unreadable matched a page-scoped warning string
    and deleted all three flows off one page to remove the one that deserved
    it - costing a real eval question whose answer lived in a kept flow. The
    test is on the chunk's OWN nodes, so a chunk with any label survives no
    matter what its page's warnings say.
    """
    c = wf_chunk(nodes=("a", "b"), edges=(("a", "b"),),
                 warnings=["3 node(s) carry no label"])
    c["graph"]["nodes"][1]["label"] = ""
    assert drop_unreadable([c]) == [c]


def test_chunks_without_a_graph_are_never_dropped():
    """vlm_description prose has no graph and a different quality signal."""
    prose = chunk(source_type="vlm_description", text="Signatures required...")
    assert drop_unreadable([prose]) == [prose]


def test_the_form_whitelist_excludes_model_written_text():
    """The whitelist is what makes a hallucinated citation detectable. Let
    model-generated text in and an invented form number whitelists itself,
    which is worse than having no validator at all."""
    truth = ground_truth_chunks([chunk(), wf_chunk(),
                                 chunk(source_type="vlm_description")])
    assert [c["source_type"] for c in truth] == ["pdf_text", "pdf_vector"]

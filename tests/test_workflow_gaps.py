"""The incompleteness caveat: does it fire when it should, and only then."""
from __future__ import annotations

from conftest import chunk, wf_chunk

from chatbot import incomplete_workflows, workflow_gaps


def test_complete_graph_makes_no_complaint():
    assert workflow_gaps(wf_chunk(nodes=("a", "b"), edges=(("a", "b"),))) == []


def test_disconnected_step_is_reported_even_at_full_coverage():
    """The case coverage alone misses.

    All 19 chunks in the real corpus with a disconnected step have
    coverage 1.0: coverage counts text spans classified, not arrows
    recovered. If this test fails, the caveat has silently stopped
    covering its main case.
    """
    gaps = workflow_gaps(wf_chunk(nodes=("a", "b", "orphan"),
                                  edges=(("a", "b"),), coverage=1.0))
    assert len(gaps) == 1
    assert "no route in or out" in gaps[0]
    assert "Orphan" in gaps[0]


def test_unplaced_text_uses_the_extractors_own_count():
    """"2 of 9 spans" is actionable; "coverage 0.78" is not."""
    gaps = workflow_gaps(wf_chunk(
        coverage=0.7778,
        warnings=["2 of 9 text spans unclassified: ['Send Back with', 'Comments']"]))
    assert gaps == ["2 of 9 text spans unclassified on this page"]


def test_coverage_without_a_span_warning_falls_back_to_a_percentage():
    gaps = workflow_gaps(wf_chunk(coverage=0.5, warnings=["repaired: merged 0"]))
    assert gaps == ["only 50% of the page's text could be placed"]


def test_many_orphans_are_summarised_not_listed_in_full():
    gaps = workflow_gaps(wf_chunk(nodes=tuple(f"n{i}" for i in range(9)),
                                  edges=(("n0", "n1"),)))
    assert "and 3 more" in gaps[0]


def test_process_chunks_are_untouched():
    """No graph means a different chunk kind with a different quality signal."""
    assert workflow_gaps(chunk()) == []


def test_page_scoped_warning_is_not_repeated_once_per_chunk():
    """The defect the first live run exposed.

    workflow_vector.py writes the unplaced-text warning once per SOURCE PAGE
    and copies it onto every chunk split from that page. Three flows off one
    page therefore carried identical warnings, and the caveat rendered the
    same sentence three times - which reads as a bug and gets skipped.
    """
    page = [wf_chunk(section=f"Approval flow {i}", coverage=0.98,
                     warnings=["5 of 449 text spans unclassified: ['x']"])
            for i in range(3)]
    out = incomplete_workflows(page)
    assert len(out) == 1
    assert out[0]["gaps"] == ["5 of 449 text spans unclassified on this page"]
    # One page, several flows: no single section owns the entry.
    assert out[0]["section"] is None


def test_both_signals_on_one_page_merge_into_one_entry():
    c = wf_chunk(nodes=("a", "b", "end"), edges=(("a", "b"),), coverage=0.94,
                 warnings=["2 of 39 text spans unclassified: ['x']"])
    out = incomplete_workflows([c])
    assert len(out) == 1
    assert len(out[0]["gaps"]) == 2


def test_different_pages_stay_separate():
    a = wf_chunk(filename="A.pdf", page=1, nodes=("x", "y"), edges=())
    b = wf_chunk(filename="A.pdf", page=2, nodes=("x", "y"), edges=())
    assert len(incomplete_workflows([a, b])) == 2


def test_real_corpus_still_has_the_cases_the_caveat_was_built_for(real_corpus):
    """A canary on the corpus, not on the code.

    If a re-extraction fixes the disconnected steps, this fails and should be
    deleted - the caveat then has nothing to warn about. If it fails because
    the count JUMPED, the extraction regressed. Either way it wants a human.
    """
    orphaned = [c for c in real_corpus
                if any("no route" in g for g in workflow_gaps(c))]
    assert 10 <= len(orphaned) <= 30, f"{len(orphaned)} chunks with orphan steps"

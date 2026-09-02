"""Version collision: detected, labelled in the context, ruled on in the prompt."""
from __future__ import annotations

from conftest import chunk, wf_chunk

from chatbot import (build_prompt, format_context, variant_conflicts,
                     variant_labels)

V1 = wf_chunk(filename="Subcontracts - Subcontract Preparation V1.pdf",
              variant="Subcontract Preparation V1")
V3 = wf_chunk(filename="Subcontracts - Subcontract Preparation V3.pdf",
              variant="Subcontract Preparation V3")


def test_two_versions_of_one_family_are_a_conflict():
    cons = variant_conflicts([V1, V3])
    assert len(cons) == 1
    assert cons[0]["family"] == "Subcontracts"
    assert cons[0]["chosen"]["variant"] == "Subcontract Preparation V1"
    assert [o["variant"] for o in cons[0]["others"]] == ["Subcontract Preparation V3"]


def test_chosen_follows_retrieval_order_not_the_filename():
    """`chosen` is the top-RANKED document, so reordering the hits changes it."""
    assert variant_conflicts([V3, V1])[0]["chosen"]["variant"] == \
        "Subcontract Preparation V3"


def test_two_chunks_of_the_same_document_are_not_a_conflict():
    """Several flows off one diagram is the normal case, not an ambiguity."""
    a = wf_chunk(section="Approval flow 1")
    b = wf_chunk(section="Approval flow 2")
    assert variant_conflicts([a, b]) == []


def test_different_families_do_not_collide():
    other = wf_chunk(filename="Bonds Request - Bonds Request WF.pdf",
                     family="Bonds Request", variant="Bonds Request WF")
    assert variant_conflicts([V1, other]) == []


def test_process_chunks_never_collide():
    assert variant_conflicts([chunk(), chunk(filename="other.pdf")]) == []


def test_labels_only_go_on_documents_whose_sibling_is_present():
    """Telling the model about a version it cannot see invites a caveat about
    something that is not in front of it."""
    labels = variant_labels([V1, V3])
    assert set(labels) == {V1["filename"], V3["filename"]}
    assert variant_labels([V1]) == {}


def test_context_carries_the_label_above_the_chunk():
    ctx = format_context([V1, V3])
    assert 'This is version "Subcontract Preparation V1"' in ctx
    assert "NOT interchangeable" in ctx
    # The chunk text itself must still be there, unmodified.
    assert V1["text"] in ctx


def test_context_is_unchanged_when_there_is_no_collision():
    ctx = format_context([V1])
    assert "This is version" not in ctx
    assert ctx.startswith(f"[{V1['filename']} | {V1['section']}]")


def test_the_rule_fires_only_on_a_collision():
    assert "Other versions in the sources" in build_prompt("q", [V1, V3])
    assert "Other versions in the sources" not in build_prompt("q", [V1])


def test_the_rule_is_absent_from_a_pure_process_answer():
    p = build_prompt("how many leave days", [chunk()])
    assert "This is version" not in p
    # The workflow format block is workflow-only too.
    assert "Returns and loops" not in p


def test_every_workflow_chunk_carries_a_family(real_corpus):
    """variant_conflicts is keyed on `family`; a chunk without one is invisible
    to it, so a re-extraction that drops the field would disable the check
    silently rather than loudly."""
    missing = [c["filename"] for c in real_corpus if not c.get("family")]
    assert not missing, f"{len(missing)} workflow chunks with no family"


def test_the_families_that_motivated_this_still_have_siblings(real_corpus):
    families = {}
    for c in real_corpus:
        families.setdefault(c["family"], set()).add(c["filename"])
    assert len(families["Subcontracts"]) >= 4
    assert len(families["Document Submittal"]) >= 5

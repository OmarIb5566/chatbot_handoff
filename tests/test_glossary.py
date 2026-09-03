"""Glossary masking round-trips known terms without needing a live model.

The LLM-dependent path (does qwen3 actually respect the placeholders and the
hint block) is covered by the smoke test in translate.py's __main__ and by
evals/eval_arabic.py. What's testable without Ollama is the mechanism itself:
a known Arabic surface form must survive to_english() as its canonical
English term regardless of what any model would have done with it, because
mask_glossary/unmask_glossary never let the model see the original text.
"""
from __future__ import annotations

from glossary import (
    GLOSSARY,
    mask_english_terms,
    mask_glossary,
    unmask_glossary,
)


def test_known_arabic_phrase_masks_and_restores_to_canonical_term():
    # Real PTN01-3 case (README_HANDOFF.md, eval_set_ar.json), which also
    # legitimately hits "Tendering" via "المناقصات" (tender records) - both
    # are correct RME terms in this sentence, not a false positive.
    text = "كم مدة يجب أن يحتفظ قسم التجارة بسجلات المناقصات؟"
    masked, terms = mask_glossary(text)
    assert "قسم التجارة" not in masked and "المناقصات" not in masked
    assert set(terms) == {"Commercial Dept.", "Tendering"}
    restored = unmask_glossary(masked, terms)
    assert "Commercial Dept." in restored and "Tendering" in restored


def test_a_model_can_never_mistranslate_a_masked_term():
    """Simulates the exact documented failure: 'trade department' for قسم التجارة.

    Even if the stand-in generate_fn ignores the placeholder and returns
    garbage around it, the canonical term is restored afterwards regardless
    of what the model did with the rest of the sentence.
    """
    masked, terms = mask_glossary("قسم التجارة يحتفظ بالسجلات")
    fake_model_output = masked.replace("قسم التجارة يحتفظ", "the department keeps")
    result = unmask_glossary(fake_model_output, terms)
    assert "Commercial Dept." in result
    assert "trade department" not in result


def test_unmatched_text_is_untouched():
    masked, terms = mask_glossary("What form is used for the survey?")
    assert terms == []
    assert masked == "What form is used for the survey?"


def test_dropped_placeholder_is_appended_not_lost():
    masked, terms = mask_glossary("قسم التجارة")
    # Simulate the model dropping the placeholder entirely.
    assert unmask_glossary("some translation with no marker", terms) == \
        "some translation with no marker (Commercial Dept.)"


def test_reverse_direction_pins_the_preferred_arabic_rendering():
    """EN->AR: an English term must come back as RME's own Arabic term, not
    whatever qwen3 would translate it to live (the make_arabic_eval.py
    defect: "PM" -> رئيس الوزراء, "Prime Minister")."""
    masked, terms = mask_english_terms("Ask the PM about the Commercial Dept. records.")
    assert "PM" not in masked and "Commercial Dept." not in masked
    assert terms == [GLOSSARY["PM"][0], GLOSSARY["Commercial Dept."][0]]
    restored = unmask_glossary(masked, terms)
    assert GLOSSARY["PM"][0] in restored
    assert GLOSSARY["Commercial Dept."][0] in restored


def test_every_canonical_term_has_at_least_one_surface_form():
    for canonical, forms in GLOSSARY.items():
        assert forms, f"{canonical!r} has no Arabic surface forms"

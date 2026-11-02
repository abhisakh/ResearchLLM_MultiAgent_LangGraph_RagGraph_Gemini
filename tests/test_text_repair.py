"""PDF text repair: fix the known damage, never touch real scientific notation."""
import pytest

from backend.core.text_repair import MIN_WORD_LENGTH, repair_pdf_text


# --------------------------------------------------------------------------------------
# What is repaired (all taken from a real PDF: arXiv 2510.24874)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "damaged, repaired",
    [
        ("reHlecting", "reflecting"),
        ("efHiciency", "efficiency"),
        ("signiTicantly", "significantly"),
    ],
)
def test_f_decoded_as_h_or_t_inside_a_word_is_repaired(damaged, repaired):
    assert repair_pdf_text(damaged) == (repaired, 1)


def test_a_whole_sentence_is_repaired_and_counted():
    text, count = repair_pdf_text("The reHlecting layer improves efHiciency.")
    assert text == "The reflecting layer improves efficiency."
    assert count == 2


def test_unicode_ligature_characters_become_plain_letters():
    text, count = repair_pdf_text("\ufb01nd \ufb02ow o\ufb00 o\ufb03ce \ufb04 \ufb05 \ufb06")
    assert text == "find flow off office ffl st st"
    assert count == 0  # these are exact conversions, not guesses


def test_both_kinds_of_damage_in_one_text():
    assert repair_pdf_text("\ufb01rst reHlection")[0] == "first reflection"


# --------------------------------------------------------------------------------------
# What must never change
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "notation",
    [
        "BaTiO3", "SrTiO3", "LiTaO3", "CdTe", "ZnTe", "HgTe", "PbTe-based", "CH3NH3PbI3", "CsPbBr3",
        "Ti3C2Tx", "H2O", "NaH", "pH", "kT", "mTOR", "YouTube",
    ],
)
def test_chemistry_and_other_notation_is_untouched(notation):
    assert repair_pdf_text(f"The {notation} sample") == (f"The {notation} sample", 0)


@pytest.mark.parametrize("unit", ["kHz", "mHz", "nHz", "mmHg"])
def test_units_that_have_the_same_shape_are_untouched(unit):
    assert len(unit) < MIN_WORD_LENGTH
    assert repair_pdf_text(f"measured at 10 {unit}") == (f"measured at 10 {unit}", 0)


def test_ordinary_capitalised_words_are_untouched():
    sentence = "The High Temperature phase. Therefore Tin and Hydrogen were Treated. Hamiltonian."
    assert repair_pdf_text(sentence) == (sentence, 0)


def test_a_word_that_starts_with_the_wrong_letter_is_left_alone():
    # "Hind" (find) and "Tinite" (finite) cannot be told from correct capitalised words, so they stay as they are.
    # This test documents the limit of the repair; it is not a wish.
    assert repair_pdf_text("Hind Tinite") == ("Hind Tinite", 0)


# --------------------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------------------
def test_nothing_in_nothing_out():
    assert repair_pdf_text("") == ("", 0)
    assert repair_pdf_text(None) == ("", 0)


def test_repairing_twice_changes_nothing_more():
    once, _ = repair_pdf_text("reHlecting efHiciency")
    assert repair_pdf_text(once) == (once, 0)

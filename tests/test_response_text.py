"""response_text: reading the answer, and the diagnostic for words that arrive glued together."""
import logging
from types import SimpleNamespace

import pytest
from google.genai import types

from backend.core.errors import ErrorKind, LLMCallError
from backend.core.utilities import response_text


def response_with(*parts):
    """A real SDK response whose answer consists of the given parts."""
    content = types.Content(role="model", parts=list(parts))
    return types.GenerateContentResponse(candidates=[types.Candidate(content=content)])


def text(value, thought=False):
    return types.Part(text=value, thought=thought)


def test_a_one_part_answer_is_returned_without_any_warning(caplog):
    with caplog.at_level(logging.INFO):
        assert response_text(response_with(text("A clean answer.")), "synthesis") == "A clean answer."
    assert caplog.records == []


def test_parts_with_whitespace_between_them_are_joined_without_a_warning(caplog):
    with caplog.at_level(logging.INFO):
        answer = response_text(response_with(text("First part. "), text("Second part.")), "synthesis")
    assert answer == "First part. Second part."
    assert [r.levelno for r in caplog.records] == [logging.INFO]   # only the "arrived in 2 parts" note


def test_parts_that_touch_are_warned_about_with_the_text_around_the_boundary(caplog):
    with caplog.at_level(logging.INFO):
        answer = response_text(response_with(text("a distinct"), text("thermodynamic phase")), "synthesis")
    assert answer == "a distinctthermodynamic phase"      # what the reader would see: the glued words
    warning = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning) == 1
    assert "a distinct|thermodynamic p" in warning[0].getMessage()   # the text around the boundary
    assert "synthesis" in warning[0].getMessage()


def test_a_boundary_after_punctuation_without_a_space_is_flagged_too(caplog):
    with caplog.at_level(logging.WARNING):
        response_text(response_with(text("while cooling,"), text("progressing slowly")), "synthesis")
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_thought_parts_are_not_part_of_the_answer_or_the_diagnostic(caplog):
    with caplog.at_level(logging.INFO):
        answer = response_text(response_with(text("private reasoning", thought=True), text("The answer.")), "synthesis")
    assert answer == "The answer."
    assert caplog.records == []


def test_a_response_without_candidates_still_works_as_in_the_tests_fakes(caplog):
    with caplog.at_level(logging.INFO):
        assert response_text(SimpleNamespace(text="  plain text  "), "synthesis") == "plain text"
    assert caplog.records == []


def test_an_empty_answer_is_still_a_bad_output_error():
    with pytest.raises(LLMCallError) as caught:
        response_text(SimpleNamespace(text="   "), "synthesis")
    assert caught.value.kind is ErrorKind.BAD_OUTPUT

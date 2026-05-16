import pytest

pytest.importorskip("torch")

from evaluation.vqa_eval import VQAEvaluator


def _evaluator_without_model() -> VQAEvaluator:
    evaluator = object.__new__(VQAEvaluator)
    evaluator.strip_thinking = True
    return evaluator


def test_vqa_digit_word_equivalence_matches():
    evaluator = _evaluator_without_model()

    assert evaluator._process_answer("two cats") == evaluator._process_answer("2 cats")


def test_vqa_contraction_equivalence_matches():
    evaluator = _evaluator_without_model()

    assert evaluator._process_answer("don't know") == evaluator._process_answer("dont know")


def test_vqa_phrase_contraction_equivalence_matches():
    evaluator = _evaluator_without_model()

    assert evaluator._process_answer("do not know") == evaluator._process_answer("dont know")


def test_vqa_punctuation_stripping_matches():
    evaluator = _evaluator_without_model()

    assert evaluator._process_answer("yes.") == evaluator._process_answer("yes")

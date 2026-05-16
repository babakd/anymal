from evaluation.checkpoint_eval.chartqa_checkpoint_eval import chartqa_relaxed_match


def test_chartqa_relaxed_numeric_within_tolerance():
    assert chartqa_relaxed_match("24", "23.5")


def test_chartqa_relaxed_numeric_outside_tolerance():
    assert not chartqa_relaxed_match("25", "23.5")


def test_chartqa_relaxed_percent_stripped():
    assert chartqa_relaxed_match("23.5%", "23.5")


def test_chartqa_relaxed_currency_and_comma_stripped():
    assert chartqa_relaxed_match("$1,234", "1234")


def test_chartqa_relaxed_text_match():
    assert chartqa_relaxed_match("yes", "yes")


def test_chartqa_relaxed_text_mismatch():
    assert not chartqa_relaxed_match("red", "blue")


def test_chartqa_relaxed_spelled_out_number_known_limitation():
    assert not chartqa_relaxed_match("two", "2")

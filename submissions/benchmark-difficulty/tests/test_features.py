"""Tests for feature extraction module."""

import math
import pytest
from src.features import extract_features, extract_all_features, FEATURE_NAMES


SAMPLE_QUESTION = {
    "id": "test_1",
    "question": "Which of the following is an example of a chemical change?",
    "choices": ["freezing water", "cutting wood", "rusting iron", "melting butter"],
    "answer": 2,
    "difficulty": 0.6,
}

NEGATION_QUESTION = {
    "id": "test_neg",
    "question": "Which is NOT an example of a physical change?",
    "choices": ["boiling water", "breaking glass", "burning paper", "melting ice"],
    "answer": 2,
    "difficulty": 0.8,
}

SHORT_QUESTION = {
    "id": "test_short",
    "question": "What is water?",
    "choices": ["a gas", "a liquid", "a solid"],
    "answer": 1,
    "difficulty": 0.1,
}


def test_feature_names_defined():
    """FEATURE_NAMES lists all expected features."""
    assert len(FEATURE_NAMES) >= 10
    assert "question_length" in FEATURE_NAMES
    assert "answer_entropy" in FEATURE_NAMES
    assert "lexical_overlap" in FEATURE_NAMES
    assert "negation_count" in FEATURE_NAMES


def test_extract_features_returns_dict():
    """extract_features returns a dict with all feature names as keys."""
    features = extract_features(SAMPLE_QUESTION)
    assert isinstance(features, dict)
    for name in FEATURE_NAMES:
        assert name in features, f"Missing feature: {name}"


def test_feature_values_are_numeric():
    """All feature values are finite numbers."""
    features = extract_features(SAMPLE_QUESTION)
    for name, value in features.items():
        assert isinstance(value, (int, float)), \
            f"{name} is not numeric: {type(value)}"
        assert math.isfinite(value), f"{name} is not finite: {value}"


def test_question_length():
    """question_length is the character count of the question text."""
    features = extract_features(SAMPLE_QUESTION)
    assert features["question_length"] == len(SAMPLE_QUESTION["question"])


def test_word_count():
    """word_count is the number of words in the question."""
    features = extract_features(SAMPLE_QUESTION)
    assert features["word_count"] == len(SAMPLE_QUESTION["question"].split())


def test_negation_count_with_negation():
    """negation_count detects negation words."""
    features = extract_features(NEGATION_QUESTION)
    assert features["negation_count"] >= 1


def test_negation_count_without_negation():
    """negation_count is 0 when no negation words are present."""
    features = extract_features(SAMPLE_QUESTION)
    assert features["negation_count"] == 0


def test_num_choices():
    """num_choices matches the number of answer options."""
    features_4 = extract_features(SAMPLE_QUESTION)
    features_3 = extract_features(SHORT_QUESTION)
    assert features_4["num_choices"] == 4
    assert features_3["num_choices"] == 3


def test_answer_entropy_nonnegative():
    """Answer entropy is non-negative."""
    features = extract_features(SAMPLE_QUESTION)
    assert features["answer_entropy"] >= 0.0


def test_lexical_overlap_range():
    """Lexical overlap (Jaccard) is in [0, 1]."""
    features = extract_features(SAMPLE_QUESTION)
    assert 0.0 <= features["lexical_overlap"] <= 1.0


def test_unique_word_ratio_range():
    """Unique word ratio is in (0, 1]."""
    features = extract_features(SAMPLE_QUESTION)
    assert 0.0 < features["unique_word_ratio"] <= 1.0


def test_max_option_length_ratio_positive():
    """max_option_length_ratio is >= 1.0."""
    features = extract_features(SAMPLE_QUESTION)
    assert features["max_option_length_ratio"] >= 1.0


def test_flesch_kincaid_grade():
    """flesch_kincaid_grade is a reasonable value."""
    features = extract_features(SAMPLE_QUESTION)
    # Grade level should be between -5 and 30 for any text
    assert -5.0 <= features["flesch_kincaid_grade"] <= 30.0


def test_extract_all_features():
    """extract_all_features processes a list of questions."""
    questions = [SAMPLE_QUESTION, NEGATION_QUESTION, SHORT_QUESTION]
    all_features = extract_all_features(questions)
    assert len(all_features) == 3
    for f in all_features:
        assert isinstance(f, dict)
        for name in FEATURE_NAMES:
            assert name in f


def test_avg_word_length():
    """avg_word_length is reasonable."""
    features = extract_features(SAMPLE_QUESTION)
    assert 1.0 <= features["avg_word_length"] <= 20.0


def test_stem_overlap_range():
    """stem_overlap is in [0, 1]."""
    features = extract_features(SAMPLE_QUESTION)
    assert 0.0 <= features["stem_overlap"] <= 1.0

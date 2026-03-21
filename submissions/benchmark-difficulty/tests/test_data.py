"""Tests for data loading module."""

import pytest
from src.data import load_arc_with_difficulty, HARDCODED_ARC_SAMPLE


def test_hardcoded_sample_exists():
    """Hardcoded sample has at least 50 questions."""
    assert len(HARDCODED_ARC_SAMPLE) >= 50


def test_hardcoded_sample_schema():
    """Each hardcoded question has required keys."""
    required_keys = {"question", "choices", "answer", "difficulty", "id"}
    for q in HARDCODED_ARC_SAMPLE:
        assert required_keys.issubset(q.keys()), f"Missing keys in {q.get('id', '?')}"


def test_hardcoded_difficulty_range():
    """Difficulty values are in [0, 1]."""
    for q in HARDCODED_ARC_SAMPLE:
        assert 0.0 <= q["difficulty"] <= 1.0, f"Bad difficulty {q['difficulty']} for {q['id']}"


def test_hardcoded_choices_format():
    """Choices are a list of strings with 3-5 elements."""
    for q in HARDCODED_ARC_SAMPLE:
        assert isinstance(q["choices"], list), f"Choices not a list for {q['id']}"
        assert 3 <= len(q["choices"]) <= 5, f"Bad choice count for {q['id']}"
        for c in q["choices"]:
            assert isinstance(c, str), f"Choice not string in {q['id']}"


def test_load_arc_returns_list():
    """load_arc_with_difficulty returns a non-empty list."""
    data = load_arc_with_difficulty(use_hardcoded=True)
    assert isinstance(data, list)
    assert len(data) >= 50


def test_load_arc_schema():
    """Each loaded question has required keys."""
    data = load_arc_with_difficulty(use_hardcoded=True)
    required_keys = {"question", "choices", "answer", "difficulty", "id"}
    for q in data:
        assert required_keys.issubset(q.keys())


def test_load_arc_difficulty_variance():
    """Difficulty values have non-trivial variance (not all the same)."""
    data = load_arc_with_difficulty(use_hardcoded=True)
    difficulties = [q["difficulty"] for q in data]
    variance = sum((d - sum(difficulties) / len(difficulties)) ** 2
                    for d in difficulties) / len(difficulties)
    assert variance > 0.01, "Difficulty values have too little variance"


def test_load_arc_answer_valid():
    """Each answer is one of the choice indices."""
    data = load_arc_with_difficulty(use_hardcoded=True)
    for q in data:
        assert isinstance(q["answer"], int)
        assert 0 <= q["answer"] < len(q["choices"]), \
            f"Answer index {q['answer']} out of range for {q['id']}"

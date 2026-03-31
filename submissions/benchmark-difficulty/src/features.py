"""Feature extraction for benchmark difficulty prediction.

Extracts structural and information-theoretic features from multiple-choice
questions without running any LLM. Features are designed to capture
surface-level indicators of question difficulty.

References:
  - Ethayarajh & Choi. "Understanding Dataset Difficulty with
    V-Usable Information." NeurIPS 2022.
  - Wang et al. "An Entropy-Driven Method for LLM Dataset Evaluation
    and Optimization." 2025.
"""

import math
import re
from typing import Any


# Negation words that may increase question difficulty
_NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "none", "nobody", "nothing",
    "nowhere", "hardly", "barely", "scarcely", "except", "without",
    "cannot", "can't", "don't", "doesn't", "didn't", "won't", "wouldn't",
    "shouldn't", "couldn't", "isn't", "aren't", "wasn't", "weren't",
}

# Question type keywords
_QUESTION_TYPES = {
    "what": 0, "which": 1, "how": 2, "why": 3, "when": 4, "where": 5,
    "who": 6,
}

# All feature names in consistent order
FEATURE_NAMES = [
    "question_length",
    "word_count",
    "avg_word_length",
    "answer_entropy",
    "num_choices",
    "lexical_overlap",
    "negation_count",
    "question_type",
    "flesch_kincaid_grade",
    "unique_word_ratio",
    "max_option_length_ratio",
    "stem_overlap",
]


def _count_syllables(word: str) -> int:
    """Estimate syllable count for a word using a simple heuristic.

    Uses the vowel-cluster method: count groups of consecutive vowels,
    with adjustments for silent-e and common patterns.
    """
    word = word.lower().strip()
    if not word:
        return 1
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    # Adjust for silent e at end
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def _flesch_kincaid_grade(text: str) -> float:
    """Compute Flesch-Kincaid grade level for a text.

    FK = 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
    """
    sentences = max(len(re.split(r'[.!?]+', text.strip())), 1)
    words = text.split()
    num_words = max(len(words), 1)
    total_syllables = sum(_count_syllables(w) for w in words)

    grade = (0.39 * (num_words / sentences)
             + 11.8 * (total_syllables / num_words)
             - 15.59)
    return grade


def _shannon_entropy(lengths: list[int]) -> float:
    """Compute Shannon entropy over a list of integer values.

    Treats the values as a probability distribution (normalized).
    Higher entropy means more uniform distribution.
    """
    total = sum(lengths)
    if total == 0:
        return 0.0
    entropy = 0.0
    for length in lengths:
        if length > 0:
            p = length / total
            entropy -= p * math.log2(p)
    return entropy


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def _tokenize(text: str) -> list[str]:
    """Simple word tokenization: lowercase, split on non-alphanumeric."""
    return re.findall(r'[a-zA-Z]+', text.lower())


def extract_features(question: dict[str, Any]) -> dict[str, float]:
    """Extract structural features from a single multiple-choice question.

    Args:
        question: Dict with keys 'question', 'choices', 'answer'.

    Returns:
        Dict mapping feature names to numeric values.
    """
    q_text = question["question"]
    choices = question["choices"]
    answer_idx = question["answer"]

    q_words = _tokenize(q_text)
    num_words = max(len(q_words), 1)

    # Basic length features
    question_length = len(q_text)
    word_count = len(q_text.split())
    avg_word_length = (sum(len(w) for w in q_words) / num_words
                       if q_words else 0.0)

    # Answer entropy: Shannon entropy over answer option character lengths
    option_lengths = [len(c) for c in choices]
    answer_entropy = _shannon_entropy(option_lengths)

    # Number of choices
    num_choices = len(choices)

    # Lexical overlap: Jaccard similarity between question words and
    # all answer option words combined
    q_word_set = set(q_words)
    all_choice_words = set()
    for c in choices:
        all_choice_words.update(_tokenize(c))
    lexical_overlap = _jaccard_similarity(q_word_set, all_choice_words)

    # Negation count
    negation_count = sum(1 for w in q_words if w in _NEGATION_WORDS)

    # Question type (encoded as integer)
    question_type = 7  # default: other
    first_word = q_words[0] if q_words else ""
    if first_word in _QUESTION_TYPES:
        question_type = _QUESTION_TYPES[first_word]

    # Flesch-Kincaid grade level
    fk_grade = _flesch_kincaid_grade(q_text)

    # Unique word ratio
    unique_word_ratio = len(set(q_words)) / num_words if q_words else 1.0

    # Max option length ratio (longest / shortest)
    if option_lengths:
        min_len = max(min(option_lengths), 1)
        max_len = max(option_lengths)
        max_option_length_ratio = max_len / min_len
    else:
        max_option_length_ratio = 1.0

    # Stem overlap: Jaccard similarity between question words and
    # correct answer words only
    correct_answer = choices[answer_idx] if 0 <= answer_idx < len(choices) else ""
    correct_words = set(_tokenize(correct_answer))
    stem_overlap = _jaccard_similarity(q_word_set, correct_words)

    return {
        "question_length": question_length,
        "word_count": word_count,
        "avg_word_length": avg_word_length,
        "answer_entropy": answer_entropy,
        "num_choices": num_choices,
        "lexical_overlap": lexical_overlap,
        "negation_count": negation_count,
        "question_type": question_type,
        "flesch_kincaid_grade": fk_grade,
        "unique_word_ratio": unique_word_ratio,
        "max_option_length_ratio": max_option_length_ratio,
        "stem_overlap": stem_overlap,
    }


def extract_all_features(questions: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Extract features from a list of questions.

    Args:
        questions: List of question dicts.

    Returns:
        List of feature dicts, one per question.
    """
    return [extract_features(q) for q in questions]

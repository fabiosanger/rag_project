"""
Text processing utilities for the RAG system.
"""

import re
from typing import List, Optional


def clean_answer(answer: str) -> str:
    """
    Clean up generated answer by removing duplicates and extra whitespace.

    Args:
        answer: The raw answer string to clean

    Returns:
        Cleaned answer string
    """
    if not answer:
        return ""

    # Remove duplicate consecutive words (case-insensitive)
    words = answer.split()
    cleaned_words = []
    for i, word in enumerate(words):
        if i == 0 or word.lower() != words[i-1].lower():
            cleaned_words.append(word)

    cleaned = ' '.join(cleaned_words)

    # Capitalize the first letter
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned


def normalize_text(text: str) -> str:
    """
    Normalize text by removing extra whitespace and standardizing formatting.

    Args:
        text: Input text to normalize

    Returns:
        Normalized text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\.\,\!\?\-\'\"]', '', text)

    return text


def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """
    Extract keywords from text using simple frequency-based approach.

    Args:
        text: Input text to extract keywords from
        max_keywords: Maximum number of keywords to return

    Returns:
        List of keywords
    """
    # Simple keyword extraction - in a real system, you might use more sophisticated methods
    words = re.findall(r'\b\w+\b', text.lower())

    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    words = [word for word in words if word not in stop_words and len(word) > 2]

    # Count frequency
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1

    # Sort by frequency and return top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:max_keywords]]


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Input text to truncate
        max_length: Maximum length of the text
        suffix: Suffix to add if text is truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix
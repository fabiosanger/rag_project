"""
Data validation utilities for the RAG system.
"""

from typing import List, Dict, Tuple, Optional


def validate_qa_data(data: List[Dict[str, str]]) -> Tuple[bool, List[str]]:
    """
    Validate question-answer data for required format and content.

    Args:
        data: List of dictionaries containing question-answer pairs

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    if not isinstance(data, list):
        errors.append("Data must be a list")
        return False, errors

    if not data:
        errors.append("Data list cannot be empty")
        return False, errors

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"Item {i} must be a dictionary")
            continue

        # Check for required keys
        if "question" not in item:
            errors.append(f"Item {i} missing 'question' key")
        elif not isinstance(item["question"], str) or not item["question"].strip():
            errors.append(f"Item {i} has invalid 'question' value")

        if "answer" not in item:
            errors.append(f"Item {i} missing 'answer' key")
        elif not isinstance(item["answer"], str) or not item["answer"].strip():
            errors.append(f"Item {i} has invalid 'answer' value")

    return len(errors) == 0, errors


def validate_question(question: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a single question.

    Args:
        question: Question string to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(question, str):
        return False, "Question must be a string"

    if not question.strip():
        return False, "Question cannot be empty"

    if len(question.strip()) < 3:
        return False, "Question must be at least 3 characters long"

    return True, None


def validate_answer(answer: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a single answer.

    Args:
        answer: Answer string to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(answer, str):
        return False, "Answer must be a string"

    if not answer.strip():
        return False, "Answer cannot be empty"

    if len(answer.strip()) < 5:
        return False, "Answer must be at least 5 characters long"

    return True, None


def sanitize_qa_data(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Sanitize question-answer data by cleaning and normalizing text.

    Args:
        data: Raw question-answer data

    Returns:
        Sanitized data
    """
    from .text_processing import normalize_text

    sanitized_data = []

    for item in data:
        if isinstance(item, dict) and "question" in item and "answer" in item:
            sanitized_item = {
                "question": normalize_text(str(item["question"])),
                "answer": normalize_text(str(item["answer"]))
            }

            # Only add if both question and answer are not empty after sanitization
            if sanitized_item["question"] and sanitized_item["answer"]:
                sanitized_data.append(sanitized_item)

    return sanitized_data
"""
Data loading utilities for the RAG system.
"""

import json
import csv
from typing import List, Dict, Optional
from pathlib import Path


def load_sample_data() -> List[Dict[str, str]]:
    """
    Load sample data for demonstration purposes.

    Returns:
        List of dictionaries containing question-answer pairs
    """
    return [
        {"question": "What is the capital of France?", "answer": "The capital of France is Paris."},
        {"question": "What is the largest planet?", "answer": "The largest planet is Jupiter."},
        {"question": "Who wrote '1984'?", "answer": "George Orwell wrote '1984'."},
        {"question": "When EPAM was established?", "answer": "EPAM was established in 1993."},
        {"question": "What is the speed of light?", "answer": "The speed of light is approximately 299,792,458 meters per second."},
        {"question": "Who is the current president of the United States?", "answer": "As of 2024, Joe Biden is the current president of the United States."},
    ]


def load_data_from_file(file_path: str, file_format: str = "auto") -> List[Dict[str, str]]:
    """
    Load question-answer data from a file.

    Args:
        file_path: Path to the data file
        file_format: Format of the file ('json', 'csv', or 'auto' for auto-detection)

    Returns:
        List of dictionaries containing question-answer pairs

    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file does not exist
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Auto-detect format based on file extension
    if file_format == "auto":
        if file_path.suffix.lower() == ".json":
            file_format = "json"
        elif file_path.suffix.lower() == ".csv":
            file_format = "csv"
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    if file_format == "json":
        return _load_json_data(file_path)
    elif file_format == "csv":
        return _load_csv_data(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def _load_json_data(file_path: Path) -> List[Dict[str, str]]:
    """Load data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle different JSON structures
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "data" in data:
        return data["data"]
    else:
        raise ValueError("Invalid JSON structure. Expected list or dict with 'data' key.")


def _load_csv_data(file_path: Path) -> List[Dict[str, str]]:
    """Load data from CSV file."""
    data = []

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        # Check if required columns exist
        if "question" not in reader.fieldnames or "answer" not in reader.fieldnames:
            raise ValueError("CSV file must contain 'question' and 'answer' columns")

        for row in reader:
            data.append({
                "question": row["question"].strip(),
                "answer": row["answer"].strip()
            })

    return data


def save_data_to_file(data: List[Dict[str, str]], file_path: str, file_format: str = "json") -> None:
    """
    Save question-answer data to a file.

    Args:
        data: List of dictionaries containing question-answer pairs
        file_path: Path to save the data file
        file_format: Format of the file ('json' or 'csv')

    Raises:
        ValueError: If file format is not supported
    """
    file_path = Path(file_path)

    if file_format == "json":
        _save_json_data(data, file_path)
    elif file_format == "csv":
        _save_csv_data(data, file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def _save_json_data(data: List[Dict[str, str]], file_path: Path) -> None:
    """Save data to JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _save_csv_data(data: List[Dict[str, str]], file_path: Path) -> None:
    """Save data to CSV file."""
    with open(file_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer"])
        writer.writeheader()
        writer.writerows(data)
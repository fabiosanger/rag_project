import pytest
import tempfile
import json
import csv
from pathlib import Path
import torch
from unittest.mock import patch, Mock
from rag_project.utils.text_processing import clean_answer, normalize_text, extract_keywords, truncate_text
from rag_project.utils.data_loader import (
    load_sample_data, load_data_from_file, save_data_to_file,
    _load_json_data, _load_csv_data
)
from rag_project.utils.validation import (
    validate_qa_data, validate_question, validate_answer, sanitize_qa_data
)
from rag_project.utils.gpu_utils import (
    get_gpu_device, is_gpu_available, get_gpu_info, clear_gpu_cache,
    get_optimal_batch_size, move_to_device, get_device_info
)


class TestTextProcessing:
    """Test cases for text processing utilities"""

    def test_clean_answer(self):
        """Test answer cleaning functionality"""
        # Test duplicate word removal
        result = clean_answer("the the capital is Paris")
        assert result == "The capital is Paris"

        # Test capitalization
        result = clean_answer("paris is beautiful")
        assert result == "Paris is beautiful"

        # Test empty string
        result = clean_answer("")
        assert result == ""

        # Test single word
        result = clean_answer("hello")
        assert result == "Hello"

    def test_normalize_text(self):
        """Test text normalization"""
        # Test extra whitespace removal
        result = normalize_text("  hello   world  ")
        assert result == "hello world"

        # Test special character removal
        result = normalize_text("hello@world#$%")
        assert result == "hello world"

        # Test empty string
        result = normalize_text("")
        assert result == ""

    def test_extract_keywords(self):
        """Test keyword extraction"""
        text = "The quick brown fox jumps over the lazy dog"
        keywords = extract_keywords(text, max_keywords=3)

        assert len(keywords) <= 3
        assert all(isinstance(k, str) for k in keywords)
        assert "quick" in keywords or "brown" in keywords or "fox" in keywords

    def test_truncate_text(self):
        """Test text truncation"""
        # Test truncation
        result = truncate_text("This is a very long text that needs to be truncated", max_length=20)
        assert len(result) <= 20
        assert result.endswith("...")

        # Test no truncation needed
        result = truncate_text("Short text", max_length=20)
        assert result == "Short text"


class TestDataLoader:
    """Test cases for data loading utilities"""

    def test_load_sample_data(self):
        """Test sample data loading"""
        data = load_sample_data()

        assert isinstance(data, list)
        assert len(data) > 0
        assert all(isinstance(item, dict) for item in data)
        assert all("question" in item and "answer" in item for item in data)

    def test_load_json_data(self, temp_dir):
        """Test JSON data loading"""
        # Create test JSON file
        test_data = [
            {"question": "Test Q1?", "answer": "Test A1"},
            {"question": "Test Q2?", "answer": "Test A2"}
        ]

        json_file = Path(temp_dir) / "test.json"
        with open(json_file, 'w') as f:
            json.dump(test_data, f)

        # Test loading
        loaded_data = _load_json_data(json_file)
        assert loaded_data == test_data

    def test_load_csv_data(self, temp_dir):
        """Test CSV data loading"""
        # Create test CSV file
        csv_file = Path(temp_dir) / "test.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["question", "answer"])
            writer.writerow(["Test Q1?", "Test A1"])
            writer.writerow(["Test Q2?", "Test A2"])

        # Test loading
        loaded_data = _load_csv_data(csv_file)
        assert len(loaded_data) == 2
        assert loaded_data[0]["question"] == "Test Q1?"
        assert loaded_data[0]["answer"] == "Test A1"

    def test_save_data_to_file(self, temp_dir):
        """Test data saving"""
        test_data = [
            {"question": "Test Q1?", "answer": "Test A1"},
            {"question": "Test Q2?", "answer": "Test A2"}
        ]

        # Test JSON saving
        json_file = Path(temp_dir) / "output.json"
        save_data_to_file(test_data, str(json_file), "json")
        assert json_file.exists()

        # Test CSV saving
        csv_file = Path(temp_dir) / "output.csv"
        save_data_to_file(test_data, str(csv_file), "csv")
        assert csv_file.exists()


class TestValidation:
    """Test cases for validation utilities"""

    def test_validate_qa_data(self):
        """Test QA data validation"""
        # Valid data
        valid_data = [
            {"question": "Test Q1?", "answer": "Test A1"},
            {"question": "Test Q2?", "answer": "Test A2"}
        ]
        is_valid, errors = validate_qa_data(valid_data)
        assert is_valid
        assert len(errors) == 0

        # Invalid data
        invalid_data = [
            {"question": "", "answer": "Test A1"},  # Empty question
            {"question": "Test Q2?"}  # Missing answer
        ]
        is_valid, errors = validate_qa_data(invalid_data)
        assert not is_valid
        assert len(errors) > 0

    def test_validate_question(self):
        """Test question validation"""
        # Valid question
        is_valid, error = validate_question("What is the capital of France?")
        assert is_valid
        assert error is None

        # Invalid questions
        is_valid, error = validate_question("")
        assert not is_valid
        assert error is not None

        is_valid, error = validate_question("Hi")
        assert not is_valid
        assert error is not None

    def test_validate_answer(self):
        """Test answer validation"""
        # Valid answer
        is_valid, error = validate_answer("The capital of France is Paris.")
        assert is_valid
        assert error is None

        # Invalid answers
        is_valid, error = validate_answer("")
        assert not is_valid
        assert error is not None

        is_valid, error = validate_answer("Hi")
        assert not is_valid
        assert error is not None

    def test_sanitize_qa_data(self):
        """Test data sanitization"""
        raw_data = [
            {"question": "  Test Q1?  ", "answer": "  Test A1  "},
            {"question": "Test Q2?", "answer": "Test A2"},
            {"question": "", "answer": "Test A3"},  # Invalid
            {"question": "Test Q4?", "answer": ""}  # Invalid
        ]

        sanitized = sanitize_qa_data(raw_data)
        assert len(sanitized) == 2  # Only valid entries
        assert sanitized[0]["question"] == "Test Q1?"
        assert sanitized[0]["answer"] == "Test A1"
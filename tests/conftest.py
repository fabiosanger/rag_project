"""
Pytest configuration and shared fixtures.
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path

# Add src to Python path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture(autouse=True)
def set_tokenizers_parallelism():
    """Set tokenizers parallelism to false for all tests"""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    yield
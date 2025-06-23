"""
Utility modules for the RAG project.

This package contains helper functions and utilities used throughout the RAG system.
"""

from .text_processing import clean_answer
from .data_loader import load_sample_data, load_data_from_file
from .validation import validate_qa_data
from .gpu_utils import (
    get_gpu_device, is_gpu_available, get_gpu_info, clear_gpu_cache,
    get_optimal_batch_size, move_to_device, get_device_info, GPUManager
)

__all__ = [
    "clean_answer",
    "load_sample_data",
    "load_data_from_file",
    "validate_qa_data",
    "get_gpu_device",
    "is_gpu_available",
    "get_gpu_info",
    "clear_gpu_cache",
    "get_optimal_batch_size",
    "move_to_device",
    "get_device_info",
    "GPUManager"
]
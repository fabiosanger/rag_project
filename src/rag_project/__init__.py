"""
RAG Project - A simple question-answering system using T5 and sentence transformers.

This package provides a Retrieval-Augmented Generation (RAG) system that combines
semantic search with transformer-based text generation for question answering.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .qa_system import SimpleQASystem

__all__ = ["SimpleQASystem"]
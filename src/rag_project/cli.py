#!/usr/bin/env python3
"""
Command-line interface for the RAG QA system.
"""

import sys
import argparse
from typing import List, Dict
from pathlib import Path

from .qa_system import SimpleQASystem
from .utils.data_loader import load_sample_data, load_data_from_file
from .utils.validation import validate_qa_data, sanitize_qa_data


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="RAG QA System - Question answering using T5 and sentence transformers"
    )

    parser.add_argument(
        "--data-file",
        type=str,
        help="Path to data file (JSON or CSV format)"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        default=True,
        help="Run in interactive mode (default)"
    )

    parser.add_argument(
        "--question",
        type=str,
        help="Single question to answer (non-interactive mode)"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate data before processing"
    )

    return parser


def main():
    """Main function with command line interface."""
    parser = create_parser()
    args = parser.parse_args()

    try:
        # Load data
        if args.data_file:
            print(f"Loading data from {args.data_file}...")
            data = load_data_from_file(args.data_file)
        else:
            print("Using sample data...")
            data = load_sample_data()

        # Validate data if requested
        if args.validate:
            is_valid, errors = validate_qa_data(data)
            if not is_valid:
                print("Data validation failed:")
                for error in errors:
                    print(f"  - {error}")
                sys.exit(1)
            print("Data validation passed!")

        # Sanitize data
        data = sanitize_qa_data(data)

        # Initialize system
        print("Initializing QA system...")
        qa_system = SimpleQASystem()

        # Prepare dataset
        print("Preparing dataset...")
        qa_system.prepare_dataset(data)

        # Handle single question or interactive mode
        if args.question:
            # Non-interactive mode
            print(f"\nQuestion: {args.question}")
            answer = qa_system.get_answer(args.question)
            print(f"Answer: {answer}")
        else:
            # Interactive mode
            print("\nInteractive Q&A session started. Type 'exit' to quit.")
            while True:
                try:
                    test_question = input("\nPlease enter your question: ")

                    if test_question.lower() in ['exit', 'quit', 'q']:
                        print("Exiting the program.")
                        break

                    if not test_question.strip():
                        continue

                    print(f"\nQuestion: {test_question}")
                    answer = qa_system.get_answer(test_question)
                    print(f"Answer: {answer}")

                except KeyboardInterrupt:
                    print("\nExiting the program.")
                    break
                except Exception as e:
                    print(f"Error processing question: {e}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
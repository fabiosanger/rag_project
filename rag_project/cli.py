#!/usr/bin/env python3
"""
Command-line interface for the RAG QA system.
"""

import sys
from typing import List, Dict
from .qa_system import SimpleQASystem


def load_sample_data() -> List[Dict[str, str]]:
    """Load sample data for demonstration"""
    return [
        {"question": "What is the capital of France?", "answer": "The capital of France is Paris."},
        {"question": "What is the largest planet?", "answer": "The largest planet is Jupiter."},
        {"question": "Who wrote '1984'?", "answer": "George Orwell wrote '1984'."},
        {"question": "When EPAM was established?", "answer": "EPAM was established in 1993'."}
    ]


def main():
    """Main function with sample usage"""
    try:
        # Sample data
        data = load_sample_data()

        # Initialize system
        print("Initializing QA system...")
        qa_system = SimpleQASystem()

        # Prepare dataset
        print("Preparing dataset...")
        qa_system.prepare_dataset(data)

        # Start interactive Q&A session
        while True:
            # Prompt the user for a question
            test_question = input("\nPlease enter your question (or 'exit' to quit): ")

            if test_question.lower() == 'exit':
                print("Exiting the program.")
                break

            # Get and print the answer
            print(f"\nQuestion: {test_question}")
            answer = qa_system.get_answer(test_question)
            print(f"Answer: {answer}")

    except Exception as e:
        print(f"Error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
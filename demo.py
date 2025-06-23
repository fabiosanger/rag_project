#!/usr/bin/env python3
"""
Demo script for the RAG QA System
Tests the core functionality before running the Streamlit interface
"""

import json
from rag_project.qa_system import SimpleQASystem

def load_sample_data():
    """Load sample data for testing"""
    return [
        {"question": "What is machine learning?", "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed."},
        {"question": "How does neural network work?", "answer": "Neural networks are computing systems inspired by biological brains, consisting of interconnected nodes that process information through weighted connections."},
        {"question": "What is deep learning?", "answer": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data."},
        {"question": "What is natural language processing?", "answer": "Natural language processing (NLP) is a field of AI that focuses on enabling computers to understand, interpret, and generate human language."},
        {"question": "What is computer vision?", "answer": "Computer vision is a field of AI that enables computers to interpret and understand visual information from the world, such as images and videos."},
        {"question": "What is the capital of France?", "answer": "The capital of France is Paris."},
        {"question": "What is the largest planet?", "answer": "The largest planet is Jupiter."},
        {"question": "What is the capital of Germany?", "answer": "The capital of Germany is Berlin."},
        {"question": "What is the largest country in the world?", "answer": "The largest country in the world is Russia."},
        {"question": "What is the capital of Japan?", "answer": "The capital of Japan is Tokyo."},
        {"question": "What is the largest city in the world?", "answer": "The largest city in the world is Tokyo."},
        {"question": "What is the capital of China?", "answer": "The capital of China is Beijing."},
        {"question": "What is the largest country in Europe?", "answer": "The largest country in Europe is Russia."},
    ]

def main():
    print("🤖 RAG QA System Demo")
    print("=" * 50)

    # Initialize the QA system
    print("📦 Initializing QA system...")
    qa_system = SimpleQASystem()

    # Load sample data
    print("📚 Loading sample data...")
    sample_data = load_sample_data()
    qa_system.prepare_dataset(sample_data)

    print(f"✅ Loaded {len(sample_data)} Q&A pairs")
    print(f"🖥️  Using device: {qa_system.device}")
    print()

    # Test questions
    test_questions = [
        "What is machine learning?",
        "How do neural networks function?",
        "Explain deep learning",
        "What does NLP stand for?",
        "Tell me about computer vision",
        "What is the capital of France?",
        "What is the largest planet?",
        "What is the capital of Germany?",
        "What is the largest country in the world?",
        "What is the capital of Japan?",
        "What is the largest city in the world?",
        "What is the capital of China?",
    ]

    print("🧪 Testing the system...")
    print("-" * 50)

    for i, question in enumerate(test_questions, 1):
        print(f"\n❓ Question {i}: {question}")

        # Get answer with confidence
        answer, confidence, similarities = qa_system.get_answer_with_confidence(question)

        print(f"📝 Answer: {answer}")
        print(f"🎯 Confidence: {confidence:.1f}%")

        # Show top 3 similarities
        top_indices = similarities.argsort()[-3:][::-1]
        print("🔍 Top 3 similar answers:")
        for j, idx in enumerate(top_indices, 1):
            similarity_score = similarities[idx] * 100
            print(f"   {j}. {qa_system.answers[idx][:60]}... ({similarity_score:.1f}%)")

    print("\n" + "=" * 50)
    print("✅ Demo completed successfully!")
    print("🚀 You can now run the Streamlit interface with:")
    print("   streamlit run streamlit_app.py")
    print("   or")
    print("   ./run_streamlit.sh")

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import json
from rag_project.qa_system import SimpleQASystem
import torch
import torch.nn.functional as F
import time

# Clear cache to force reload
st.cache_resource.clear()

# Page configuration
st.set_page_config(
    page_title="RAG QA System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .confidence-score {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .high-confidence {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .medium-confidence {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    .low-confidence {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .answer-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_qa_system():
    """Load the QA system with caching to avoid reloading"""
    with st.spinner("Loading QA system..."):
        qa_system = SimpleQASystem()
        print(f"DEBUG: Model name from QA system: {qa_system.model_name}")
        return qa_system

def calculate_confidence_score(similarities, best_idx):
    """Calculate confidence score based on similarity scores"""
    best_score = similarities[best_idx]
    max_possible = 1.0  # Cosine similarity ranges from -1 to 1, but embeddings are typically positive

    # Normalize to 0-1 range and convert to percentage
    confidence = (best_score / max_possible) * 100

    # Apply some smoothing and ensure reasonable bounds
    confidence = max(0, min(100, confidence))

    return confidence

def get_confidence_category(confidence):
    """Get confidence category for styling"""
    if confidence >= 70:
        return "high-confidence"
    elif confidence >= 40:
        return "medium-confidence"
    else:
        return "low-confidence"

def load_sample_data():
    """Load sample data for demonstration"""
    try:
        with open('sample_data.json', 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
        return sample_data
    except FileNotFoundError:
        st.error("❌ sample_data.json file not found!")
        # Fallback to minimal data
        return [
            {"question": "What is machine learning?", "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed."},
            {"question": "What is the capital of France?", "answer": "The capital of France is Paris."},
        ]
    except json.JSONDecodeError as e:
        st.error(f"❌ Error parsing sample_data.json: {e}")
        # Fallback to minimal data
        return [
            {"question": "What is machine learning?", "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed."},
            {"question": "What is the capital of France?", "answer": "The capital of France is Paris."},
        ]

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 RAG QA System</h1>', unsafe_allow_html=True)

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Device selection
        device_options = ["Auto-detect", "CPU", "GPU"]
        selected_device = st.selectbox("Device", device_options, index=0)

        # Model parameters
        st.subheader("Model Parameters")
        max_length = st.slider("Max Answer Length", 20, 100, 50)
        num_beams = st.slider("Number of Beams", 1, 8, 4)

        # Confidence threshold
        confidence_threshold = st.slider("Confidence Threshold (%)", 0, 100, 40)

        st.divider()

        # Data upload section
        st.subheader("📁 Data Management")

        # Option to use sample data or upload custom data
        data_source = st.radio("Data Source", ["Sample Data", "Upload Custom Data"])

        if data_source == "Sample Data":
            sample_data = load_sample_data()
            st.success("✅ Using sample data")
        else:
            uploaded_file = st.file_uploader(
                "Upload JSON file with Q&A pairs",
                type=['json'],
                help="File should contain a list of dictionaries with 'question' and 'answer' keys"
            )

            if uploaded_file is not None:
                try:
                    sample_data = json.load(uploaded_file)
                    st.success(f"✅ Loaded {len(sample_data)} Q&A pairs")
                except Exception as e:
                    st.error(f"❌ Error loading file: {e}")
                    sample_data = load_sample_data()
            else:
                sample_data = load_sample_data()
                st.info("📋 Please upload a JSON file or use sample data")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Ask a Question")

        # Question input
        question = st.text_area(
            "Enter your question:",
            placeholder="Type your question here...",
            height=100
        )

        # Submit button
        if st.button("🚀 Get Answer", type="primary", use_container_width=True):
            if question.strip():
                with st.spinner("Processing your question..."):
                    try:
                        # Load QA system
                        qa_system = load_qa_system()

                        # Prepare dataset if not already done
                        if not qa_system.answers:
                            qa_system.prepare_dataset(sample_data)

                        # Get answer and calculate confidence
                        start_time = time.time()

                        # Modified get_answer to return confidence
                        answer, confidence, similarities = get_answer_with_confidence(
                            qa_system, question, max_length, num_beams
                        )

                        processing_time = time.time() - start_time

                        # Display results
                        st.subheader("📝 Answer")

                        # Confidence score with color coding
                        confidence_class = get_confidence_category(confidence)
                        st.markdown(
                            f'<div class="confidence-score {confidence_class}">'
                            f'Confidence: {confidence:.1f}%</div>',
                            unsafe_allow_html=True
                        )

                        # Answer display
                        if confidence >= confidence_threshold:
                            st.markdown(
                                f'<div class="answer-box">{answer}</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.warning(f"⚠️ Confidence ({confidence:.1f}%) is below threshold ({confidence_threshold}%). Answer may not be reliable.")
                            st.markdown(
                                f'<div class="answer-box">{answer}</div>',
                                unsafe_allow_html=True
                            )

                        # Processing time
                        st.caption(f"⏱️ Processing time: {processing_time:.2f} seconds")

                        print(f"DEBUG: Answer returned: '{answer}' (confidence: {confidence})")

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Please enter a question")

    with col2:
        st.header("📊 Statistics")

        if 'qa_system' in locals():
            try:
                qa_system = load_qa_system()

                # System info
                st.metric("Total Answers", len(qa_system.answers) if qa_system.answers else 0)
                st.metric("Device", str(qa_system.device))

                # Extract just the model name from the path
                model_display_name = qa_system.model_name.split('/')[-1] if '/' in qa_system.model_name else qa_system.model_name
                st.metric("Model", model_display_name)

                # Recent questions (if we had a session state)
                if 'recent_questions' not in st.session_state:
                    st.session_state.recent_questions = []

                if st.session_state.recent_questions:
                    st.subheader("🕒 Recent Questions")
                    for q in st.session_state.recent_questions[-5:]:
                        st.text(f"• {q[:50]}...")

            except Exception as e:
                st.error(f"Error loading statistics: {e}")
        else:
            st.info("Start by asking a question to see statistics")

    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        Built with ❤️ using Streamlit and PyTorch
        </div>
        """,
        unsafe_allow_html=True
    )

def get_answer_with_confidence(qa_system, question, max_length, num_beams):
    """Modified version of get_answer that returns confidence score"""
    try:
        if not qa_system.answers or qa_system.answer_embeddings is None:
            raise ValueError("Dataset not prepared. Call prepare_dataset first.")

        # Encode question using SentenceTransformer
        question_embedding = qa_system.encoder.encode(
            question,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        # Move to the same device as answer embeddings
        question_embedding = question_embedding.to(qa_system.device)

        # Stack answer embeddings into a single tensor
        answer_embeddings_tensor = torch.stack(qa_system.answer_embeddings).to(qa_system.device)

        # Calculate cosine similarity using PyTorch
        similarities = F.cosine_similarity(
            question_embedding.unsqueeze(0),
            answer_embeddings_tensor,
            dim=1
        )

        best_idx = similarities.argmax().item()
        confidence = calculate_confidence_score(similarities, best_idx)
        context = qa_system.answers[best_idx]

        print(f"DEBUG: Question: {question}")
        print(f"DEBUG: Context: {context}")
        print(f"DEBUG: Using context directly as answer")

        # Return the context directly instead of generating with T5
        return context, confidence, similarities.cpu().numpy()

    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"Error: {str(e)}", 0.0, []

if __name__ == "__main__":
    main()
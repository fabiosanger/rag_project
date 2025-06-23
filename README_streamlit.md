# RAG QA System - Streamlit Interface

A beautiful web interface for the RAG (Retrieval-Augmented Generation) Question Answering system built with Streamlit.

## Features

- 🤖 **Interactive Web Interface**: Modern, responsive design
- 📊 **Confidence Scores**: Visual confidence indicators with color coding
- ⚙️ **Configurable Parameters**: Adjust model settings in real-time
- 📁 **Data Upload**: Upload custom JSON files with Q&A pairs
- 🎯 **Real-time Processing**: Instant answers with processing time display
- 📈 **Statistics Dashboard**: System metrics and recent questions

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements_streamlit.txt
   ```

2. **Run the application**:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open your browser**:
   The app will automatically open at `http://localhost:8501`

## Usage

### Basic Usage
1. The app loads with sample data by default
2. Type your question in the text area
3. Click "Get Answer" to receive a response with confidence score
4. Adjust parameters in the sidebar as needed

### Custom Data
1. Prepare a JSON file with Q&A pairs in this format:
   ```json
   [
     {
       "question": "Your question here?",
       "answer": "The corresponding answer here."
     }
   ]
   ```

2. Select "Upload Custom Data" in the sidebar
3. Upload your JSON file
4. Start asking questions!

### Configuration Options

**Model Parameters**:
- **Max Answer Length**: Control the length of generated answers (20-100 tokens)
- **Number of Beams**: Adjust beam search for better quality (1-8 beams)
- **Confidence Threshold**: Set minimum confidence for reliable answers (0-100%)

**Device Selection**:
- **Auto-detect**: Automatically choose GPU if available
- **CPU**: Force CPU usage
- **GPU**: Force GPU usage (if available)

## File Structure

```
rag_project/
├── streamlit_app.py          # Main Streamlit application
├── requirements_streamlit.txt # Dependencies for Streamlit
├── sample_data.json          # Sample Q&A data
├── README_streamlit.md       # This file
└── rag_project/
    └── qa_system.py          # Core QA system
```

## Confidence Scoring

The system provides confidence scores based on semantic similarity:

- 🟢 **High Confidence (70%+)**: Green indicator - reliable answer
- 🟡 **Medium Confidence (40-69%)**: Yellow indicator - moderate reliability
- 🔴 **Low Confidence (<40%)**: Red indicator - low reliability

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**:
   - If you encounter GPU errors, try selecting "CPU" in device options
   - Ensure you have compatible CUDA drivers installed

2. **Memory Issues**:
   - Reduce "Max Answer Length" parameter
   - Use smaller datasets
   - Close other applications using GPU memory

3. **Model Loading Issues**:
   - Check internet connection (models are downloaded on first run)
   - Ensure sufficient disk space for model downloads

### Performance Tips

- **GPU Usage**: Enable GPU for faster processing
- **Batch Processing**: Process multiple questions in sequence
- **Model Caching**: The app caches the model to avoid reloading

## Customization

### Styling
Modify the CSS in the `st.markdown()` section to customize the appearance.

### Adding Features
- Extend the `get_answer_with_confidence()` function for additional metrics
- Add more configuration options in the sidebar
- Implement session state for conversation history

## Dependencies

- **Streamlit**: Web framework
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face models
- **Sentence-Transformers**: Embedding models
- **Scikit-learn**: Machine learning utilities
- **NumPy/Pandas**: Data processing

## License

This project is part of the RAG QA System. See the main project license for details.
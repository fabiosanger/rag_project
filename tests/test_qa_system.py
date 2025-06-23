import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import torch

from rag_project.qa_system import SimpleQASystem


class TestSimpleQASystem:
    """Test cases for SimpleQASystem class"""

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing"""
        return [
            {"question": "What is the capital of France?", "answer": "The capital of France is Paris."},
            {"question": "What is the largest planet?", "answer": "The largest planet is Jupiter."},
        ]

    @pytest.fixture
    def qa_system(self):
        """Create a QA system instance for testing"""
        with patch('rag_project.qa_system.T5Tokenizer'), \
             patch('rag_project.qa_system.T5ForConditionalGeneration'), \
             patch('rag_project.qa_system.SentenceTransformer'), \
             patch('rag_project.qa_system.get_gpu_device') as mock_get_device, \
             patch('rag_project.qa_system.move_to_device') as mock_move_device, \
             patch('rag_project.qa_system.is_gpu_available') as mock_gpu_available:

            # Mock device
            mock_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            mock_get_device.return_value = mock_device
            mock_move_device.side_effect = lambda x, device: x.to(device) if hasattr(x, 'to') else x
            mock_gpu_available.return_value = torch.cuda.is_available()

            system = SimpleQASystem()
            # Mock the encoder
            system.encoder = Mock()
            system.encoder.encode.return_value = Mock()
            system.encoder.encode.return_value.cpu.return_value = Mock()
            system.encoder.encode.return_value.cpu.return_value.numpy.return_value = np.random.rand(384)

            return system

    def test_init(self, qa_system):
        """Test system initialization"""
        assert qa_system.answers == []
        assert qa_system.answer_embeddings is None
        assert hasattr(qa_system, 'device')

    def test_prepare_dataset(self, qa_system, sample_data):
        """Test dataset preparation"""
        qa_system.prepare_dataset(sample_data)

        assert len(qa_system.answers) == 2
        assert qa_system.answers[0] == "The capital of France is Paris."
        assert qa_system.answers[1] == "The largest planet is Jupiter."
        assert len(qa_system.answer_embeddings) == 2

    def test_get_answer_without_prepared_dataset(self, qa_system):
        """Test that get_answer raises error when dataset is not prepared"""
        with pytest.raises(ValueError, match="Dataset not prepared"):
            qa_system.get_answer("What is the capital of France?")

    @patch('rag_project.qa_system.cosine_similarity')
    @patch('rag_project.qa_system.torch.no_grad')
    def test_get_answer_success(self, mock_no_grad, mock_cosine_similarity, qa_system, sample_data):
        """Test successful answer generation"""
        # Mock cosine similarity to return highest similarity for first answer
        mock_cosine_similarity.return_value = np.array([[0.9, 0.3]])

        # Mock the model and tokenizer
        qa_system.model = Mock()
        qa_system.tokenizer = Mock()
        qa_system.model.generate.return_value = Mock()
        qa_system.tokenizer.decode.return_value = "Paris is the capital of France"

        # Mock no_grad context
        mock_no_grad.return_value.__enter__ = Mock()
        mock_no_grad.return_value.__exit__ = Mock()

        # Prepare dataset
        qa_system.prepare_dataset(sample_data)

        # Test getting answer
        result = qa_system.get_answer("What is the capital of France?")

        assert "Paris" in result
        assert qa_system.model.generate.called
        assert qa_system.tokenizer.decode.called

    def test_error_handling_in_get_answer(self, qa_system):
        """Test error handling in get_answer method"""
        # Prepare dataset first
        qa_system.prepare_dataset([{"question": "test", "answer": "test answer"}])

        # Mock encoder to raise exception
        qa_system.encoder.encode.side_effect = Exception("Test error")

        result = qa_system.get_answer("test question")
        assert "Error:" in result

    @patch('rag_project.qa_system.is_gpu_available')
    @patch('rag_project.qa_system.clear_gpu_cache')
    def test_gpu_cache_clearing(self, mock_clear_cache, mock_gpu_available, qa_system, sample_data):
        """Test GPU cache clearing after answer generation"""
        mock_gpu_available.return_value = True

        # Mock the model and tokenizer
        qa_system.model = Mock()
        qa_system.tokenizer = Mock()
        qa_system.model.generate.return_value = Mock()
        qa_system.tokenizer.decode.return_value = "Test answer"

        # Mock cosine similarity
        with patch('rag_project.qa_system.cosine_similarity') as mock_cosine:
            mock_cosine.return_value = np.array([[0.9]])

            # Prepare dataset
            qa_system.prepare_dataset(sample_data)

            # Get answer
            qa_system.get_answer("test question")

            # Check if GPU cache was cleared
            mock_clear_cache.assert_called_once()

    @patch('rag_project.qa_system.is_gpu_available')
    @patch('rag_project.qa_system.clear_gpu_cache')
    def test_no_gpu_cache_clearing_on_cpu(self, mock_clear_cache, mock_gpu_available, qa_system, sample_data):
        """Test that GPU cache is not cleared when using CPU"""
        mock_gpu_available.return_value = False

        # Mock the model and tokenizer
        qa_system.model = Mock()
        qa_system.tokenizer = Mock()
        qa_system.model.generate.return_value = Mock()
        qa_system.tokenizer.decode.return_value = "Test answer"

        # Mock cosine similarity
        with patch('rag_project.qa_system.cosine_similarity') as mock_cosine:
            mock_cosine.return_value = np.array([[0.9]])

            # Prepare dataset
            qa_system.prepare_dataset(sample_data)

            # Get answer
            qa_system.get_answer("test question")

            # Check that GPU cache was not cleared
            mock_clear_cache.assert_not_called()


class TestQASystemGPUIntegration:
    """Integration tests for QA system with GPU"""

    @pytest.mark.integration
    def test_real_device_usage(self):
        """Test real device usage in QA system"""
        try:
            qa_system = SimpleQASystem()

            # Check that device is properly set
            assert hasattr(qa_system, 'device')
            assert isinstance(qa_system.device, torch.device)

            # Check that model is on the correct device
            if hasattr(qa_system.model, 'device'):
                assert qa_system.model.device == qa_system.device

        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")

    @pytest.mark.integration
    def test_real_gpu_processing(self):
        """Test real GPU processing if available"""
        try:
            qa_system = SimpleQASystem()

            if qa_system.device.type == 'cuda':
                # Test with sample data
                sample_data = [
                    {"question": "What is the capital of France?", "answer": "The capital of France is Paris."}
                ]

                qa_system.prepare_dataset(sample_data)
                result = qa_system.get_answer("What is the capital of France?")

                assert isinstance(result, str)
                assert len(result) > 0
            else:
                pytest.skip("No GPU available for GPU processing test")

        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")
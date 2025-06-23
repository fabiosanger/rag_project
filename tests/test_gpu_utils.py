import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
import logging

from rag_project.utils.gpu_utils import (
    GPUManager, get_gpu_device, is_gpu_available, get_gpu_info,
    clear_gpu_cache, get_optimal_batch_size, move_to_device,
    get_device_info
)


class TestGPUManager:
    """Test cases for GPUManager class"""

    @pytest.fixture
    def gpu_manager(self):
        """Create GPU manager instance"""
        return GPUManager()

    def test_init(self, gpu_manager):
        """Test GPU manager initialization"""
        assert hasattr(gpu_manager, 'device')
        assert hasattr(gpu_manager, 'gpu_info')
        assert hasattr(gpu_manager, 'logger')

    @patch('torch.cuda.is_available')
    @patch('torch.device')
    def test_get_optimal_device_gpu_available(self, mock_device, mock_cuda_available):
        """Test device selection when GPU is available"""
        mock_cuda_available.return_value = True
        mock_device.return_value = torch.device('cuda')

        # Mock CUDA functionality
        with patch('torch.tensor') as mock_tensor:
            mock_tensor.return_value = Mock()
            with patch('torch.cuda.empty_cache'):
                gpu_manager = GPUManager()
                assert gpu_manager.device.type == 'cuda'

    @patch('torch.cuda.is_available')
    @patch('torch.device')
    def test_get_optimal_device_gpu_unavailable(self, mock_device, mock_cuda_available):
        """Test device selection when GPU is not available"""
        mock_cuda_available.return_value = False
        mock_device.return_value = torch.device('cpu')

        gpu_manager = GPUManager()
        assert gpu_manager.device.type == 'cpu'

    @patch('torch.cuda.is_available')
    def test_get_optimal_device_cuda_failure(self, mock_cuda_available):
        """Test device selection when CUDA fails to initialize"""
        mock_cuda_available.return_value = True

        # Mock CUDA failure
        with patch('torch.tensor', side_effect=Exception("CUDA error")):
            gpu_manager = GPUManager()
            assert gpu_manager.device.type == 'cpu'

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_name')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.get_device_capability')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    def test_get_gpu_info_gpu_available(self, mock_memory_reserved, mock_memory_allocated,
                                       mock_capability, mock_properties, mock_name, mock_available):
        """Test GPU info retrieval when GPU is available"""
        mock_available.return_value = True
        mock_name.return_value = "NVIDIA GeForce RTX 3080"
        mock_properties.return_value = Mock(total_memory=8589934592)  # 8GB
        mock_capability.return_value = (8, 6)
        mock_memory_allocated.return_value = 1073741824  # 1GB
        mock_memory_reserved.return_value = 2147483648  # 2GB

        gpu_manager = GPUManager()
        gpu_info = gpu_manager.get_gpu_info()

        assert gpu_info['available'] is True
        assert gpu_info['device_count'] > 0
        assert gpu_info['device_name'] == "NVIDIA GeForce RTX 3080"
        assert gpu_info['memory_info']['total'] == 8589934592

    @patch('torch.cuda.is_available')
    def test_get_gpu_info_gpu_unavailable(self, mock_available):
        """Test GPU info retrieval when GPU is not available"""
        mock_available.return_value = False

        gpu_manager = GPUManager()
        gpu_info = gpu_manager.get_gpu_info()

        assert gpu_info['available'] is False
        assert gpu_info['device_count'] == 0
        assert gpu_info['device_name'] is None

    def test_get_device(self, gpu_manager):
        """Test getting current device"""
        device = gpu_manager.get_device()
        assert isinstance(device, torch.device)

    @patch('torch.cuda.is_available')
    def test_is_gpu_available(self, mock_available):
        """Test GPU availability check"""
        mock_available.return_value = True

        with patch('torch.tensor') as mock_tensor:
            mock_tensor.return_value = Mock()
            with patch('torch.cuda.empty_cache'):
                gpu_manager = GPUManager()
                assert gpu_manager.is_gpu_available() is True

    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    @patch('torch.cuda.get_device_properties')
    def test_get_memory_usage_gpu(self, mock_properties, mock_reserved, mock_allocated):
        """Test memory usage retrieval on GPU"""
        mock_properties.return_value = Mock(total_memory=8589934592)
        mock_allocated.return_value = 1073741824
        mock_reserved.return_value = 2147483648

        with patch('torch.cuda.is_available', return_value=True):
            gpu_manager = GPUManager()
            memory_info = gpu_manager.get_memory_usage()

            assert memory_info['total'] == 8589934592
            assert memory_info['allocated'] == 1073741824
            assert memory_info['cached'] == 2147483648

    @patch('torch.cuda.is_available')
    def test_get_memory_usage_cpu(self, mock_available):
        """Test memory usage retrieval on CPU"""
        mock_available.return_value = False

        gpu_manager = GPUManager()
        memory_info = gpu_manager.get_memory_usage()

        assert memory_info['total'] == 0
        assert memory_info['allocated'] == 0
        assert memory_info['cached'] == 0

    @patch('torch.cuda.empty_cache')
    def test_clear_cache_gpu(self, mock_clear_cache):
        """Test GPU cache clearing"""
        with patch('torch.cuda.is_available', return_value=True):
            gpu_manager = GPUManager()
            gpu_manager.clear_cache()
            mock_clear_cache.assert_called_once()

    @patch('torch.cuda.is_available')
    def test_clear_cache_cpu(self, mock_available):
        """Test cache clearing on CPU (should do nothing)"""
        mock_available.return_value = False

        gpu_manager = GPUManager()
        gpu_manager.clear_cache()  # Should not raise any error

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.set_device')
    def test_set_device_valid(self, mock_set_device, mock_device_count, mock_available):
        """Test setting valid GPU device"""
        mock_available.return_value = True
        mock_device_count.return_value = 2

        gpu_manager = GPUManager()
        gpu_manager.set_device(1)

        mock_set_device.assert_called_once_with(1)

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_set_device_invalid(self, mock_device_count, mock_available):
        """Test setting invalid GPU device"""
        mock_available.return_value = True
        mock_device_count.return_value = 1

        gpu_manager = GPUManager()
        gpu_manager.set_device(5)  # Invalid device ID

        # Should log warning but not raise error

    @patch('torch.cuda.is_available')
    def test_get_optimal_batch_size_gpu(self, mock_available):
        """Test optimal batch size calculation on GPU"""
        mock_available.return_value = True

        with patch.object(GPUManager, 'get_memory_usage') as mock_memory:
            mock_memory.return_value = {
                'total': 8589934592,  # 8GB
                'allocated': 1073741824,  # 1GB
                'cached': 2147483648,  # 2GB
                'free': 5368709120  # 5GB
            }

            gpu_manager = GPUManager()
            batch_size = gpu_manager.get_optimal_batch_size(1000, 32)

            # Should return a reasonable batch size
            assert 1 <= batch_size <= 32

    @patch('torch.cuda.is_available')
    def test_get_optimal_batch_size_cpu(self, mock_available):
        """Test optimal batch size calculation on CPU"""
        mock_available.return_value = False

        gpu_manager = GPUManager()
        batch_size = gpu_manager.get_optimal_batch_size(1000, 32)

        assert batch_size == 1


class TestGPUUtilityFunctions:
    """Test cases for GPU utility functions"""

    @patch('rag_project.utils.gpu_utils.GPUManager')
    def test_get_gpu_device(self, mock_gpu_manager_class):
        """Test get_gpu_device function"""
        mock_manager = Mock()
        mock_manager.get_device.return_value = torch.device('cuda')
        mock_gpu_manager_class.return_value = mock_manager

        device = get_gpu_device()
        assert device == torch.device('cuda')

    @patch('rag_project.utils.gpu_utils.GPUManager')
    def test_is_gpu_available(self, mock_gpu_manager_class):
        """Test is_gpu_available function"""
        mock_manager = Mock()
        mock_manager.is_gpu_available.return_value = True
        mock_gpu_manager_class.return_value = mock_manager

        result = is_gpu_available()
        assert result is True

    @patch('rag_project.utils.gpu_utils.GPUManager')
    def test_get_gpu_info(self, mock_gpu_manager_class):
        """Test get_gpu_info function"""
        mock_manager = Mock()
        mock_manager.get_gpu_info.return_value = {'available': True, 'device_count': 1}
        mock_gpu_manager_class.return_value = mock_manager

        info = get_gpu_info()
        assert info['available'] is True
        assert info['device_count'] == 1

    @patch('rag_project.utils.gpu_utils.GPUManager')
    def test_clear_gpu_cache(self, mock_gpu_manager_class):
        """Test clear_gpu_cache function"""
        mock_manager = Mock()
        mock_gpu_manager_class.return_value = mock_manager

        clear_gpu_cache()
        mock_manager.return_value.clear_cache.assert_called_once()

    @patch('rag_project.utils.gpu_utils.GPUManager')
    def test_get_optimal_batch_size(self, mock_gpu_manager_class):
        """Test get_optimal_batch_size function"""
        mock_manager = Mock()
        mock_manager.get_optimal_batch_size.return_value = 16
        mock_gpu_manager_class.return_value = mock_manager

        batch_size = get_optimal_batch_size(1000, 32)
        assert batch_size == 16

    def test_move_to_device_with_device(self):
        """Test move_to_device function with specified device"""
        tensor = torch.tensor([1, 2, 3])
        device = torch.device('cpu')

        result = move_to_device(tensor, device)
        assert result.device == device

    @patch('rag_project.utils.gpu_utils.get_gpu_device')
    def test_move_to_device_without_device(self, mock_get_device):
        """Test move_to_device function without specified device"""
        mock_get_device.return_value = torch.device('cuda')
        tensor = torch.tensor([1, 2, 3])

        result = move_to_device(tensor)
        assert result.device == torch.device('cuda')

    @patch('rag_project.utils.gpu_utils.GPUManager')
    def test_get_device_info_gpu(self, mock_gpu_manager_class):
        """Test get_device_info function with GPU"""
        mock_manager = Mock()
        mock_manager.get_device.return_value = torch.device('cuda')
        mock_manager.is_gpu_available.return_value = True
        mock_manager.get_gpu_info.return_value = {'device_name': 'NVIDIA RTX 3080'}
        mock_manager.get_memory_usage.return_value = {
            'allocated': 1073741824,  # 1GB
            'free': 7516192768  # 7GB
        }
        mock_gpu_manager_class.return_value = mock_manager

        info = get_device_info()
        assert "GPU:" in info
        assert "NVIDIA RTX 3080" in info

    @patch('rag_project.utils.gpu_utils.GPUManager')
    def test_get_device_info_cpu(self, mock_gpu_manager_class):
        """Test get_device_info function with CPU"""
        mock_manager = Mock()
        mock_manager.get_device.return_value = torch.device('cpu')
        mock_gpu_manager_class.return_value = mock_manager

        info = get_device_info()
        assert info == "CPU"


class TestGPUIntegration:
    """Integration tests for GPU functionality"""

    @pytest.mark.integration
    def test_real_gpu_detection(self):
        """Test real GPU detection (integration test)"""
        gpu_manager = GPUManager()

        # This test will run with real GPU if available
        device = gpu_manager.get_device()
        assert isinstance(device, torch.device)

        gpu_available = gpu_manager.is_gpu_available()
        assert isinstance(gpu_available, bool)

        gpu_info = gpu_manager.get_gpu_info()
        assert isinstance(gpu_info, dict)
        assert 'available' in gpu_info

    @pytest.mark.integration
    def test_real_tensor_operations(self):
        """Test real tensor operations on available device"""
        device = get_gpu_device()

        # Create tensor on the device
        tensor = torch.tensor([1, 2, 3, 4, 5], device=device)
        assert tensor.device == device

        # Test basic operations
        result = tensor * 2
        assert result.device == device
        assert torch.all(result == torch.tensor([2, 4, 6, 8, 10], device=device))

    @pytest.mark.integration
    def test_real_memory_operations(self):
        """Test real memory operations (if GPU available)"""
        gpu_manager = GPUManager()

        if gpu_manager.is_gpu_available():
            # Test memory usage
            memory_info = gpu_manager.get_memory_usage()
            assert 'total' in memory_info
            assert 'allocated' in memory_info
            assert 'free' in memory_info

            # Test cache clearing
            gpu_manager.clear_cache()
        else:
            # Skip test if no GPU
            pytest.skip("No GPU available for memory operations test")
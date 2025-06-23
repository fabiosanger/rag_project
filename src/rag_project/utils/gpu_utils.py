"""
GPU utilities for the RAG system.
"""

import torch
import logging
from typing import Optional, Dict, Any, List
import subprocess
import json


class GPUManager:
    """
    Manages GPU availability and device selection for the RAG system.
    """

    def __init__(self):
        """Initialize GPU manager."""
        self.logger = logging.getLogger(__name__)
        self.device = self._get_optimal_device()
        self.gpu_info = self._get_gpu_info()

    def _get_optimal_device(self) -> torch.device:
        """
        Get the optimal device (GPU if available, otherwise CPU).

        Returns:
            torch.device: The optimal device to use
        """
        if torch.cuda.is_available():
            # Check if CUDA is properly configured
            try:
                # Test CUDA functionality
                test_tensor = torch.tensor([1.0], device='cuda')
                del test_tensor
                torch.cuda.empty_cache()

                device = torch.device('cuda')
                self.logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
                return device
            except Exception as e:
                self.logger.warning(f"CUDA available but failed to initialize: {e}")
                return torch.device('cpu')
        else:
            self.logger.info("No GPU available, using CPU")
            return torch.device('cpu')

    def _get_gpu_info(self) -> Dict[str, Any]:
        """
        Get detailed GPU information.

        Returns:
            Dictionary containing GPU information
        """
        gpu_info = {
            'available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': None,
            'device_name': None,
            'memory_info': None,
            'compute_capability': None
        }

        if torch.cuda.is_available():
            try:
                gpu_info['current_device'] = torch.cuda.current_device()
                gpu_info['device_name'] = torch.cuda.get_device_name(0)
                gpu_info['memory_info'] = {
                    'total': torch.cuda.get_device_properties(0).total_memory,
                    'allocated': torch.cuda.memory_allocated(0),
                    'cached': torch.cuda.memory_reserved(0)
                }
                gpu_info['compute_capability'] = torch.cuda.get_device_capability(0)
            except Exception as e:
                self.logger.warning(f"Failed to get detailed GPU info: {e}")

        return gpu_info

    def get_device(self) -> torch.device:
        """
        Get the current device.

        Returns:
            torch.device: Current device
        """
        return self.device

    def is_gpu_available(self) -> bool:
        """
        Check if GPU is available and working.

        Returns:
            bool: True if GPU is available and functional
        """
        return self.device.type == 'cuda'

    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Get GPU information.

        Returns:
            Dictionary with GPU information
        """
        return self.gpu_info.copy()

    def get_memory_usage(self) -> Dict[str, int]:
        """
        Get current GPU memory usage.

        Returns:
            Dictionary with memory usage information
        """
        if not self.is_gpu_available():
            return {'total': 0, 'allocated': 0, 'cached': 0, 'free': 0}

        try:
            allocated = torch.cuda.memory_allocated(0)
            cached = torch.cuda.memory_reserved(0)
            total = torch.cuda.get_device_properties(0).total_memory
            free = total - allocated

            return {
                'total': total,
                'allocated': allocated,
                'cached': cached,
                'free': free
            }
        except Exception as e:
            self.logger.error(f"Failed to get memory usage: {e}")
            return {'total': 0, 'allocated': 0, 'cached': 0, 'free': 0}

    def clear_cache(self):
        """Clear GPU memory cache."""
        if self.is_gpu_available():
            torch.cuda.empty_cache()
            self.logger.debug("GPU cache cleared")

    def set_device(self, device_id: int = 0):
        """
        Set specific GPU device.

        Args:
            device_id: GPU device ID to use
        """
        if torch.cuda.is_available() and device_id < torch.cuda.device_count():
            torch.cuda.set_device(device_id)
            self.device = torch.device(f'cuda:{device_id}')
            self.gpu_info = self._get_gpu_info()
            self.logger.info(f"Switched to GPU device {device_id}: {self.gpu_info['device_name']}")
        else:
            self.logger.warning(f"GPU device {device_id} not available")

    def get_optimal_batch_size(self, model_size_mb: float, max_batch_size: int = 32) -> int:
        """
        Calculate optimal batch size based on available GPU memory.

        Args:
            model_size_mb: Size of the model in MB
            max_batch_size: Maximum batch size to consider

        Returns:
            Optimal batch size
        """
        if not self.is_gpu_available():
            return 1  # CPU processing, use small batches

        try:
            memory_info = self.get_memory_usage()
            available_memory = memory_info['free'] / (1024 * 1024)  # Convert to MB

            # Reserve some memory for other operations
            usable_memory = available_memory * 0.8

            # Rough estimation: assume each sample needs ~100MB
            estimated_batch_size = int(usable_memory / 100)

            return min(estimated_batch_size, max_batch_size, 1)
        except Exception as e:
            self.logger.warning(f"Failed to calculate optimal batch size: {e}")
            return 1


def get_gpu_device() -> torch.device:
    """
    Get the optimal GPU device.

    Returns:
        torch.device: Optimal device (GPU if available, otherwise CPU)
    """
    return GPUManager().get_device()


def is_gpu_available() -> bool:
    """
    Check if GPU is available.

    Returns:
        bool: True if GPU is available and functional
    """
    return GPUManager().is_gpu_available()


def get_gpu_info() -> Dict[str, Any]:
    """
    Get GPU information.

    Returns:
        Dictionary with GPU information
    """
    return GPUManager().get_gpu_info()


def clear_gpu_cache():
    """Clear GPU memory cache."""
    GPUManager().clear_cache()


def get_optimal_batch_size(model_size_mb: float = 1000, max_batch_size: int = 32) -> int:
    """
    Get optimal batch size for GPU processing.

    Args:
        model_size_mb: Size of the model in MB
        max_batch_size: Maximum batch size to consider

    Returns:
        Optimal batch size
    """
    return GPUManager().get_optimal_batch_size(model_size_mb, max_batch_size)


def move_to_device(tensor_or_model, device: Optional[torch.device] = None):
    """
    Move tensor or model to the specified device.

    Args:
        tensor_or_model: Tensor or model to move
        device: Target device (uses optimal device if None)

    Returns:
        Tensor or model on the target device
    """
    if device is None:
        device = get_gpu_device()

    return tensor_or_model.to(device)


def get_device_info() -> str:
    """
    Get human-readable device information.

    Returns:
        String with device information
    """
    gpu_manager = GPUManager()
    device = gpu_manager.get_device()

    if device.type == 'cuda':
        gpu_info = gpu_manager.get_gpu_info()
        memory_info = gpu_manager.get_memory_usage()

        return (f"GPU: {gpu_info['device_name']} "
                f"(Memory: {memory_info['allocated']/1024**3:.1f}GB used, "
                f"{memory_info['free']/1024**3:.1f}GB free)")
    else:
        return "CPU"
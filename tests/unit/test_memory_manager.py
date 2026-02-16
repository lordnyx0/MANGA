"""
Unit tests for MemoryManager.
Tests VRAM lifecycle management.
"""

import pytest
from unittest.mock import MagicMock, patch

from core.generation.engines.memory_manager import MemoryManager


@pytest.fixture
def manager():
    return MemoryManager()


class TestOffload:
    @patch("core.generation.engines.memory_manager.torch")
    def test_clears_cuda_cache_when_available(self, mock_torch, manager):
        mock_torch.cuda.is_available.return_value = True
        manager.offload()
        mock_torch.cuda.empty_cache.assert_called_once()

    @patch("core.generation.engines.memory_manager.torch")
    def test_skips_cuda_when_unavailable(self, mock_torch, manager):
        mock_torch.cuda.is_available.return_value = False
        manager.offload()
        mock_torch.cuda.empty_cache.assert_not_called()


class TestSetupOptimizations:
    def test_applies_optimizations_on_cuda(self, manager):
        mock_pipe = MagicMock()
        manager.setup_optimizations(mock_pipe, "cuda")

        mock_pipe.enable_model_cpu_offload.assert_called_once()
        mock_pipe.enable_vae_slicing.assert_called_once()

    def test_skips_optimizations_on_cpu(self, manager):
        mock_pipe = MagicMock()
        manager.setup_optimizations(mock_pipe, "cpu")

        mock_pipe.enable_model_cpu_offload.assert_not_called()
        mock_pipe.enable_vae_slicing.assert_not_called()

    @patch("core.generation.engines.memory_manager.ENABLE_VAE_TILING", True)
    def test_enables_vae_tiling_when_configured(self, manager):
        mock_pipe = MagicMock()
        manager.setup_optimizations(mock_pipe, "cuda")
        mock_pipe.enable_vae_tiling.assert_called_once()

    @patch("core.generation.engines.memory_manager.ENABLE_VAE_TILING", False)
    def test_disables_vae_tiling_when_not_configured(self, manager):
        mock_pipe = MagicMock()
        manager.setup_optimizations(mock_pipe, "cuda")
        mock_pipe.disable_vae_tiling.assert_called_once()

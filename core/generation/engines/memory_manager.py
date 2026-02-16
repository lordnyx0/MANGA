"""
MemoryManager — VRAM lifecycle management for diffusion pipelines.

Extracted from SD15LineartEngine (SRP refactoring).
Handles:
- Garbage collection and CUDA cache clearing
- Pipeline memory optimizations (cpu_offload, vae_slicing, vae_tiling)
"""

import gc
import torch

from config.settings import ENABLE_VAE_TILING
from core.logging.setup import get_logger

logger = get_logger("MemoryManager")


class MemoryManager:
    """
    Manages GPU/CPU memory lifecycle for diffusion pipelines.

    Responsibilities:
    - Offload models and clear VRAM
    - Configure pipeline memory optimizations
    """

    @staticmethod
    def offload() -> None:
        """Force garbage collection and clear CUDA cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def setup_optimizations(pipe, device: str) -> None:
        """
        Apply memory optimizations to a diffusion pipeline.

        Args:
            pipe: A diffusers pipeline instance
            device: Target device ("cuda" or "cpu")
        """
        if device != "cuda":
            return

        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()

        if ENABLE_VAE_TILING:
            logger.info("VAE Tiling ENABLED via settings.")
            pipe.enable_vae_tiling()
        else:
            logger.info("VAE Tiling DISABLED (avoids edge artifacts).")
            pipe.disable_vae_tiling()

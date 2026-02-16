"""
LineartPreprocessor — Image analysis and ControlNet conditioning preparation.

Extracted from SD15LineartEngine (SRP refactoring).
Handles:
- Lineart quality metrics (edge density, contrast, brightness)
- Auto-contrast for low-quality scans
- Image inversion for ControlNet (Black-on-White → White-on-Black)
- Resolution safety (downscale for SD 1.5 max dim)
"""

from typing import Dict, Tuple
from PIL import Image, ImageOps
import numpy as np

from config.settings import (
    V3_LINEART_MIN_EDGE_DENSITY,
    V3_LINEART_AUTOCONTRAST_CUTOFF,
)
from core.logging.setup import get_logger

logger = get_logger("LineartPreprocessor")


class LineartPreprocessor:
    """
    Prepares lineart images for ControlNet conditioning.

    Responsibilities:
    - Compute quality metrics (edge density, contrast, brightness)
    - Auto-contrast enhancement for low-quality scans
    - Inversion for ControlNet (manga is Black-on-White, ControlNet expects White-on-Black)
    - Resolution downscaling safety for SD 1.5
    """

    DEFAULT_MAX_DIM = 1024

    @staticmethod
    def compute_metrics(image_gray: Image.Image) -> Dict[str, float]:
        """
        Compute lineart quality metrics from a grayscale image.

        Returns dict with:
        - edge_density: fraction of pixels with strong gradients
        - contrast_std: standard deviation of pixel intensities
        - mean_brightness: average pixel brightness (0-255)
        """
        arr = np.array(image_gray, dtype=np.float32)
        gx = np.abs(np.diff(arr, axis=1))
        gy = np.abs(np.diff(arr, axis=0))
        grad = np.zeros_like(arr)
        grad[:, 1:] += gx
        grad[1:, :] += gy
        edge_density = float(np.mean(grad > 24.0))
        contrast_std = float(np.std(arr))
        mean_brightness = float(np.mean(arr))
        return {
            "edge_density": edge_density,
            "contrast_std": contrast_std,
            "mean_brightness": mean_brightness,
        }

    def prepare_control_image(
        self,
        line_art: Image.Image,
        max_dim: int = DEFAULT_MAX_DIM,
    ) -> Tuple[Image.Image, bool, Tuple[int, int]]:
        """
        Prepare a lineart image for ControlNet conditioning.

        Steps:
        1. Downscale if larger than max_dim (avoids SD 1.5 artifacts)
        2. Auto-contrast if edge density is too low
        3. Invert if manga format (Black-on-White → White-on-Black)

        Args:
            line_art: Input RGB lineart image
            max_dim: Maximum dimension for generation safety

        Returns:
            (control_image, was_resized, original_size)
            - control_image: Prepared RGB image for ControlNet
            - was_resized: Whether downscaling was applied (caller must upscale result)
            - original_size: Original (width, height) for upscaling
        """
        original_size = line_art.size
        needs_resize = max(original_size) > max_dim

        if needs_resize:
            ratio = max_dim / max(original_size)
            new_w = int(original_size[0] * ratio)
            new_h = int(original_size[1] * ratio)
            target_size = (new_w, new_h)
            logger.info(f"Downscaling for generation: {original_size} -> {target_size}")
            control_image = line_art.resize(target_size, Image.LANCZOS)
        else:
            control_image = line_art

        # Convert to grayscale for analysis
        bw_control = control_image.convert("L")
        metrics = self.compute_metrics(bw_control)
        logger.debug(
            "Lineart metrics: edge=%.4f contrast_std=%.2f mean=%.2f",
            metrics["edge_density"],
            metrics["contrast_std"],
            metrics["mean_brightness"],
        )

        # Auto-contrast for low-quality scans
        if metrics["edge_density"] < V3_LINEART_MIN_EDGE_DENSITY:
            bw_control = ImageOps.autocontrast(
                bw_control, cutoff=V3_LINEART_AUTOCONTRAST_CUTOFF
            )
            metrics = self.compute_metrics(bw_control)
            logger.info(
                "Lineart auto-contrast applied (edge=%.4f).",
                metrics["edge_density"],
            )

        # Inversion logic for ControlNet
        if metrics["mean_brightness"] > 127:
            # Black-on-White (standard manga) → invert to White-on-Black
            control_image = ImageOps.invert(bw_control).convert("RGB")
        else:
            # Already White-on-Black (lineart map) → keep
            control_image = bw_control.convert("RGB")

        return control_image, needs_resize, original_size

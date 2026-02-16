"""
CompositionService — Final image compositing and bubble masking.

Extracted from SD15LineartEngine (SRP refactoring).
Handles:
- Bubble masking: cleans text regions before/after generation
- Alpha compositing: blends colorized output with original lineart
- Blur smoothing: optional Gaussian blur on color layer
"""

from typing import Dict, List, Optional
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

from core.logging.setup import get_logger

logger = get_logger("CompositionService")


class CompositionService:
    """
    Handles final compositing of colorized output with original lineart.

    Responsibilities:
    - Pre-clean text/bubble regions (fill white) before generation
    - Post-clean text regions in the color layer
    - Alpha-blend colorized image with original lineart (preserving black lines)
    - Optional Gaussian blur smoothing on color layer
    """

    @staticmethod
    def clean_bubble_regions(
        image: Image.Image,
        detections: List[Dict],
        padding: int = 2,
    ) -> Image.Image:
        """
        Fill text/bubble regions with white to prevent ghosting.

        Args:
            image: RGB image to clean
            detections: List of detection dicts with 'class_id'/'class_name' and 'bbox'
            padding: Extra pixels around each bbox

        Returns:
            Cleaned copy of the image
        """
        cleaned = image.copy()
        draw = ImageDraw.Draw(cleaned)
        count = 0

        for det in detections:
            if det.get("class_id") == 3 or det.get("class_name") == "text":
                bbox = det.get("bbox")
                if bbox:
                    x1, y1, x2, y2 = bbox
                    draw.rectangle(
                        [x1 - padding, y1 - padding, x2 + padding, y2 + padding],
                        fill=(255, 255, 255),
                    )
                    count += 1

        if count > 0:
            logger.info(f"Cleaned {count} bubble region(s) (Anti-Ghosting).")

        return cleaned

    @staticmethod
    def compose_final(
        base_image: Image.Image,
        colorized_image: Image.Image,
        detections: Optional[List[Dict]] = None,
        blur_radius: float = 0.0,
    ) -> Image.Image:
        """
        Combine original lineart (base) with AI-generated color using alpha compositing.

        Preserves the original black lineart by using luminance as an alpha mask:
        - Dark pixels (lines) → high alpha → base dominates
        - Bright pixels (white areas) → low alpha → color dominates

        Args:
            base_image: Original lineart image
            colorized_image: AI-generated colorized image
            detections: Optional YOLO detections for bubble masking
            blur_radius: Gaussian blur on the color layer (smooths halos)

        Returns:
            Composed PIL Image
        """
        base = base_image.convert("RGB")
        color = colorized_image.convert("RGB")

        # Resize safety
        if base.size != color.size:
            color = color.resize(base.size, Image.LANCZOS)

        # Post-clean text regions in the color layer
        if detections:
            draw = ImageDraw.Draw(color)
            for det in detections:
                if det.get("class_id") == 3 or det.get("class_name") == "text":
                    bbox = det.get("bbox")
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        draw.rectangle(
                            [x1 - 4, y1 - 4, x2 + 4, y2 + 4],
                            fill=(255, 255, 255),
                        )

        # Optional blur smoothing
        if blur_radius > 0:
            color = color.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # Alpha blending using lineart luminance
        line_np = np.array(base).astype(np.float32) / 255.0
        color_np = np.array(color).astype(np.float32) / 255.0

        # Alpha from brightness: darker = more opaque (preserves lines)
        gray = np.mean(line_np, axis=2)
        alpha = 1.0 - np.clip(gray, 0.0, 1.0)
        alpha = alpha[..., np.newaxis]

        # Blend: color * (1 - alpha) + line * alpha
        out_np = color_np * (1.0 - alpha) + line_np * alpha

        composed = Image.fromarray(
            (np.clip(out_np, 0.0, 1.0) * 255).astype(np.uint8)
        )
        return composed

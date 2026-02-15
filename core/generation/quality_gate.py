from typing import Dict
import numpy as np
from PIL import Image


def analyze_avqv_metrics(image: Image.Image) -> Dict[str, float]:
    """Métricas AVQV mínimas para triagem de artefatos cromáticos."""
    rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    max_ch = np.max(rgb, axis=2)
    min_ch = np.min(rgb, axis=2)
    sat = (max_ch - min_ch) / (max_ch + 1e-8)

    extreme = np.logical_or(rgb <= 0.01, rgb >= 0.99)

    return {
        "saturation_mean": float(np.mean(sat)),
        "extreme_pixels_ratio": float(np.mean(np.any(extreme, axis=2))),
        "color_std": float(np.std(rgb)),
    }


def should_retry_safe(metrics: Dict[str, float], thresholds: Dict[str, float]) -> bool:
    return (
        metrics.get("saturation_mean", 0.0) >= thresholds.get("saturation_mean", 1.0)
        or metrics.get("extreme_pixels_ratio", 0.0) >= thresholds.get("extreme_pixels_ratio", 1.0)
        or metrics.get("color_std", 0.0) >= thresholds.get("color_std", 1.0)
    )

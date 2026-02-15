from PIL import Image

from core.generation.quality_gate import analyze_avqv_metrics, should_retry_safe


def test_analyze_avqv_metrics_keys():
    img = Image.new("RGB", (32, 32), (255, 0, 255))
    metrics = analyze_avqv_metrics(img)
    assert set(metrics.keys()) == {"saturation_mean", "extreme_pixels_ratio", "color_std"}


def test_should_retry_safe_thresholds():
    metrics = {"saturation_mean": 0.9, "extreme_pixels_ratio": 0.1, "color_std": 0.2}
    thresholds = {"saturation_mean": 0.8, "extreme_pixels_ratio": 0.5, "color_std": 0.5}
    assert should_retry_safe(metrics, thresholds) is True

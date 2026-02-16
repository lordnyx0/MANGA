"""
Unit tests for LineartPreprocessor.
Tests image analysis, auto-contrast, and ControlNet inversion.
"""

import pytest
from PIL import Image, ImageDraw
import numpy as np

from core.generation.engines.lineart_preprocessor import LineartPreprocessor


@pytest.fixture
def preprocessor():
    return LineartPreprocessor()


class TestComputeMetrics:
    def test_returns_expected_keys(self):
        gray = Image.new("L", (64, 64), 255)
        m = LineartPreprocessor.compute_metrics(gray)
        assert "edge_density" in m
        assert "contrast_std" in m
        assert "mean_brightness" in m

    def test_uniform_image_has_zero_edge_density(self):
        gray = Image.new("L", (64, 64), 128)
        m = LineartPreprocessor.compute_metrics(gray)
        assert m["edge_density"] == 0.0

    def test_full_contrast_image_has_high_edge_density(self):
        # Create checkerboard: alternating 0 and 255
        arr = np.zeros((64, 64), dtype=np.uint8)
        arr[::2, ::2] = 255
        arr[1::2, 1::2] = 255
        gray = Image.fromarray(arr, mode="L")
        m = LineartPreprocessor.compute_metrics(gray)
        assert m["edge_density"] > 0.3


class TestPrepareControlImage:
    def test_inverts_bright_manga(self, preprocessor):
        """Black-on-white manga (brightness > 127) should be inverted for ControlNet."""
        img = Image.new("RGB", (100, 100), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.line([(0, 50), (100, 50)], fill=(0, 0, 0), width=3)

        result, _, _ = preprocessor.prepare_control_image(img, max_dim=512)
        arr = np.array(result.convert("L"))
        # After inversion, the bright background should become dark
        assert np.mean(arr) < 127

    def test_keeps_dark_lineart(self, preprocessor):
        """White-on-black lineart (brightness < 127) should NOT be inverted."""
        img = Image.new("RGB", (100, 100), (20, 20, 20))
        draw = ImageDraw.Draw(img)
        draw.line([(0, 50), (100, 50)], fill=(255, 255, 255), width=3)

        result, _, _ = preprocessor.prepare_control_image(img, max_dim=512)
        arr = np.array(result.convert("L"))
        # Dark image stays dark (mean < 127)
        assert np.mean(arr) < 127

    def test_downscales_large_image(self, preprocessor):
        """Images exceeding max_dim should be downscaled."""
        img = Image.new("RGB", (1500, 1000), (200, 200, 200))
        result, needs_resize, original_size = preprocessor.prepare_control_image(
            img, max_dim=1024
        )
        assert needs_resize is True
        assert original_size == (1500, 1000)
        assert max(result.size) <= 1024

    def test_preserves_small_image(self, preprocessor):
        """Images smaller than max_dim should not be resized."""
        img = Image.new("RGB", (512, 512), (200, 200, 200))
        result, needs_resize, original_size = preprocessor.prepare_control_image(
            img, max_dim=1024
        )
        assert needs_resize is False
        assert original_size == (512, 512)

    def test_returns_rgb(self, preprocessor):
        """Control image should always be RGB."""
        img = Image.new("RGB", (100, 100), (200, 200, 200))
        result, _, _ = preprocessor.prepare_control_image(img, max_dim=512)
        assert result.mode == "RGB"

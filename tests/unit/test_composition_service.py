"""
Unit tests for CompositionService.
Tests bubble masking and alpha compositing.
"""

import pytest
from PIL import Image, ImageDraw
import numpy as np

from core.generation.engines.composition_service import CompositionService


@pytest.fixture
def compositor():
    return CompositionService()


class TestCleanBubbleRegions:
    def test_fills_text_regions_white(self, compositor):
        img = Image.new("RGB", (200, 200), (128, 128, 128))
        detections = [
            {"class_id": 3, "bbox": [10, 10, 50, 50]},
        ]
        cleaned = compositor.clean_bubble_regions(img, detections)
        # Center of the bbox should be white
        px = cleaned.getpixel((30, 30))
        assert all(c >= 250 for c in px)

    def test_ignores_non_text_detections(self, compositor):
        img = Image.new("RGB", (200, 200), (128, 128, 128))
        detections = [
            {"class_id": 0, "class_name": "panel", "bbox": [10, 10, 50, 50]},
        ]
        cleaned = compositor.clean_bubble_regions(img, detections)
        px = cleaned.getpixel((30, 30))
        assert all(c == 128 for c in px)

    def test_does_not_modify_original(self, compositor):
        img = Image.new("RGB", (100, 100), (64, 64, 64))
        detections = [{"class_id": 3, "bbox": [10, 10, 50, 50]}]
        _ = compositor.clean_bubble_regions(img, detections)
        px = img.getpixel((30, 30))
        assert all(c == 64 for c in px), "Original image should not be mutated"


class TestComposeFinal:
    def test_preserves_black_lines(self, compositor):
        """Black lines in lineart should appear dark in the output."""
        lineart = Image.new("L", (100, 100), 255)
        for x in range(100):
            lineart.putpixel((x, 50), 0)
        color = Image.new("RGB", (100, 100), (255, 0, 0))

        result = compositor.compose_final(lineart.convert("RGB"), color)
        px_line = result.getpixel((50, 50))
        assert px_line[0] < 10, "Line area should be dark"

    def test_preserves_color_on_white(self, compositor):
        """White areas in lineart should show full color."""
        lineart = Image.new("RGB", (100, 100), (255, 255, 255))
        color = Image.new("RGB", (100, 100), (255, 0, 0))

        result = compositor.compose_final(lineart, color)
        px = result.getpixel((50, 50))
        assert px[0] > 240, "White areas should pass through color"
        assert px[1] < 15
        assert px[2] < 15

    def test_resizes_mismatched_inputs(self, compositor):
        lineart = Image.new("RGB", (200, 200), (255, 255, 255))
        color = Image.new("RGB", (100, 100), (0, 128, 255))

        result = compositor.compose_final(lineart, color)
        assert result.size == (200, 200), "Output should match base size"

    def test_with_detections_masks_text(self, compositor):
        lineart = Image.new("RGB", (200, 200), (255, 255, 255))
        color = Image.new("RGB", (200, 200), (0, 0, 255))

        detections = [{"class_id": 3, "bbox": [10, 10, 50, 50]}]
        result = compositor.compose_final(lineart, color, detections=detections)

        # Inside detection bbox should be nearly white (bubble cleaned)
        px = result.getpixel((30, 30))
        assert all(c > 230 for c in px), "Text area should be white"

    def test_with_blur(self, compositor):
        lineart = Image.new("RGB", (100, 100), (255, 255, 255))
        color = Image.new("RGB", (100, 100), (255, 0, 0))

        result = compositor.compose_final(lineart, color, blur_radius=2.0)
        assert result.size == (100, 100), "Blur should not change dimensions"

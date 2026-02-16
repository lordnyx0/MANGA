import unittest
from PIL import Image
import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

from core.generation.engines.composition_service import CompositionService


class TestBubbleMasking(unittest.TestCase):
    def test_bubble_masking_cleans_color_layer(self):
        compositor = CompositionService()

        # Base image: white (255) everywhere
        base = Image.new("L", (100, 100), 255)

        # Color layer: solid blue (0, 0, 255) - represents "dirty" bubble
        color = Image.new("RGB", (100, 100), (0, 0, 255))

        # Detection: text bubble at [20, 20, 60, 60]
        detections = [
            {'class_name': 'text', 'bbox': [20, 20, 60, 60], 'class_id': 3}
        ]

        # Call compose_final with detections
        # This should turn the color layer region white before blending
        result = compositor.compose_final(base.convert("RGB"), color, detections=detections)

        # 1. Check inside bubble (should be white now)
        px_inside = result.getpixel((30, 30))
        # Expected: white because color layer was flushed and base is white
        self.assertTrue(all(c > 240 for c in px_inside), f"Inside bubble should be white, got {px_inside}")

        # 2. Check outside bubble (should have the blue color since base is white)
        px_outside = result.getpixel((80, 80))
        self.assertTrue(px_outside[2] > 200, f"Outside bubble should have blue, got {px_outside}")

        # 3. Check padding area
        px_edge = result.getpixel((18, 18))
        self.assertTrue(all(c > 240 for c in px_edge), f"Padding area should also be white, got {px_edge}")

if __name__ == '__main__':
    unittest.main()

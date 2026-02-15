import unittest
from PIL import Image
import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

from core.generation.engines.sd15_lineart_engine import SD15LineartEngine

class TestBubbleMasking(unittest.TestCase):
    def test_bubble_masking_cleans_color_layer(self):
        engine = SD15LineartEngine()
        
        # Base image: white (255) everywhere
        base = Image.new("L", (100, 100), 255)
        
        # Color layer: solid blue (0, 0, 255) - represents "dirty" bubble
        color = Image.new("RGB", (100, 100), (0, 0, 255))
        
        # Detection: text bubble at [20, 20, 60, 60]
        detections = [
            {'class_name': 'text', 'bbox': [20, 20, 60, 60], 'class_id': 3}
        ]
        
        # Call compose_final with detections
        # This should turn the color layer region [16, 16, 64, 64] white before multiply
        result = engine.compose_final(base, color, detections=detections)
        
        # 1. Check inside bubble (should be white now)
        px_inside = result.getpixel((30, 30))
        # Expected: white because color layer was flushed and base is white
        self.assertEqual(px_inside, (255, 255, 255), f"Inside bubble should be white, got {px_inside}")
        
        # 2. Check outside bubble (should remain blue)
        px_outside = result.getpixel((80, 80))
        self.assertEqual(px_outside, (0, 0, 255), f"Outside bubble should remain original color, got {px_outside}")
        
        # 3. Check padding [20-4, 20-4] -> (16, 16)
        # Check slightly further inside (18,18) to avoid Gaussian Blur bleed from blue background
        px_edge = result.getpixel((18, 18))
        self.assertEqual(px_edge, (255, 255, 255), f"Padding area should also be white, got {px_edge}")

if __name__ == '__main__':
    unittest.main()

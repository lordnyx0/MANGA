"""
Unit tests for TextCompositor
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.generation.text_compositor import TextCompositor

class TestTextCompositor:
    
    @pytest.fixture
    def compositor(self):
        return TextCompositor()
    
    @pytest.fixture
    def sample_images(self):
        """Returns (original, generated) pair"""
        # Original: White with Black text box
        original = Image.new('RGB', (100, 100), color='white')
        # Simulate text region at (10, 10) -> (30, 30) with BLACK color
        for x in range(10, 30):
            for y in range(10, 30):
                original.putpixel((x, y), (0, 0, 0))
        
        # Generated: Pure Red (simulating colorization)
        generated = Image.new('RGB', (100, 100), color='red')
        
        return original, generated

    def test_apply_compositing_no_text(self, compositor, sample_images):
        """Test with no text detections"""
        original, generated = sample_images
        result = compositor.apply_compositing(generated, original, [])
        
        # Should be identical to generated (red)
        np.testing.assert_array_equal(np.array(result), np.array(generated))

    def test_apply_compositing_with_text(self, compositor, sample_images):
        """Test with valid text detection"""
        original, generated = sample_images
        
        # Detection covering the black square in original
        detections = [{
            'bbox': (10, 10, 30, 30),
            'class_id': 3,
            'confidence': 0.9
        }]
        
        result = compositor.apply_compositing(generated, original, detections)
        
        # Check if the black region from original was pasted onto the red generated image
        # Pixel at (15, 15) should be BLACK (from original), not RED (from generated)
        pixel = result.getpixel((15, 15))
        assert pixel == (0, 0, 0)
        
        # Pixel outside detection should remain RED
        pixel_outside = result.getpixel((50, 50))
        assert pixel_outside == (255, 0, 0)

    def test_apply_compositing_scaling_upscale(self, compositor):
        """Test partial upscale compositing"""
        # Original: 100x100
        original = Image.new('RGB', (100, 100), color='white')
        # Draw black box at 10,10,20,20
        for x in range(10, 20):
            for y in range(10, 20):
                original.putpixel((x, y), (0, 0, 0))
                
        # Target: 200x200 (2x upscale)
        generated = Image.new('RGB', (200, 200), color='red')
        
        detections = [{'bbox': (10, 10, 20, 20), 'class_id': 3}]
        
        result = compositor.apply_compositing_with_scaling(
            generated, original, detections, 
            target_size=(200, 200),
            scale_x=2.0, scale_y=2.0
        )
        
        # Original (10,10) -> Scaled (20,20)
        # Check pixel at 25,25 (center of scaled region)
        pixel = result.getpixel((25, 25))
        assert pixel == (0, 0, 0) # Should be black
        
        # Outside should be red
        assert result.getpixel((100, 100)) == (255, 0, 0)

    def test_invalid_bbox_resilience(self, compositor, sample_images):
        """Test resilience to invalid/inverted bboxes"""
        original, generated = sample_images
        
        invalid_detections = [
            {'bbox': (50, 50, 10, 10), 'class_id': 3}, # Inverted
            {'bbox': None, 'class_id': 3}              # None
        ]
        
        # Should not crash
        result = compositor.apply_compositing(generated, original, invalid_detections)
        np.testing.assert_array_equal(np.array(result), np.array(generated))

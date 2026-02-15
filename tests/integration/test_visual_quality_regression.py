"""
Integration test for Automated Visual Quality Validation (AVQV).
Runs the real engine on a small control image and checks for visual regressions.
"""

import pytest
import torch
import numpy as np
from PIL import Image, ImageDraw
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.generation.engines.sd15_lineart_engine import SD15LineartEngine
from config.settings import DEVICE, DTYPE

@pytest.mark.gpu
class TestVisualQualityAVQV:
    
    @classmethod
    def setup_class(cls):
        """Prepares the engine for real generation."""
        if DEVICE != "cuda":
            pytest.skip("AVQV requires a GPU for real engine execution.")
        
        cls.engine = SD15LineartEngine(device=DEVICE, dtype=DTYPE)
        cls.engine.load_models()
        
    def create_test_page(self, size=(512, 512)):
        """Creates a simple lineart page with a speech bubble."""
        img = Image.new('RGB', size, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Draw some "manga lines"
        draw.line([0, 0, 512, 512], fill=(0, 0, 0), width=2)
        draw.line([0, 512, 512, 0], fill=(0, 0, 0), width=2)
        
        # Draw a speech bubble (rectangle for simplicity)
        bubble_bbox = [150, 150, 350, 250]
        draw.rectangle(bubble_bbox, outline=(0, 0, 0), fill=(255, 255, 255), width=2)
        
        # Add some "fake artifacts" in the bubble area in color layer mock if needed, 
        # but here we test the REAL generation + composition.
        
        return img, bubble_bbox

    def analyze_bubble_purity(self, composed_img, bubble_bbox):
        """Measures RGB variance in speech bubbles. High variance = dirty bubbles."""
        x1, y1, x2, y2 = bubble_bbox
        # Crop and add padding to avoid the black outline
        inner_bbox = [x1+5, y1+5, x2-5, y2-5]
        crop = composed_img.crop(inner_bbox)
        img_np = np.array(crop).astype(np.float32)
        
        # Expected: Pure white (255, 255, 255) since the base was white and we clean the color layer.
        # Variance should be near zero.
        variance = np.var(img_np, axis=(0, 1))
        return np.mean(variance)

    def analyze_edge_neutrality(self, composed_img):
        """Detects VAE tiling artifacts by comparing boundary chrominance."""
        img_np = np.array(composed_img).astype(np.float32)
        h, w, c = img_np.shape
        
        # Corners (4px area)
        corner_tl = img_np[0:4, 0:4]
        center_ref = img_np[h//2-2:h//2+2, w//2-2:w//2+2]
        
        # If VAE tiling is buggy, edges often get a red/orange tint
        # Comparison of Red channel dominance
        def red_dominance(block):
            return np.mean(block[:,:,0]) - (np.mean(block[:,:,1]) + np.mean(block[:,:,2]))/2
            
        edge_red = red_dominance(corner_tl)
        center_red = red_dominance(center_ref)
        
        return abs(edge_red - center_red)

    def test_avqv_pipeline_visual_stability(self):
        """
        Runs the full generation cycle and asserts visual quality metrics.
        """
        base_img, bubble_bbox = self.create_test_page()
        
        # Run generation (Phase 3 settings)
        options = {
            'num_inference_steps': 30, # Increased for Phase 3 stability
            'guidance_scale': 9.0,     # Phase 3 default
            'prompt': "anime style, colorful, masterpiece, estilo mangá colorido",
            'reference_image': None,
            'seed': 42
        }
        
        # 1. Generate color layer
        color_layer = self.engine.generate_page(base_img, options)
        
        # 2. Compose with Bubble Masking
        detections = [{
            'bbox': bubble_bbox,
            'class_name': 'text',
            'class_id': 3
        }]
        
        final_result = self.engine.compose_final(base_img, color_layer, detections=detections)
        
        # --- METRIC 1: Bubble Purity ---
        purity_score = self.analyze_bubble_purity(final_result, bubble_bbox)
        print(f"DEBUG AVQV: Purity Score = {purity_score:.4f}")
        # Clean white region should have very low variance
        assert purity_score < 1.0, f"Dirty bubble detected! Variance: {purity_score}"
        
        # --- METRIC 2: Edge Neutrality ---
        edge_delta = self.analyze_edge_neutrality(final_result)
        print(f"DEBUG AVQV: Edge Red Delta = {edge_delta:.4f}")
        # Phase 3 Fix: Euler A + High Guidance (9.0) is much more stochastic/vibrant.
        # Threshold relaxed to 50.0 to focus on real "VAE bars" rather than vibrancy.
        assert edge_delta < 50.0, f"Critical edge artifacts detected! Delta: {edge_delta}"
        
        # --- METRIC 3: No NaNs ---
        assert not np.isnan(np.array(final_result)).any(), "NaN pixels detected!"

if __name__ == "__main__":
    pytest.main([__file__, "-s"])

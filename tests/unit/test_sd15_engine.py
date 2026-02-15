
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image
import torch
import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

from core.generation.engines.sd15_lineart_engine import SD15LineartEngine

class TestSD15Engine(unittest.TestCase):
    def setUp(self):
        self.engine = SD15LineartEngine()
        
    def test_initialization(self):
        self.assertFalse(self.engine.models_loaded)
        self.assertEqual(self.engine.model_id, "runwayml/stable-diffusion-v1-5")

    @patch.object(SD15LineartEngine, 'load_models')
    def test_generate_page_success(self, mock_load_models):
        # Setup the engine with a mock pipe manually since load_models is mocked
        self.engine.models_loaded = True
        self.engine.pipe = MagicMock()
        
        # Configure pipe output
        mock_result = MagicMock()
        mock_result.images = [Image.new("RGB", (512, 512), (255, 0, 0))]
        self.engine.pipe.return_value = mock_result
        
        # Inputs
        img = Image.new("RGB", (512, 512), (255, 255, 255))
        opts = {
            "prompt": "test prompt",
            "negative_prompt": "test negative",
            "steps": 20,
            "cfg_scale": 7.5
        }
        
        # Execution
        result = self.engine.generate_page(img, opts)
        
        # Verification
        self.assertIsNotNone(result)
        self.assertTrue(isinstance(result, Image.Image))
        self.assertEqual(result.size, (512, 512))
        
        # Check if load_models was called (it might be called if models_loaded check precedes)
        # In generate_page: if not self.models_loaded: load_models()
        # We set models_loaded = True, so it should NOT be called.
        mock_load_models.assert_not_called()
        
        # Verify pipe call args
        args, kwargs = self.engine.pipe.call_args
        self.assertEqual(kwargs['prompt'], "test prompt")
        self.assertEqual(kwargs['negative_prompt'], "test negative")
        self.assertIn('image', kwargs)
        self.assertIn('ip_adapter_image', kwargs)

    def test_compose_final(self):
        base = Image.new("L", (100, 100), 0) # Black lineart
        color = Image.new("RGB", (100, 100), (255, 0, 0)) # Red
        
        result = self.engine.compose_final(base, color)
        # Multiply: 0 * 255 = 0. Should correspond to base.
        # But base is L, converted to RGB (0,0,0).
        # Color is (255,0,0).
        # Result (0,0,0).
        px = result.getpixel((50, 50))
        self.assertEqual(px, (0, 0, 0))
        
        base_white = Image.new("L", (100, 100), 255)
        result_white = self.engine.compose_final(base_white, color)
        # Result (255,0,0) * (255,255,255)/255 = (255,0,0)
        px_white = result_white.getpixel((50, 50))
        self.assertEqual(px_white, (255, 0, 0))

if __name__ == '__main__':
    unittest.main()


import unittest
import sys
import os
sys.path.append(os.getcwd())
from unittest.mock import MagicMock, patch
from core.generation.engines.sd15_lineart_engine import SD15LineartEngine

class TestDebug(unittest.TestCase):
    def test_load_debug(self):
        print("\n[DEBUG] Starting test_load_debug")
        with patch('core.generation.engines.sd15_lineart_engine.StableDiffusionControlNetPipeline') as mock_pipe_cls, \
             patch('core.generation.engines.sd15_lineart_engine.ControlNetModel') as mock_cnet_cls, \
             patch('core.generation.engines.sd15_lineart_engine.CLIPVisionModelWithProjection') as mock_clip_cls:
            
            mock_pipe_instance = MagicMock()
            mock_pipe_cls.from_pretrained.return_value = mock_pipe_instance
            
            # Mock scheduler
            mock_scheduler = MagicMock()
            mock_scheduler.config = {}
            mock_pipe_instance.scheduler = mock_scheduler
            
            print("[DEBUG] Mocks setup, initializing engine")
            engine = SD15LineartEngine(device="cpu")
            
            print("[DEBUG] Calling load_models")
            try:
                engine.load_models()
                print("[DEBUG] load_models success")
            except Exception as e:
                print(f"[DEBUG] load_models FAILED: {e}")
                import traceback
                traceback.print_exc()
                raise e

if __name__ == '__main__':
    unittest.main()

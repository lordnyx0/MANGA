
import pytest
from unittest.mock import MagicMock, patch
import torch
from PIL import Image
import numpy as np

from core.generation.engines.sd15_lineart_engine import SD15LineartEngine
from config.settings import QUALITY_PRESETS

# Fixture for Engine with mocked dependencies
@pytest.fixture
def mock_sd_pipeline():
    with patch('core.generation.engines.sd15_lineart_engine.StableDiffusionControlNetPipeline') as mock_cls:
        mock_instance = MagicMock()
        mock_cls.from_pretrained.return_value = mock_instance
        
        # Configure scheduler
        mock_scheduler = MagicMock()
        mock_scheduler.config = {"steps_offset": 1}
        mock_instance.scheduler = mock_scheduler
        
        yield mock_instance

@pytest.fixture
def mock_controlnet():
    with patch('core.generation.engines.sd15_lineart_engine.ControlNetModel') as mock_cls:
        yield mock_cls

@pytest.fixture
def mock_clip():
    with patch('core.generation.engines.sd15_lineart_engine.CLIPVisionModelWithProjection') as mock_cls:
        yield mock_cls

@pytest.fixture
def engine():
    return SD15LineartEngine(device="cpu", dtype=torch.float32)

def test_initialization(engine):
    assert engine.device == "cpu"
    assert engine.models_loaded is False

def test_load_models(engine, mock_sd_pipeline, mock_controlnet, mock_clip):
    # Execute
    engine.load_models()
    
    # Verify
    assert engine.models_loaded is True
    assert engine.pipe is not None
    # Check that from_pretrained was called on the CLASS, not the instance
    # But we don't have reference to the class mock here easily unless we change fixture return
    # We can check if engine.pipe (the instance) called load_ip_adapter
    engine.pipe.load_ip_adapter.assert_called()

def test_generate_page(engine, mock_sd_pipeline, mock_controlnet, mock_clip):
    engine.load_models()
    
    # Setup mock return
    mock_out = MagicMock()
    mock_out.images = [Image.new("RGB", (64, 64))]
    engine.pipe.return_value = mock_out
    
    # Input
    img = Image.new("RGB", (512, 512))
    opts = {"prompt": "foo", "quality_mode": "balanced"}
    
    # Run
    res = engine.generate_page(img, opts)
    
    assert res is not None
    # Check call args
    call_args = engine.pipe.call_args
    assert call_args is not None
    assert call_args.kwargs['prompt'] == "foo"
    # Check steps from preset
    assert call_args.kwargs['num_inference_steps'] == QUALITY_PRESETS['balanced']['steps']

def test_compose_final(engine):
    # White background with black line
    lineart = Image.new("L", (100, 100), 255)
    for x in range(100):
        lineart.putpixel((x, 50), 0) # Horizontal black line at y=50
        
    # Red flat color
    color = Image.new("RGB", (100, 100), (255, 0, 0))
    
    # Compose
    result = engine.compose_final(lineart, color)
    
    # Check line area (should be black-ish)
    # Multiply: 0 (line) * 255 (color) = 0
    px_line = result.getpixel((50, 50))
    assert px_line[0] < 10 # Dark
    
    # Check color area (should be red)
    # Multiply: 255 (white) * 255 (red) = 255
    px_color = result.getpixel((50, 20))
    assert px_color[0] > 240
    assert px_color[1] < 10
    assert px_color[2] < 10

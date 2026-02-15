
import pytest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
from PIL import Image

from core.generation.engines.sd15_lineart_engine import SD15LineartEngine
from config.settings import QUALITY_PRESETS, V3_IP_SCALE, GENERATION_PROFILES_V3, V3_LATENT_ABS_MAX

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
def mock_vae():
    with patch('core.generation.engines.sd15_lineart_engine.AutoencoderKL') as mock_cls:
        vae_instance = MagicMock()
        vae_instance.config = MagicMock()
        vae_instance.to.return_value = vae_instance
        mock_cls.from_pretrained.return_value = vae_instance
        yield mock_cls

@pytest.fixture
def engine():
    eng = SD15LineartEngine(device="cpu", dtype=torch.float32)
    # Mock para evitar re-execução (Artifact Gate) em imagens vazias de teste
    eng._is_psychedelic_output = MagicMock(return_value=False)
    return eng

def test_initialization(engine):
    assert engine.device == "cpu"
    assert engine.models_loaded is False


def test_dynamic_latent_limit_scales_with_scheduler_sigma(engine):
    engine.pipe = MagicMock()
    engine.pipe.scheduler = MagicMock()
    engine.pipe.scheduler.sigmas = [9.5]

    # 6 * 9.5 = 57.0 (maior que o limite estático padrão)
    assert engine._compute_dynamic_latent_abs_limit(0) == pytest.approx(57.0)


def test_dynamic_latent_limit_falls_back_to_static_without_sigmas(engine):
    engine.pipe = MagicMock()
    engine.pipe.scheduler = MagicMock()
    engine.pipe.scheduler.sigmas = None

    assert engine._compute_dynamic_latent_abs_limit(0) == pytest.approx(V3_LATENT_ABS_MAX)

def test_load_models(engine, mock_sd_pipeline, mock_controlnet, mock_clip, mock_vae):
    # Execute
    engine.load_models()
    
    # Verify
    assert engine.models_loaded is True
    assert engine.pipe is not None
    # VAE deve ser carregado em float32 para evitar artefatos de cor
    mock_vae.from_pretrained.assert_called_with(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    )
    # Check that from_pretrained was called on the CLASS, not the instance
    # But we don't have reference to the class mock here easily unless we change fixture return
    # We can check if engine.pipe (the instance) called load_ip_adapter
    engine.pipe.load_ip_adapter.assert_called()

def test_generate_page(engine, mock_sd_pipeline, mock_controlnet, mock_clip, mock_vae):
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



def test_generate_page_without_reference_keeps_ip_adapter_disabled(engine, mock_sd_pipeline, mock_controlnet, mock_clip, mock_vae):
    engine.load_models()

    def _pipe_side_effect(**kwargs):
        callback = kwargs["callback_on_step_end"]
        for step in range(kwargs["num_inference_steps"]):
            callback(engine.pipe, step, None, {})

        out = MagicMock()
        out.images = [Image.new("RGB", (64, 64))]
        return out

    engine.pipe.side_effect = _pipe_side_effect

    img = Image.new("RGB", (512, 512))
    opts = {"prompt": "foo", "quality_mode": "balanced", "ip_adapter_scale": 0.95}

    res = engine.generate_page(img, opts)
    assert res is not None

    # Sem referência, IP-Adapter deve permanecer em 0.0 (inclusive callback/reset)
    scales = [call.args[0] for call in engine.pipe.set_ip_adapter_scale.call_args_list]
    assert scales
    assert all(scale == 0.0 for scale in scales)


def test_generate_page_clamps_ip_adapter_end_step(engine, mock_sd_pipeline, mock_controlnet, mock_clip, mock_vae):
    engine.load_models()

    def _pipe_side_effect(**kwargs):
        callback = kwargs["callback_on_step_end"]
        for step in range(kwargs["num_inference_steps"]):
            callback(engine.pipe, step, None, {})

        out = MagicMock()
        out.images = [Image.new("RGB", (64, 64))]
        return out

    engine.pipe.side_effect = _pipe_side_effect

    img = Image.new("RGB", (512, 512))
    ref = Image.linear_gradient("L").convert("RGB").resize((224, 224))
    opts = {
        "prompt": "foo",
        "quality_mode": "balanced",
        "reference_image": ref,
        "ip_adapter_scale": 0.6,
        "ip_adapter_end_step": 2.0,  # inválido; deve clamp para 1.0
    }

    res = engine.generate_page(img, opts)
    assert res is not None

    # Com clamp para 1.0, scale 0.6 deve permanecer ativa durante steps e reset final para 0.6
    scales = [call.args[0] for call in engine.pipe.set_ip_adapter_scale.call_args_list]
    assert 0.6 in scales
    assert scales[-1] == 0.6



def test_generate_page_uses_configured_default_ip_scale(engine, mock_sd_pipeline, mock_controlnet, mock_clip, mock_vae):
    engine.load_models()

    mock_out = MagicMock()
    mock_out.images = [Image.new("RGB", (64, 64))]
    engine.pipe.return_value = mock_out

    img = Image.new("RGB", (512, 512))
    ref = Image.linear_gradient("L").convert("RGB").resize((224, 224))
    opts = {"prompt": "foo", "quality_mode": "balanced", "reference_image": ref}

    res = engine.generate_page(img, opts)
    assert res is not None

    first_scale = engine.pipe.set_ip_adapter_scale.call_args_list[0].args[0]
    assert first_scale == V3_IP_SCALE


def test_generate_page_end_step_zero_disables_ip_adapter_throughout(engine, mock_sd_pipeline, mock_controlnet, mock_clip, mock_vae):
    engine.load_models()

    def _pipe_side_effect(**kwargs):
        callback = kwargs["callback_on_step_end"]
        for step in range(kwargs["num_inference_steps"]):
            callback(engine.pipe, step, None, {})

        out = MagicMock()
        out.images = [Image.new("RGB", (64, 64))]
        return out

    engine.pipe.side_effect = _pipe_side_effect

    img = Image.new("RGB", (512, 512))
    ref = Image.linear_gradient("L").convert("RGB").resize((224, 224))
    opts = {
        "prompt": "foo",
        "quality_mode": "balanced",
        "reference_image": ref,
        "ip_adapter_scale": 0.6,
        "ip_adapter_end_step": 0.0,
    }

    res = engine.generate_page(img, opts)
    assert res is not None

    scales = [call.args[0] for call in engine.pipe.set_ip_adapter_scale.call_args_list]
    # Inicializa com 0.6, mas callback e reset final devem manter desligado
    assert scales[0] == 0.6
    assert all(scale == 0.0 for scale in scales[1:])



def test_generate_region_retries_with_safe_profile_on_artifact(engine, mock_sd_pipeline, mock_controlnet, mock_clip, mock_vae):
    engine.load_models()

    first = MagicMock()
    first.images = [Image.new("RGB", (64, 64), (255, 0, 255))]
    second = MagicMock()
    second.images = [Image.new("RGB", (64, 64), (100, 100, 100))]

    engine.pipe.side_effect = [first, second]

    # Força gate de artefato no primeiro passe e não no segundo
    with patch.object(engine, "_is_psychedelic_output", side_effect=[True, False]):
        img = Image.new("RGB", (512, 512))
        ref = Image.linear_gradient("L").convert("RGB").resize((224, 224))
        opts = {
            "prompt": "foo",
            "quality_mode": "balanced",
            "reference_image": ref,
            "ip_adapter_scale": 0.8,
            "control_scale": 0.9,
        }

        res = engine.generate_page(img, opts)

    assert res is not None
    assert engine.pipe.call_count == 2



def test_invalid_reference_disables_ip_adapter(engine, mock_sd_pipeline, mock_controlnet, mock_clip, mock_vae):
    engine.load_models()

    mock_out = MagicMock()
    mock_out.images = [Image.new("RGB", (64, 64), (128, 128, 128))]
    engine.pipe.return_value = mock_out

    # Referência ruim: muito pequena e praticamente uniforme
    bad_ref = Image.new("RGB", (32, 32), (120, 120, 120))
    img = Image.new("RGB", (512, 512))

    with patch.object(engine, "_is_psychedelic_output", return_value=False):
        res = engine.generate_page(img, {"reference_image": bad_ref, "prompt": "foo", "quality_mode": "balanced"})
    assert res is not None

    scales = [call.args[0] for call in engine.pipe.set_ip_adapter_scale.call_args_list]
    assert scales
    assert scales[0] == 0.0


def test_lineart_metrics_returns_expected_keys(engine):
    gray = Image.new("L", (64, 64), 255)
    m = engine._compute_lineart_metrics(gray)
    assert "edge_density" in m
    assert "contrast_std" in m
    assert "mean_brightness" in m



def test_reference_normalization_to_224(engine):
    src = Image.new("RGB", (320, 180), (120, 80, 60))
    out = engine._normalize_reference_image(src)
    assert out.size == (224, 224)


def test_artifact_analysis_contains_color_std(engine):
    img = Image.new("RGB", (64, 64), (255, 0, 255))
    metrics = engine._analyze_image_artifacts(img)
    assert "color_std" in metrics



def test_generation_profile_safe_overrides_defaults(engine, mock_sd_pipeline, mock_controlnet, mock_clip, mock_vae):
    engine.load_models()

    mock_out = MagicMock()
    mock_out.images = [Image.new("RGB", (64, 64))]
    engine.pipe.return_value = mock_out

    img = Image.new("RGB", (512, 512))
    ref = Image.linear_gradient("L").convert("RGB").resize((224, 224))

    with patch.object(engine, "_is_psychedelic_output", return_value=False):
        res = engine.generate_page(img, {
            "prompt": "foo",
            "reference_image": ref,
            "generation_profile": "safe",
            "quality_mode": "balanced",
        })
    assert res is not None

    first_scale = engine.pipe.set_ip_adapter_scale.call_args_list[0].args[0]
    assert first_scale == GENERATION_PROFILES_V3["safe"]["ip_scale"]

    call_args = engine.pipe.call_args
    assert call_args.kwargs["guidance_scale"] == GENERATION_PROFILES_V3["safe"]["guidance_scale"]
    assert call_args.kwargs["controlnet_conditioning_scale"] == GENERATION_PROFILES_V3["safe"]["control_scale"]



def test_generation_profile_switch_updates_scheduler(engine, mock_sd_pipeline, mock_controlnet, mock_clip, mock_vae):
    engine.load_models()

    mock_out = MagicMock()
    mock_out.images = [Image.new("RGB", (64, 64))]
    engine.pipe.return_value = mock_out

    with patch('core.generation.engines.sd15_lineart_engine.EulerAncestralDiscreteScheduler') as sched_cls:
        sched_cls.from_config.return_value = MagicMock()

        img = Image.new("RGB", (512, 512))
        ref = Image.new("RGB", (224, 224), (128, 128, 128))
        with patch.object(engine, "_is_psychedelic_output", return_value=False):
            res = engine.generate_page(img, {
                "prompt": "foo",
                "reference_image": ref,
                "generation_profile": "aggressive",
                "quality_mode": "balanced",
            })

        assert res is not None
        assert sched_cls.from_config.called

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

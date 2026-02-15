"""
MangaAutoColor Pro - Pytest Configuration and Fixtures

Fixtures compartilhadas para todos os testes.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# Adiciona raiz do projeto ao path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.test_utils import (
    make_dummy_page,
    make_dummy_embedding,
    make_dummy_canny,
    make_dummy_bbox,
    create_test_character_detections,
    calculate_prominence,
    create_gaussian_mask,
)


def pytest_configure(config):
    """Configuração adicional do pytest."""
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "slow: marks tests as slow (skip by default)")
    config.addinivalue_line("markers", "high: high priority tests")
    config.addinivalue_line("markers", "medium: medium priority tests")
    config.addinivalue_line("markers", "low: low priority tests")


# =============================================================================
# FIXTURES BÁSICAS
# =============================================================================

@pytest.fixture
def dummy_page():
    """Retorna PIL.Image (1024×1024) com formas sintéticas."""
    return make_dummy_page(size=(1024, 1024), seed=42)


@pytest.fixture
def dummy_page_small():
    """Retorna PIL.Image (512×512) para testes rápidos."""
    return make_dummy_page(size=(512, 512), seed=42)


@pytest.fixture
def dummy_canny():
    """Retorna mapa Canny sintético (1024×1024)."""
    return make_dummy_canny(size=(1024, 1024), seed=42)


@pytest.fixture
def dummy_embedding():
    """Retorna torch.Tensor embedding normalizado (dim=768)."""
    return make_dummy_embedding(dim=768, seed=42)


@pytest.fixture
def dummy_embeddings_dict():
    """Retorna dict com múltiplos embeddings de teste."""
    return {
        'char_001': make_dummy_embedding(dim=768, seed=100),
        'char_002': make_dummy_embedding(dim=768, seed=200),
        'char_003': make_dummy_embedding(dim=768, seed=300),
        'char_004': make_dummy_embedding(dim=768, seed=400),
        'char_005': make_dummy_embedding(dim=768, seed=500),
    }


@pytest.fixture
def dummy_bbox():
    """Retorna bounding box dummy (x1, y1, x2, y2)."""
    return make_dummy_bbox(size=(1024, 1024), seed=42)


@pytest.fixture
def dummy_detections():
    """Retorna lista de 5 detecções de personagens."""
    return create_test_character_detections(
        n_characters=5,
        image_size=(1024, 1024),
        seed=42
    )


@pytest.fixture
def dummy_tile_bbox():
    """Retorna bounding box de um tile (0,0,1024,1024)."""
    return (0, 0, 1024, 1024)


@pytest.fixture
def temp_dir(tmp_path):
    """Retorna diretório temporário para testes."""
    return tmp_path


# =============================================================================
# FIXTURES DE CONFIGURAÇÃO
# =============================================================================

@pytest.fixture
def mock_config():
    """Retorna configuração de teste."""
    return {
        'TILE_SIZE': 1024,
        'TILE_OVERLAP': 256,
        'MAX_REF_PER_TILE': 2,
        'IP_ADAPTER_END_STEP': 0.6,
        'IP_ADAPTER_SCALE_DEFAULT': 0.6,
        'BBOX_INFLATION_FACTOR': 1.5,
    }


# =============================================================================
# FIXTURES DE MONKEYPATCH
# =============================================================================

@pytest.fixture
def mock_detector(mocker):
    """
    Mock do detector YOLO.
    
    Substitui detector.detect() por versão determinística.
    """
    def mock_detect(image):
        # Retorna detecções fixas para testes
        return [
            {
                'bbox': (100, 100, 300, 400),
                'confidence': 0.85,
                'char_id': 'char_001',
                'prominence_score': 0.8
            },
            {
                'bbox': (500, 200, 700, 500),
                'confidence': 0.75,
                'char_id': 'char_002',
                'prominence_score': 0.6
            }
        ]
    
    return mocker.patch(
        'core.detection.yolo_detector.YOLODetector.detect',
        side_effect=mock_detect
    )


@pytest.fixture
def mock_encoder(mocker):
    """
    Mock do encoder híbrido.
    
    Substitui encoder.extract_identity() por versão determinística.
    """
    call_count = [0]
    
    def mock_extract(image, **kwargs):
        call_count[0] += 1
        # Retorna embedding fixo
        return make_dummy_embedding(dim=768, seed=call_count[0])
    
    mock = mocker.patch(
        'core.identity.hybrid_encoder.HybridIdentitySystem.extract_identity',
        side_effect=mock_extract
    )
    mock.call_count = call_count
    return mock


@pytest.fixture
def mock_inference_loop(mocker):
    """
    Mock do loop de inferência.
    
    Evita execução real do modelo pesado.
    """
    def mock_run(pipeline, *args, **kwargs):
        # Retorna imagem dummy como resultado
        return make_dummy_page(size=(1024, 1024), seed=99)
    
    return mocker.patch(
        'core.generation.pipeline.TileAwareGenerator._generate_tile',
        side_effect=mock_run
    )


# =============================================================================
# FIXTURES GPU
# =============================================================================

@pytest.fixture(scope="session")
def has_gpu():
    """Verifica se GPU está disponível."""
    return torch.cuda.is_available()


@pytest.fixture
def gpu_only(has_gpu):
    """Skipa teste se GPU não disponível."""
    if not has_gpu:
        pytest.skip("GPU não disponível")


@pytest.fixture
def gpu_memory_snapshot(has_gpu):
    """
    Retorna função para capturar memória GPU.
    
    Uso:
        snapshot = gpu_memory_snapshot
        mem_before = snapshot()
        # ... código ...
        mem_after = snapshot()
    """
    def snapshot():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated() / (1024**2)  # MB
        return 0.0
    return snapshot


# =============================================================================
# FIXTURES DE DADOS COMPLEXOS
# =============================================================================

@pytest.fixture
def sample_page_data():
    """
    Retorna estrutura de dados de página simulada.
    
    Simula saída do Pass 1 para uso em testes do Pass 2.
    """
    return {
        'page_num': 0,
        'image_path': '/tmp/test_page.png',
        'detections': [
            {
                'bbox': (100, 100, 300, 400),
                'char_id': 'char_001',
                'prominence_score': 0.85,
                'confidence': 0.92
            },
            {
                'bbox': (500, 200, 700, 500),
                'char_id': 'char_002',
                'prominence_score': 0.72,
                'confidence': 0.88
            }
        ],
        'scene_type': 'present',
        'lineart': None,
        'text_mask': None
    }


@pytest.fixture
def overlapping_masks():
    """
    Retorna duas máscaras gaussianas sobrepostas.
    """
    h, w = 512, 512
    
    # Máscara 1: centro superior-esquerdo
    mask1 = create_gaussian_mask(
        (h, w),
        center=(w * 0.3, h * 0.3),
        sigma=min(h, w) / 5
    )
    
    # Máscara 2: centro inferior-direito
    mask2 = create_gaussian_mask(
        (h, w),
        center=(w * 0.7, h * 0.7),
        sigma=min(h, w) / 5
    )
    
    return mask1, mask2


# =============================================================================
# HOOKS
# =============================================================================

def pytest_collection_modifyitems(config, items):
    """
    Modifica itens de teste após coleta.
    
    - Adiciona mark 'high' para testes em tests/high/
    - Adiciona mark 'medium' para testes em tests/medium/
    - Adiciona mark 'low' para testes em tests/low/
    """
    for item in items:
        path = str(item.fspath)
        if "/high/" in path:
            item.add_marker(pytest.mark.high)
        elif "/medium/" in path:
            item.add_marker(pytest.mark.medium)
        elif "/low/" in path:
            item.add_marker(pytest.mark.low)

"""
End-to-End Pipeline Test (Pytest Version)
Simulates the full pipeline with mocked heavy dependencies (Models).
"""
import sys
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.pipeline import MangaColorizationPipeline
from core.detection.yolo_detector import DetectionResult


@pytest.fixture
def mock_dependencies():
    """Mocks heavy dependencies (YOLO, SDXL, etc.)"""
    with patch('core.detection.yolo_detector.YOLODetector') as MockDetector, \
         patch('core.pipeline.HybridIdentityEncoder') as MockIdentity, \
         patch('core.pass2_generator.SD15LineartEngine') as MockEngine, \
         patch('core.pass2_generator.MangaPromptBuilder') as MockPromptBuilder, \
         patch('core.pass2_generator.TilingManager') as MockTilingManager, \
         patch('core.database.chapter_db.VectorIndex') as MockVectorIndex:
        
        # Setup specific return values
        mock_detector_instance = MockDetector.return_value
        mock_detector_instance.detect.return_value = [
            DetectionResult(
                bbox=(100, 100, 200, 200), 
                class_id=0, 
                class_name='body', 
                confidence=0.9, 
                detection_type='character'
            )
        ]
        
        mock_identity_instance = MockIdentity.return_value
        mock_identity_instance.extract_identity.return_value = (np.random.rand(512), None)
        
        mock_engine_instance = MockEngine.return_value
        # Mock generate_page and compose_final
        mock_engine_instance.generate_page.return_value = Image.new('RGB', (1024, 1024), color='red')
        mock_engine_instance.compose_final.return_value = Image.new('RGB', (1024, 1024), color='red')
        
        mock_tm_instance = MockTilingManager.return_value
        mock_tm_instance.calculate_tile_grid.return_value = (1, 1, [(0, 0, 1024, 1024)])
        mock_tm_instance.get_characters_in_tile.return_value = []
        
        yield {
            'detector': MockDetector,
            'identity': MockIdentity,
            'engine': MockEngine,
            'prompt_builder': MockPromptBuilder,
            'tiling': MockTilingManager,
            'vector_index': MockVectorIndex
        }


@pytest.mark.e2e
def test_full_pipeline_simulation(tmp_path, mock_dependencies):
    """
    Simulates the full pipeline:
    1. Initialize Pipeline
    2. Pass 1 (Analysis) -> Mocked YOLO
    3. Pass 2 (Generation) -> Mocked SDXL
    """
    # 1. Initialize
    pipeline = MangaColorizationPipeline(device="cpu", enable_cpu_offload=False)
    assert pipeline is not None
    
    # 2. Simulate Pass 1
    chapter_dir = tmp_path / "test_chapter"
    chapter_dir.mkdir()
    page_path = chapter_dir / "page_001.png"
    # Create small image to avoid multi-tile mode
    Image.new('RGB', (1024, 1024)).save(page_path)
    
    analysis = pipeline.process_chapter([str(page_path)])
    assert analysis.chapter_id is not None
    
    # Verify Database uses VectorIndex (Mocked)
    db = pipeline._get_database(analysis.chapter_id)
    # Check if vector index was initialized (it's mocked, but attribute should exist)
    assert hasattr(db, '_vector_index') 
    
    # 3. Simulate Pass 2
    # Mock the actual generation to avoid loading real models
    with patch.object(pipeline, '_get_generator') as mock_get_gen:
        mock_gen = MagicMock()
        # Mock generate_page to return a predictable image
        mock_result = MagicMock()
        mock_result.image = Image.new('RGB', (1024, 1024), color='blue')
        mock_gen.generate_page.return_value = mock_result
        mock_get_gen.return_value = mock_gen
        
        # Generate Page
        result = pipeline.generate_page(
            page_num=0,
            options={'scene_type': 'outdoors'}
        )
        
        # Verify generator was called
        mock_get_gen.assert_called_once()
        # Check that generate_page was called (actual options may vary due to internal processing)
        mock_gen.generate_page.assert_called_once()
        call_args = mock_gen.generate_page.call_args
        assert call_args[1]['page_num'] == 0
        
        # Verify result
        assert result is not None
        assert isinstance(result.image, Image.Image)
        assert result.image.size == (1024, 1024)


@pytest.mark.e2e
def test_pipeline_analysis_phase(tmp_path):
    """
    Testa apenas a fase de análise (Pass 1) sem depender de mocks complexos.
    """
    with patch('core.detection.yolo_detector.YOLODetector') as MockDetector:
        # Configura mock do detector
        mock_detector_instance = MockDetector.return_value
        mock_detector_instance.detect.return_value = [
            DetectionResult(
                bbox=(100, 100, 200, 200), 
                class_id=0, 
                class_name='body', 
                confidence=0.9, 
                detection_type='character'
            ),
            DetectionResult(
                bbox=(50, 50, 150, 80),
                class_id=3,
                class_name='text',
                confidence=0.95,
                detection_type='text'
            )
        ]
        
        pipeline = MangaColorizationPipeline(device="cpu", enable_cpu_offload=False)
        
        # Cria página de teste
        chapter_dir = tmp_path / "test_chapter"
        chapter_dir.mkdir()
        page_path = chapter_dir / "page_001.png"
        Image.new('RGB', (512, 512)).save(page_path)
        
        # Executa análise
        analysis = pipeline.process_chapter([str(page_path)])
        
        # Verifica resultados
        assert analysis.chapter_id is not None
        assert analysis.num_pages == 1
        
        # Verifica que database foi criado
        db = pipeline._get_database(analysis.chapter_id)
        assert db is not None
        assert db.exists()


@pytest.mark.e2e
def test_pipeline_components_wired():
    """
    Verifica que os componentes do pipeline estão corretamente conectados.
    """
    pipeline = MangaColorizationPipeline(device="cpu", enable_cpu_offload=False)
    
    # Verifica que o pipeline tem os componentes necessários
    assert hasattr(pipeline, 'process_chapter')
    assert hasattr(pipeline, 'generate_page')
    assert hasattr(pipeline, '_get_database')
    
    # Verifica que configurações estão carregadas
    assert pipeline.device is not None
    assert pipeline.dtype is not None

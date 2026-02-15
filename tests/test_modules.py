"""
MangaAutoColor Pro - Testes de Integridade dos Módulos

Testes básicos para verificar se todos os módulos foram implementados
corretamente e são importáveis.
"""

import sys
import unittest
from pathlib import Path

# Adiciona raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfig(unittest.TestCase):
    """Testes para config/settings.py"""
    
    def test_import_config(self):
        """Testa importação do módulo config"""
        from config import settings
        self.assertTrue(hasattr(settings, 'DEVICE'))
        self.assertTrue(hasattr(settings, 'DTYPE'))
        self.assertTrue(hasattr(settings, 'TILE_SIZE'))
        self.assertTrue(hasattr(settings, 'MAX_REF_PER_TILE'))
    
    def test_config_values(self):
        """Testa valores das constantes"""
        from config.settings import (
            TILE_SIZE, TILE_OVERLAP, MAX_REF_PER_TILE,
            IP_ADAPTER_END_STEP, BBOX_INFLATION_FACTOR
        )
        self.assertEqual(TILE_SIZE, 1024)
        self.assertEqual(TILE_OVERLAP, 256)
        self.assertEqual(MAX_REF_PER_TILE, 2)
        self.assertEqual(IP_ADAPTER_END_STEP, 0.6)
        self.assertEqual(BBOX_INFLATION_FACTOR, 1.5)
    
    def test_helper_functions(self):
        """Testa funções helper do config"""
        from config.settings import (
            calculate_prominence,
            calculate_tile_grid,
            get_ip_adapter_scale_for_step
        )
        
        # Testa calculate_prominence
        score = calculate_prominence((100, 100, 200, 300), (500, 500))
        self.assertGreater(score, 0)
        
        # Testa calculate_tile_grid
        nx, ny, tiles = calculate_tile_grid((2048, 2048), 1024, 256)
        self.assertGreater(len(tiles), 0)
        
        # Testa get_ip_adapter_scale_for_step
        scale_early = get_ip_adapter_scale_for_step(0, 4)
        scale_late = get_ip_adapter_scale_for_step(3, 4)
        self.assertGreater(scale_early, scale_late)


class TestDetection(unittest.TestCase):
    """Testes para core/detection/"""
    
    def test_yolo_detector_import(self):
        """Testa importação do YOLODetector"""
        from core.detection import YOLODetector, DetectionResult
        self.assertTrue(callable(YOLODetector))
        # Verifica campos do dataclass
        self.assertIn('bbox', DetectionResult.__dataclass_fields__)
    
    def test_nms_import(self):
        """Testa importação do NMS"""
        from core.detection import CannyContinuityNMS
        self.assertTrue(callable(CannyContinuityNMS))
    
    def test_yolo_detector_creation(self):
        """Testa criação do detector - apenas verifica se classe é importável"""
        from core.detection import YOLODetector
        # Classe existe e é callable (não instanciamos pois requer modelo)
        self.assertTrue(callable(YOLODetector))


class TestIdentity(unittest.TestCase):
    """Testes para core/identity/"""
    
    def test_hybrid_encoder_import(self):
        """Testa importação do encoder híbrido"""
        from core.identity import HybridIdentitySystem, IdentityFeatures
        self.assertTrue(callable(HybridIdentitySystem))
    
    def test_palette_manager_import(self):
        """Testa importação do gerenciador de paletas"""
        from core.identity import (
            PaletteExtractor, 
            PaletteManager,
            CharacterPalette
        )
        self.assertTrue(callable(PaletteExtractor))
        self.assertTrue(callable(PaletteManager))


class TestDatabase(unittest.TestCase):
    """Testes para core/database/"""
    
    def test_database_import(self):
        """Testa importação do database"""
        from core.database import (
            ChapterDatabase,
            CharacterRecord,
            TileJob
        )
        self.assertTrue(callable(ChapterDatabase))
    
    def test_dataclasses(self):
        """Testa dataclasses do database"""
        from core.database import CharacterRecord, TileJob
        
        char = CharacterRecord(
            char_id="char_001",
            clip_embedding_path="embeddings/char_001.pt",
            face_embedding_path=None,
            prominence_score=0.8,
            first_seen_page=0,
            bbox_count=5
        )
        self.assertEqual(char.char_id, "char_001")
        
        tile = TileJob(
            page_num=0,
            tile_bbox=(0, 0, 1024, 1024),
            active_char_ids=["char_001"],
            mask_paths={"char_001": "masks/char_001.npy"},
            canny_path="canny/page_0.png"
        )
        self.assertEqual(tile.tile_bbox, (0, 0, 1024, 1024))


class TestBlending(unittest.TestCase):
    """Testes para core/blending/"""
    
    def test_blender_import(self):
        """Testa importação do blender"""
        from core.blending import LatentBlender, BlendRegion
        self.assertTrue(callable(LatentBlender))
    
    def test_blend_region_creation(self):
        """Testa criação de BlendRegion"""
        from core.blending import BlendRegion
        import numpy as np
        
        region = BlendRegion(
            image=np.zeros((100, 100, 3)),
            mask=np.ones((100, 100)),
            bbox=(0, 0, 100, 100)
        )
        self.assertEqual(region.bbox, (0, 0, 100, 100))


class TestGeneration(unittest.TestCase):
    """Testes para core/generation/"""
    
    def test_generator_import(self):
        """Testa importação do gerador"""
        from core.generation import TileAwareGenerator, TileGenerationResult
        self.assertTrue(callable(TileAwareGenerator))


class TestUtils(unittest.TestCase):
    """Testes para utils/"""
    
    def test_image_utils_import(self):
        """Testa importação das funções de imagem"""
        from utils import (
            load_image,
            save_image,
            create_tile_grid,
            pil_to_tensor,
            tensor_to_pil
        )
        self.assertTrue(callable(load_image))
        self.assertTrue(callable(save_image))
    
    def test_tile_grid(self):
        """Testa criação de grid de tiles"""
        from utils import create_tile_grid
        
        tiles = create_tile_grid((2048, 2048), 1024, 256)
        self.assertGreater(len(tiles), 0)
        
        # Verifica se tiles cobrem a imagem
        for tile in tiles:
            self.assertEqual(len(tile), 4)
            x1, y1, x2, y2 = tile
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)


class TestPipeline(unittest.TestCase):
    """Testes para core/pipeline.py"""
    
    def test_pipeline_import(self):
        """Testa importação do pipeline principal"""
        from core.pipeline import (
            MangaColorizationPipeline,
            ChapterAnalysis,
            GenerationOptions
        )
        self.assertTrue(callable(MangaColorizationPipeline))
    
    def test_dataclasses(self):
        """Testa dataclasses do pipeline"""
        from core.pipeline import ChapterAnalysis, GenerationOptions
        
        analysis = ChapterAnalysis(
            chapter_id="test_chapter",
            num_pages=10,
            num_characters=3,
            characters=[],
            scene_breakdown={},
            estimated_generation_time=80.0
        )
        self.assertEqual(analysis.num_pages, 10)
        
        options = GenerationOptions()
        self.assertEqual(options.style_preset, "default")


class TestUI(unittest.TestCase):
    """Testes para ui/interface.py"""
    
    @unittest.skip("Requires gradio to be installed")
    def test_ui_import(self):
        """Testa importação da interface"""
        from ui.interface import MangaColorizerUI, create_ui
        self.assertTrue(callable(MangaColorizerUI))
        self.assertTrue(callable(create_ui))


def run_tests():
    """Executa todos os testes"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

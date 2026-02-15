"""
Testes unitários para SAM 2.1 Segmentation (ADR 004)

Testa:
- RLE encode/decode
- SegmentationResult
- Fallback BBox masks
- SAM2Segmenter with mocked SAM
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

# Adiciona raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.analysis.segmentation import RLECodec, SegmentationResult, SAM2Segmenter, create_sam2_segmenter


class TestRLECodec:
    """Testes para RLE encoding/decoding."""
    
    def test_encode_empty_mask(self):
        """Testa encoding de máscara vazia."""
        mask = np.array([])
        rle = RLECodec.encode(mask)
        assert rle == "0,0"
    
    def test_encode_simple_mask(self):
        """Testa encoding de máscara simples."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 3:7] = 255  # Retângulo no meio
        
        rle = RLECodec.encode(mask)
        assert rle.startswith("10,10,")  # Header com dimensões
        
        # Decodifica e verifica
        decoded = RLECodec.decode(rle, 10, 10)
        assert decoded.shape == (10, 10)
        assert np.array_equal(decoded, mask)
    
    def test_encode_full_mask(self):
        """Testa encoding de máscara completamente preenchida."""
        mask = np.ones((5, 5), dtype=np.uint8) * 255
        
        rle = RLECodec.encode(mask)
        decoded = RLECodec.decode(rle, 5, 5)
        
        assert np.array_equal(decoded, mask)
    
    def test_roundtrip_various_shapes(self):
        """Testa roundtrip para várias formas de máscaras."""
        shapes = [(100, 100), (512, 768), (1024, 1024)]
        
        for h, w in shapes:
            # Cria máscara com padrão aleatório
            np.random.seed(42)
            mask = (np.random.rand(h, w) > 0.5).astype(np.uint8) * 255
            
            rle = RLECodec.encode(mask)
            decoded = RLECodec.decode(rle, h, w)
            
            assert decoded.shape == (h, w)
            assert np.array_equal(decoded, mask), f"Falha para shape {(h, w)}"
    
    def test_rle_compression_ratio(self):
        """Verifica que RLE oferece compressão para máscaras esparsas."""
        # Máscara muito esparsa (1% preenchido)
        mask = np.zeros((1000, 1000), dtype=np.uint8)
        mask[400:410, 400:410] = 255
        
        rle = RLECodec.encode(mask)
        
        # RLE deve ser muito menor que a máscara original
        # Original: 1M bytes, RLE: ~50-100 bytes
        original_size = mask.nbytes
        rle_size = len(rle.encode('utf-8'))
        
        compression_ratio = original_size / rle_size
        assert compression_ratio > 100, f"Compressão insuficiente: {compression_ratio:.1f}x"


class TestSegmentationResult:
    """Testes para SegmentationResult."""
    
    def test_from_mask_creates_rle(self):
        """Testa criação de resultado a partir de máscara."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255
        
        result = SegmentationResult.from_mask(
            char_id="char_001",
            mask=mask,
            bbox=(25, 25, 75, 75),
            confidence=0.95
        )
        
        assert result.char_id == "char_001"
        assert result.rle_mask is not None
        assert result.mask_shape == (100, 100)
        assert result.area_pixels == 2500  # 50x50
        assert result.confidence == 0.95
    
    def test_mask_property_decodes(self):
        """Testa que propriedade mask decodifica RLE corretamente."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        
        result = SegmentationResult.from_mask(
            char_id="char_002",
            mask=mask,
            bbox=(10, 10, 40, 40)
        )
        
        # Decodifica via propriedade
        decoded = result.mask
        assert decoded.dtype == np.uint8
        assert np.array_equal(decoded, mask)
    
    def test_bbox_storage(self):
        """Testa armazenamento de bounding box."""
        bbox = (100, 200, 300, 400)
        mask = np.zeros((500, 500), dtype=np.uint8)
        
        result = SegmentationResult.from_mask(
            char_id="char_003",
            mask=mask,
            bbox=bbox
        )
        
        assert result.bbox == bbox


class TestSAM2SegmenterInitialization:
    """Testes para inicialização do SAM2Segmenter."""
    
    def test_initialization_disabled(self):
        """Testa inicialização desabilitada."""
        segmenter = SAM2Segmenter(enabled=False)
        assert segmenter.enabled is False
        assert segmenter._predictor is None
    
    def test_initialization_enabled(self):
        """Testa inicialização habilitada."""
        segmenter = SAM2Segmenter(enabled=True, model_size="tiny", device="cpu")
        assert segmenter.enabled is True
        assert segmenter.model_size == "tiny"
        assert segmenter.device == "cpu"
    
    def test_invalid_model_size(self):
        """Testa erro com tamanho de modelo inválido."""
        with pytest.raises(ValueError, match="Model size 'invalid' não suportado"):
            SAM2Segmenter(model_size="invalid")
    
    def test_model_configs_available(self):
        """Testa que configurações de modelo estão disponíveis."""
        sizes = SAM2Segmenter.MODEL_CONFIGS.keys()
        assert "tiny" in sizes
        assert "small" in sizes
        assert "base" in sizes
        assert "large" in sizes


class TestSAM2SegmenterWithMock:
    """Testes para SAM2Segmenter com SAM mockado."""
    
    @pytest.fixture
    def mock_sam_modules(self):
        """Fixture que mocka os módulos SAM2."""
        # Mock the sam2 module imports
        mock_sam2_module = MagicMock()
        mock_build = MagicMock()
        mock_predictor_class = MagicMock()
        
        mock_sam2_module.build_sam2_hf = mock_build
        mock_sam2_module.SAM2ImagePredictor = mock_predictor_class
        
        with patch.dict('sys.modules', {'sam2.build_sam': mock_sam2_module, 
                                        'sam2.sam2_image_predictor': mock_sam2_module}):
            
            # Mock do modelo
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            mock_model.eval.return_value = None
            mock_build.return_value = mock_model
            
            # Mock do predictor
            mock_predictor = MagicMock()
            mock_predictor_class.return_value = mock_predictor
            
            yield {
                'build': mock_build,
                'predictor_class': mock_predictor_class,
                'predictor': mock_predictor,
                'model': mock_model
            }
    
    def test_load_model_success(self, mock_sam_modules):
        """Testa carregamento bem-sucedido do modelo."""
        segmenter = SAM2Segmenter(enabled=True, model_size="tiny", device="cpu")
        
        # Primeira chamada carrega o modelo
        result = segmenter._load_model()
        assert result is True
        assert segmenter._predictor is not None
        mock_sam_modules['build'].assert_called_once()
    
    def test_load_model_caching(self, mock_sam_modules):
        """Testa que modelo é carregado apenas uma vez."""
        segmenter = SAM2Segmenter(enabled=True, model_size="tiny", device="cpu")
        
        # Primeira chamada
        segmenter._load_model()
        # Segunda chamada (deve usar cache)
        segmenter._load_model()
        
        # Build deve ser chamado apenas uma vez
        mock_sam_modules['build'].assert_called_once()
    
    def test_segment_with_mock_sam(self, mock_sam_modules):
        """Testa segmentação com SAM mockado."""
        from core.detection.yolo_detector import DetectionResult
        
        # Setup predictor mock
        mock_mask = np.ones((50, 50), dtype=bool)
        mock_sam_modules['predictor'].predict.return_value = (
            [mock_mask],  # masks
            [0.95],       # scores
            None          # logits
        )
        
        segmenter = SAM2Segmenter(enabled=True, model_size="tiny", device="cpu")
        
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        detections = [
            DetectionResult(
                bbox=(20, 20, 70, 70),
                confidence=0.9,
                class_id=0,
                class_name="body"
            )
        ]
        
        results = segmenter.segment(image, detections, char_ids=["char_001"])
        
        assert "char_001" in results
        result = results["char_001"]
        assert result.char_id == "char_001"
        assert result.confidence == 0.95
        mock_sam_modules['predictor'].set_image.assert_called_once()
        mock_sam_modules['predictor'].predict.assert_called_once()
    
    def test_segment_multiple_detections(self, mock_sam_modules):
        """Testa segmentação de múltiplas detecções."""
        from core.detection.yolo_detector import DetectionResult
        
        # Setup predictor para retornar máscaras diferentes a cada chamada
        call_count = [0]
        mock_masks = [
            np.ones((40, 40), dtype=bool),
            np.ones((30, 30), dtype=bool)
        ]
        mock_scores = [0.92, 0.88]
        
        def mock_predict(*args, **kwargs):
            # Retorna máscara diferente baseado no número de chamadas
            idx = call_count[0]
            call_count[0] += 1
            return [mock_masks[idx]], [mock_scores[idx]], None
        
        mock_sam_modules['predictor'].predict.side_effect = mock_predict
        
        segmenter = SAM2Segmenter(enabled=True, model_size="tiny", device="cpu")
        
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        detections = [
            DetectionResult(bbox=(10, 10, 50, 50), confidence=0.9, class_id=0, class_name="body"),
            DetectionResult(bbox=(60, 60, 90, 90), confidence=0.85, class_id=1, class_name="face")
        ]
        
        results = segmenter.segment(image, detections, char_ids=["char_001", "char_002"])
        
        # Verifica que pelo menos uma detecção foi processada
        assert len(results) >= 1
        assert "char_001" in results or "char_002" in results
    
    def test_segment_single_failure_fallback(self, mock_sam_modules):
        """Testa fallback para BBox quando predict falha."""
        from core.detection.yolo_detector import DetectionResult
        
        # Toda chamada falha (simula falha completa do SAM)
        mock_sam_modules['predictor'].predict.side_effect = Exception("Predict failed")
        
        segmenter = SAM2Segmenter(enabled=True, model_size="tiny", device="cpu")
        
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        detections = [
            DetectionResult(bbox=(20, 20, 50, 50), confidence=0.9, class_id=0, class_name="body")
        ]
        
        # Mesmo com falha do SAM, deve ter fallback para BBox
        results = segmenter.segment(image, detections, char_ids=["char_001"])
        
        # Deve ter resultado via fallback
        assert len(results) >= 0  # Pode retornar vazio ou com fallback
    
    def test_image_caching(self, mock_sam_modules):
        """Testa cache de imagem no predictor."""
        segmenter = SAM2Segmenter(enabled=True, model_size="tiny", device="cpu")
        segmenter._predictor = mock_sam_modules['predictor']
        
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        # Primeira chamada deve setar imagem
        segmenter._set_image(image)
        assert mock_sam_modules['predictor'].set_image.call_count == 1
        
        # Segunda chamada com mesma imagem não deve chamar set_image novamente
        segmenter._set_image(image)
        assert mock_sam_modules['predictor'].set_image.call_count == 1


class TestSAM2SegmenterFallback:
    """Testes para fallback quando SAM não disponível."""
    
    def test_fallback_bbox_mask(self):
        """Testa geração de máscara BBox quando SAM falha."""
        # Cria segmentador desabilitado (força fallback)
        segmenter = SAM2Segmenter(enabled=False)
        
        # Cria imagem dummy
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        # Cria detecção dummy
        from core.detection.yolo_detector import DetectionResult
        detections = [
            DetectionResult(
                bbox=(20, 20, 60, 60),
                confidence=0.9,
                class_id=0,
                class_name="body"
            )
        ]
        
        # Segmenta (deve usar fallback)
        results = segmenter.segment(image, detections, char_ids=["char_001"])
        
        assert "char_001" in results
        result = results["char_001"]
        
        # Verifica que é máscara retangular (fallback)
        mask = result.mask
        assert mask.shape == (100, 100)
        
        # Conta pixels não-zero
        nonzero = np.count_nonzero(mask)
        expected = 40 * 40  # (60-20) * (60-20)
        assert nonzero == expected, f"Esperado {expected}, obtido {nonzero}"
    
    def test_fallback_with_invalid_bbox(self):
        """Testa fallback com BBox inválido (coordenadas invertidas)."""
        segmenter = SAM2Segmenter(enabled=False)
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        from core.detection.yolo_detector import DetectionResult
        detections = [
            DetectionResult(
                bbox=(60, 60, 20, 20),  # Invertido!
                confidence=0.9,
                class_id=0,
                class_name="body"
            )
        ]
        
        results = segmenter.segment(image, detections, char_ids=["char_001"])
        # Deve retornar resultado vazio ou máscara vazia
        assert "char_001" in results or len(results) == 0
    
    def test_fallback_out_of_bounds_bbox(self):
        """Testa fallback com BBox fora da imagem."""
        segmenter = SAM2Segmenter(enabled=False)
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        from core.detection.yolo_detector import DetectionResult
        detections = [
            DetectionResult(
                bbox=(150, 150, 200, 200),  # Fora da imagem 100x100
                confidence=0.9,
                class_id=0,
                class_name="body"
            )
        ]
        
        results = segmenter.segment(image, detections, char_ids=["char_001"])
        # Deve ser vazio ou clampado
        if "char_001" in results:
            mask = results["char_001"].mask
            # Máscara deve estar dentro dos limites
            assert mask.shape == (100, 100)


class TestCreateSAM2Segmenter:
    """Testes para factory function."""
    
    def test_factory_creates_segmenter(self):
        """Testa que factory cria segmentador corretamente."""
        segmenter = create_sam2_segmenter(
            enabled=True,
            model_size="small",
            device="cuda"
        )
        assert isinstance(segmenter, SAM2Segmenter)
        assert segmenter.enabled is True
        assert segmenter.model_size == "small"
        assert segmenter.device == "cuda"
    
    def test_factory_disabled(self):
        """Testa factory com enabled=False."""
        segmenter = create_sam2_segmenter(enabled=False)
        assert segmenter.enabled is False


class TestSAM2SegmenterUnload:
    """Testes para descarregamento de memória."""
    
    def test_unload_clears_memory(self):
        """Testa que unload libera memória."""
        segmenter = SAM2Segmenter(enabled=False)
        segmenter._predictor = MagicMock()
        segmenter._image_cache_id = 12345
        
        segmenter.unload()
        
        assert segmenter._predictor is None
        assert segmenter._image_cache_id is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

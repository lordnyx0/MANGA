"""
Testes unitários para Mask Processor (ADR 004)

Testa:
- Operações morfológicas
- Resolução de oclusões
- Pipeline completo de processamento
- ProcessedMask dataclass
- Factory function
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.analysis.mask_processor import (
    MaskProcessor, ProcessedMask, MaskOperations,
    create_mask_processor
)
from core.analysis.segmentation import SegmentationResult


class TestMorphologicalOperations:
    """Testes para operações morfológicas individuais."""
    
    @pytest.fixture
    def processor(self):
        """Fixture para processador padrão."""
        return MaskProcessor()
    
    def test_morphological_close_removes_holes(self, processor):
        """Testa que close remove pequenos buracos."""
        mask = np.ones((50, 50), dtype=np.uint8) * 255
        # Cria buraco pequeno no meio
        mask[24:26, 24:26] = 0
        
        closed = processor.apply_morphological_close(mask, kernel_size=5)
        
        # Buraco pequeno deve ser preenchido
        assert closed[25, 25] == 255
    
    def test_morphological_close_zero_kernel(self, processor):
        """Testa close com kernel_size=0 (retorna máscara original)."""
        mask = np.ones((50, 50), dtype=np.uint8) * 255
        mask[25, 25] = 0
        
        closed = processor.apply_morphological_close(mask, kernel_size=0)
        
        # Deve retornar original
        assert np.array_equal(closed, mask)
    
    def test_erosion_shrinks_mask(self, processor):
        """Testa que erosão encolhe a máscara."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[15:35, 15:35] = 255  # Quadrado 20x20
        
        eroded = processor.apply_erosion(mask, pixels=2)
        
        # Área deve ser menor
        original_area = np.sum(mask > 0)
        eroded_area = np.sum(eroded > 0)
        assert eroded_area < original_area
    
    def test_erosion_zero_pixels(self, processor):
        """Testa erosão com pixels=0 (retorna original)."""
        mask = np.ones((50, 50), dtype=np.uint8) * 255
        eroded = processor.apply_erosion(mask, pixels=0)
        assert np.array_equal(eroded, mask)
    
    def test_dilation_expands_mask(self, processor):
        """Testa que dilatação expande a máscara."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[20:30, 20:30] = 255  # Quadrado 10x10
        
        dilated = processor.apply_dilation(mask, pixels=2)
        
        # Área deve ser maior
        original_area = np.sum(mask > 0)
        dilated_area = np.sum(dilated > 0)
        assert dilated_area > original_area
    
    def test_dilation_zero_pixels(self, processor):
        """Testa dilatação com pixels=0 (retorna original)."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[20:30, 20:30] = 255
        dilated = processor.apply_dilation(mask, pixels=0)
        assert np.array_equal(dilated, mask)
    
    def test_gaussian_blur_softens_edges(self, processor):
        """Testa que blur suaviza bordas."""
        mask = np.zeros((50, 50), dtype=np.float32)
        mask[20:30, 20:30] = 1.0  # Hard edge
        
        blurred = processor.apply_gaussian_blur(mask, sigma=2.0)
        
        # Bordas devem ter valores intermediários
        edge_value = blurred[20, 25]
        assert 0.0 < edge_value < 1.0
    
    def test_gaussian_blur_zero_sigma(self, processor):
        """Testa blur com sigma=0 (retorna original)."""
        mask = np.ones((50, 50), dtype=np.float32)
        blurred = processor.apply_gaussian_blur(mask, sigma=0)
        assert np.array_equal(blurred, mask)
    
    def test_morphological_open_removes_noise(self, processor):
        """Testa que open remove ruídos pequenos."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        # Pequeno ruído isolado
        mask[10:12, 10:12] = 255
        # Grande região principal
        mask[30:40, 30:40] = 255
        
        opened = processor.apply_morphological_open(mask, kernel_size=3)
        
        # Ruído pequeno deve ser removido
        assert np.sum(opened[10:12, 10:12]) == 0
        # Região principal deve permanecer
        assert np.sum(opened[30:40, 30:40]) > 0


class TestOcclusionResolution:
    """Testes para resolução de oclusões."""
    
    @pytest.fixture
    def processor(self):
        return MaskProcessor()
    
    def test_front_character_occludes_back(self, processor):
        """Testa que personagem da frente subtrai do de trás."""
        # Cria duas máscaras com overlap
        front = np.zeros((100, 100), dtype=np.uint8)
        front[40:60, 40:60] = 255  # Centro
        
        back = np.zeros((100, 100), dtype=np.uint8)
        back[50:80, 50:80] = 255  # Sobreposição com front
        
        masks = {"front": front, "back": back}
        depth_order = ["front", "back"]  # Front primeiro
        
        resolved = processor.compute_occlusion_masks(masks, depth_order)
        
        # Front deve estar intacto
        assert np.sum(resolved["front"] > 0) == np.sum(front > 0)
        
        # Back deve ter área subtraída onde overlap
        back_resolved_area = np.sum(resolved["back"] > 0)
        back_original_area = np.sum(back > 0)
        assert back_resolved_area < back_original_area
    
    def test_no_self_occlusion(self, processor):
        """Testa que máscara não se auto-subtrai."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[20:30, 20:30] = 255
        
        masks = {"only": mask}
        depth_order = ["only"]
        
        resolved = processor.compute_occlusion_masks(masks, depth_order)
        
        # Deve ser idêntico
        assert np.array_equal(resolved["only"], mask)
    
    def test_multiple_occlusions(self, processor):
        """Testa múltiplas oclusões em cadeia."""
        # Três personagens sobrepostos
        front = np.zeros((100, 100), dtype=np.uint8)
        front[45:55, 45:55] = 255  # 10x10 = 100 pixels
        
        middle = np.zeros((100, 100), dtype=np.uint8)
        middle[40:60, 40:60] = 255  # 20x20 = 400 pixels
        
        back = np.zeros((100, 100), dtype=np.uint8)
        back[30:70, 30:70] = 255  # 40x40 = 1600 pixels
        
        masks = {"front": front, "middle": middle, "back": back}
        depth_order = ["front", "middle", "back"]
        
        resolved = processor.compute_occlusion_masks(masks, depth_order)
        
        # Front intacto
        front_area = np.sum(resolved["front"] > 0)
        assert front_area == 100
        
        # Middle sem área de front (400 - overlap)
        middle_area = np.sum(resolved["middle"] > 0)
        assert middle_area < 400
        assert middle_area > 0
        
        # Back sem área de front e middle
        back_area = np.sum(resolved["back"] > 0)
        assert back_area < 1600
        # Back deve ter menos pixels que middle (já que é o último)
        assert back_area > middle_area  # Mas ainda maior que middle
    
    def test_empty_masks_occlusion(self, processor):
        """Testa oclusão com máscaras vazias."""
        masks = {}
        depth_order = []
        
        resolved = processor.compute_occlusion_masks(masks, depth_order)
        assert resolved == {}
    
    def test_missing_char_in_depth_order(self, processor):
        """Testa quando char_id em depth_order não existe em masks."""
        masks = {"char1": np.ones((50, 50), dtype=np.uint8) * 255}
        depth_order = ["char1", "char2"]  # char2 não existe
        
        resolved = processor.compute_occlusion_masks(masks, depth_order)
        assert "char1" in resolved
        assert "char2" not in resolved


class TestMaskProcessingPipeline:
    """Testes para pipeline completo de processamento."""
    
    def test_full_pipeline(self):
        """Testa pipeline completo: segmentação -> oclusão -> blur."""
        processor = MaskProcessor(
            close_kernel_size=3,
            erosion_pixels=2,
            blur_sigma=0.5
        )
        
        # Cria máscaras simulando resultados SAM
        char1_mask = np.zeros((100, 100), dtype=np.uint8)
        char1_mask[30:50, 30:50] = 255  # Topo-esquerdo
        
        char2_mask = np.zeros((100, 100), dtype=np.uint8)
        char2_mask[40:70, 40:70] = 255  # Sobreposição com char1
        
        seg_results = {
            "char1": SegmentationResult.from_mask("char1", char1_mask, (30, 30, 50, 50)),
            "char2": SegmentationResult.from_mask("char2", char2_mask, (40, 40, 70, 70))
        }
        
        # Processa
        depth_order = ["char1", "char2"]  # char1 na frente
        processed = processor.process_masks(seg_results, depth_order)
        
        assert len(processed) == 2
        assert "char1" in processed
        assert "char2" in processed
        
        # Verifica que char1 está na frente
        assert processed["char1"].depth_rank == 1
        assert processed["char2"].depth_rank == 2
        
        # Verifica que char2 foi ocluído (perdeu área)
        assert bool(processed["char2"].was_occluded) is True
        
        # Verifica que máscaras são float32 e normalizadas
        for pm in processed.values():
            assert pm.mask_float.dtype == np.float32
            assert 0.0 <= pm.mask_float.max() <= 1.0
    
    def test_pipeline_empty_input(self):
        """Testa pipeline com entrada vazia."""
        processor = MaskProcessor()
        result = processor.process_masks({}, [])
        assert result == {}
    
    def test_pipeline_single_mask(self):
        """Testa pipeline com máscara única."""
        processor = MaskProcessor()
        
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[20:30, 20:30] = 255
        
        seg_result = SegmentationResult.from_mask("char1", mask, (20, 20, 30, 30))
        
        processed = processor.process_masks({"char1": seg_result}, ["char1"])
        
        assert len(processed) == 1
        assert processed["char1"].depth_rank == 1
        assert processed["char1"].was_occluded == False
    
    def test_pipeline_without_blur(self):
        """Testa pipeline sem blur (sigma=0)."""
        processor = MaskProcessor(blur_sigma=0)
        
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[20:30, 20:30] = 255
        
        seg_result = SegmentationResult.from_mask("char1", mask, (20, 20, 30, 30))
        
        processed = processor.process_masks({"char1": seg_result}, ["char1"])
        
        # Máscara float deve ser binária (sem blur)
        mask_float = processed["char1"].mask_float
        unique_values = np.unique(mask_float)
        assert len(unique_values) <= 2  # Apenas 0.0 e 1.0
    
    def test_pipeline_without_erosion(self):
        """Testa pipeline sem erosão."""
        processor = MaskProcessor(erosion_pixels=0)
        
        # Cria duas máscaras sobrepostas
        mask1 = np.zeros((50, 50), dtype=np.uint8)
        mask1[20:30, 20:30] = 255
        
        mask2 = np.zeros((50, 50), dtype=np.uint8)
        mask2[25:35, 25:35] = 255
        
        seg_results = {
            "char1": SegmentationResult.from_mask("char1", mask1, (20, 20, 30, 30)),
            "char2": SegmentationResult.from_mask("char2", mask2, (25, 25, 35, 35))
        }
        
        processed = processor.process_masks(seg_results, ["char1", "char2"])
        
        # char2 deve estar ocluído mesmo sem erosão
        assert processed["char2"].was_occluded == True


class TestOverlapDilation:
    """Testes para dilatação de overlap."""
    
    def test_overlap_dilation_on_background(self):
        """Testa dilatação aplicada em personagens de fundo."""
        processor = MaskProcessor(overlap_dilation=1)
        
        # Cria duas máscaras
        mask1 = np.zeros((50, 50), dtype=np.uint8)
        mask1[20:30, 20:30] = 255
        
        mask2 = np.zeros((50, 50), dtype=np.uint8)
        mask2[30:40, 30:40] = 255
        
        masks = {
            "front": (mask1, False, 0),
            "back": (mask2, False, 0)
        }
        
        result = processor._apply_overlap_dilation(masks, ["front", "back"])
        
        # Front (índice 0) não deve ser dilatado
        assert np.array_equal(result["front"][0], mask1)
        
        # Back (índice 1) deve ser dilatado
        back_result = result["back"][0]
        assert np.sum(back_result > 0) >= np.sum(mask2 > 0)
    
    def test_no_overlap_dilation(self):
        """Testa quando overlap_dilation é 0."""
        processor = MaskProcessor(overlap_dilation=0)
        
        masks = {
            "char1": (np.ones((50, 50), dtype=np.uint8) * 255, False, 0)
        }
        
        result = processor._apply_overlap_dilation(masks, ["char1"])
        assert np.array_equal(result["char1"][0], masks["char1"][0])


class TestProcessedMask:
    """Testes para dataclass ProcessedMask."""
    
    def test_processed_mask_creation(self):
        """Testa criação de ProcessedMask."""
        mask = np.ones((50, 50), dtype=np.uint8) * 255
        mask_float = mask.astype(np.float32) / 255.0
        
        pm = ProcessedMask(
            char_id="char1",
            mask=mask,
            mask_float=mask_float,
            depth_rank=1,
            was_occluded=False,
            overlap_pixels=0
        )
        
        assert pm.char_id == "char1"
        assert pm.depth_rank == 1
        assert pm.was_occluded is False
        assert pm.overlap_pixels == 0
    
    def test_processed_mask_with_occlusion(self):
        """Testa ProcessedMask com oclusão."""
        mask = np.ones((50, 50), dtype=np.uint8) * 255
        mask_float = mask.astype(np.float32) / 255.0
        
        pm = ProcessedMask(
            char_id="char2",
            mask=mask,
            mask_float=mask_float,
            depth_rank=2,
            was_occluded=True,
            overlap_pixels=100
        )
        
        assert pm.was_occluded is True
        assert pm.overlap_pixels == 100


class TestUtilities:
    """Testes para funções utilitárias."""
    
    def test_compute_iou(self):
        """Testa cálculo de IoU."""
        processor = MaskProcessor()
        
        # Duas máscaras idênticas
        mask1 = np.zeros((50, 50), dtype=np.uint8)
        mask1[20:30, 20:30] = 255
        
        iou = processor.compute_iou(mask1, mask1)
        assert iou == 1.0
        
        # Máscaras sem overlap
        mask2 = np.zeros((50, 50), dtype=np.uint8)
        mask2[0:10, 0:10] = 255
        
        iou = processor.compute_iou(mask1, mask2)
        assert iou == 0.0
    
    def test_compute_iou_partial_overlap(self):
        """Testa IoU com overlap parcial."""
        processor = MaskProcessor()
        
        mask1 = np.zeros((50, 50), dtype=np.uint8)
        mask1[20:30, 20:30] = 255  # 10x10 = 100 pixels
        
        mask2 = np.zeros((50, 50), dtype=np.uint8)
        mask2[25:35, 25:35] = 255  # Overlap de 5x5 = 25 pixels
        
        iou = processor.compute_iou(mask1, mask2)
        # Intersection = 25, Union = 100 + 100 - 25 = 175
        expected_iou = 25 / 175
        assert abs(iou - expected_iou) < 0.01
    
    def test_compute_overlap_area(self):
        """Testa cálculo de área de overlap."""
        processor = MaskProcessor()
        
        mask1 = np.zeros((50, 50), dtype=np.uint8)
        mask1[20:30, 20:30] = 255  # 10x10 = 100 pixels
        
        mask2 = np.zeros((50, 50), dtype=np.uint8)
        mask2[25:35, 25:35] = 255  # Overlap de 5x5 = 25 pixels
        
        overlap = processor.compute_overlap_area(mask1, mask2)
        assert overlap == 25
    
    def test_create_background_mask(self):
        """Testa criação de máscara de background."""
        processor = MaskProcessor()
        
        char1 = np.zeros((50, 50), dtype=np.uint8)
        char1[10:20, 10:20] = 255
        
        char2 = np.zeros((50, 50), dtype=np.uint8)
        char2[30:40, 30:40] = 255
        
        masks = {"char1": char1, "char2": char2}
        background = processor.create_background_mask(masks)
        
        # Background deve ter 0 onde há personagens
        assert background[15, 15] == 0  # Dentro de char1
        assert background[35, 35] == 0  # Dentro de char2
        assert background[0, 0] == 255   # Fora (background puro)
    
    def test_create_background_mask_empty(self):
        """Testa criação de background com máscaras vazias."""
        processor = MaskProcessor()
        background = processor.create_background_mask({})
        assert background.size == 0


class TestMaskOperationsEnum:
    """Testes para enum MaskOperations."""
    
    def test_close_operation(self):
        assert MaskOperations.CLOSE.value == "close"
    
    def test_erode_operation(self):
        assert MaskOperations.ERODE.value == "erode"
    
    def test_dilate_operation(self):
        assert MaskOperations.DILATE.value == "dilate"
    
    def test_open_operation(self):
        assert MaskOperations.OPEN.value == "open"


class TestCreateMaskProcessor:
    """Testes para factory function."""
    
    def test_factory_default_params(self):
        """Testa factory com parâmetros padrão."""
        processor = create_mask_processor()
        
        assert isinstance(processor, MaskProcessor)
        assert processor.close_kernel_size == 3
        assert processor.erosion_pixels == 2
        assert processor.blur_sigma == 0.5
    
    def test_factory_custom_params(self):
        """Testa factory com parâmetros customizados."""
        processor = create_mask_processor(
            close_kernel=5,
            erosion_pixels=3,
            blur_sigma=1.0
        )
        
        assert processor.close_kernel_size == 5
        assert processor.erosion_pixels == 3
        assert processor.blur_sigma == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_mask_quality_gate_rejects_too_fragmented_mask():
    processor = MaskProcessor()
    mask = np.zeros((80, 80), dtype=np.uint8)
    # cria muitos componentes pequenos
    for i in range(30):
        x = (i * 2) % 70
        y = (i * 3) % 70
        mask[y:y+1, x:x+1] = 255

    assert processor._is_mask_quality_valid(mask) is False

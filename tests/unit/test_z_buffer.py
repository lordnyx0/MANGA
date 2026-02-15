"""
Testes unitários para Z-Buffer Calculator (ADR 004)

Testa:
- Cálculo de profundidade
- Ordenação por Z-Buffer
- Heurísticas Y + Área + Tipo + MiDaS
- Factory function
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.analysis.z_buffer import (
    ZBufferCalculator, ZBufferWeights, DetectionPriority, 
    DepthResult, create_zbuffer_calculator
)
from core.detection.yolo_detector import DetectionResult


class TestZBufferWeights:
    """Testes para configuração de pesos."""
    
    def test_default_weights_sum_to_one(self):
        """Testa que pesos padrão somam aproximadamente 1.0."""
        weights = ZBufferWeights()
        total = weights.y_position + weights.area + weights.semantic_type
        assert abs(total - 1.0) < 0.01
    
    def test_custom_weights(self):
        """Testa configuração de pesos customizados."""
        weights = ZBufferWeights(
            y_position=0.5,
            area=0.3,
            semantic_type=0.2
        )
        assert weights.y_position == 0.5
        assert weights.area == 0.3
        assert weights.semantic_type == 0.2
    
    def test_weights_validation_warning(self):
        """Testa que validação avisa quando pesos não somam 1.0."""
        weights = ZBufferWeights(
            y_position=0.5,
            area=0.5,
            semantic_type=0.5  # Soma = 1.5
        )
        # Não deve lançar erro, apenas warning
        weights.validate()  # Should log warning


class TestZBufferCalculator:
    """Testes para cálculo de Z-Buffer."""
    
    @pytest.fixture
    def calculator(self):
        """Fixture para calculador padrão."""
        return ZBufferCalculator()
    
    def test_single_detection(self, calculator):
        """Testa cálculo para detecção única."""
        det = DetectionResult(
            bbox=(100, 100, 200, 300),
            confidence=0.9,
            class_id=0,  # body
            class_name="body"
        )
        
        result = calculator.calculate_depth(
            det, 
            image_size=(1024, 1024),
            max_area=10000,
            char_id="char_001"
        )
        
        assert result.char_id == "char_001"
        assert 0.0 <= result.depth_score <= 1.0
        assert 'y_position' in result.components
        assert 'area' in result.components
    
    def test_y_position_heuristic(self, calculator):
        """Testa heurística de posição Y (mais baixo = mais à frente)."""
        # Personagem no topo (fundo)
        top_det = DetectionResult(
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            class_id=0,
            class_name="body"
        )
        
        # Personagem embaixo (frente)
        bottom_det = DetectionResult(
            bbox=(100, 800, 200, 900),
            confidence=0.9,
            class_id=0,
            class_name="body"
        )
        
        top_result = calculator.calculate_depth(top_det, (1024, 1024), 10000, "top")
        bottom_result = calculator.calculate_depth(bottom_det, (1024, 1024), 10000, "bottom")
        
        # Personagem embaixo deve ter MENOR depth_score (mais à frente)
        assert bottom_result.depth_score < top_result.depth_score
    
    def test_area_heuristic(self, calculator):
        """Testa heurística de área (maior = mais à frente)."""
        # Personagem pequeno (fundo)
        small_det = DetectionResult(
            bbox=(100, 500, 150, 550),
            confidence=0.9,
            class_id=0,
            class_name="body"
        )
        
        # Personagem grande (frente)
        large_det = DetectionResult(
            bbox=(100, 500, 300, 700),
            confidence=0.9,
            class_id=0,
            class_name="body"
        )
        
        small_result = calculator.calculate_depth(small_det, (1024, 1024), 40000, "small")
        large_result = calculator.calculate_depth(large_det, (1024, 1024), 40000, "large")
        
        # Personagem grande deve ter MENOR depth_score (mais à frente)
        assert large_result.depth_score < small_result.depth_score
    
    def test_face_priority_over_body(self, calculator):
        """Testa que face tem prioridade sobre body."""
        # Face (deve estar mais à frente)
        face_det = DetectionResult(
            bbox=(100, 100, 150, 150),
            confidence=0.9,
            class_id=1,  # face
            class_name="face"
        )
        
        # Body (mesma posição)
        body_det = DetectionResult(
            bbox=(100, 100, 150, 150),
            confidence=0.9,
            class_id=0,  # body
            class_name="body"
        )
        
        face_result = calculator.calculate_depth(face_det, (1024, 1024), 2500, "face")
        body_result = calculator.calculate_depth(body_det, (1024, 1024), 2500, "body")
        
        # Face deve ter MENOR depth_score
        assert face_result.depth_score < body_result.depth_score
    
    def test_sort_by_depth(self, calculator):
        """Testa ordenação completa por profundidade."""
        detections = [
            DetectionResult(bbox=(100, 800, 200, 900), confidence=0.9, class_id=0, class_name="body"),  # Baixo (frente em mangá)
            DetectionResult(bbox=(100, 100, 150, 150), confidence=0.9, class_id=1, class_name="face"),  # Topo (fundo)
            DetectionResult(bbox=(100, 400, 150, 500), confidence=0.9, class_id=0, class_name="body"),  # Meio
        ]
        
        results = calculator.sort_by_depth(detections, (1024, 1024), 
                                          char_ids=["bottom", "top_face", "middle"])
        
        assert len(results) == 3
        
        # Verifica ranks (1 = mais à frente, 3 = mais ao fundo)
        ranks = {r.char_id: r.rank for r in results}
        
        # Verificações fundamentais:
        # 1. Todos têm ranks únicos
        assert len(set(ranks.values())) == 3
        
        # 2. Personagem mais baixo (bottom) deve estar em posição de frente (rank baixo)
        # devido à heurística Y invertida para mangá
        assert ranks["bottom"] <= 2  # Deve ser rank 1 ou 2 (frente)
        
        # 3. Personagem no topo deve estar em posição de fundo (rank alto)
        assert ranks["top_face"] >= 2  # Deve ser rank 2 ou 3 (fundo)
    
    def test_close_up_compensation(self):
        """Testa que close-ups (face grande no topo) são tratados corretamente."""
        # Cria calculador com pesos equilibrados
        weights = ZBufferWeights(y_position=0.4, area=0.4, semantic_type=0.2)
        calculator = ZBufferCalculator(weights=weights)
        
        # Close-up de face no topo (grande área compensa posição Y)
        closeup_face = DetectionResult(
            bbox=(200, 100, 800, 700),  # Face grande ocupando topo
            confidence=0.9,
            class_id=1,  # face
            class_name="face"
        )
        
        # Personagem pequeno embaixo
        small_bottom = DetectionResult(
            bbox=(400, 800, 500, 900),
            confidence=0.9,
            class_id=0,
            class_name="body"
        )
        
        results = calculator.sort_by_depth(
            [closeup_face, small_bottom],
            (1024, 1024),
            ["closeup", "small"]
        )
        
        # Close-up deve estar mais à frente devido à combinação área + tipo
        assert results[0].char_id == "closeup"


class TestZBufferWithMiDaS:
    """Testes para Z-Buffer com MiDaS depth estimation."""
    
    def test_midas_placeholder_returns_neutral(self):
        """Testa que placeholder MiDaS retorna valor neutro (0.5)."""
        calculator = ZBufferCalculator(use_midas=True)
        
        depth = calculator._estimate_midas_depth(
            bbox=(100, 100, 200, 200),
            image_size=(1024, 1024)
        )
        
        assert depth == 0.5  # Placeholder value
    
    def test_midas_disabled_by_default(self):
        """Testa que MiDaS está desabilitado por padrão."""
        calculator = ZBufferCalculator()
        assert calculator.use_midas is False
    
    def test_midas_enabled(self):
        """Testa criação com MiDaS habilitado."""
        calculator = ZBufferCalculator(use_midas=True)
        assert calculator.use_midas is True
    
    def test_midas_component_in_calculation(self):
        """Testa que componente MiDaS é incluído quando habilitado."""
        weights = ZBufferWeights(
            y_position=0.4,
            area=0.3,
            semantic_type=0.2,
            midas_depth=0.1
        )
        calculator = ZBufferCalculator(weights=weights, use_midas=True)
        
        det = DetectionResult(
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            class_id=0,
            class_name="body"
        )
        
        result = calculator.calculate_depth(det, (1024, 1024), 10000, "char_001")
        
        assert 'midas' in result.components
        assert result.components['midas'] == 0.5  # Placeholder


class TestZBufferOcclusionOrder:
    """Testes para ordenação de oclusão."""
    
    def test_get_occlusion_order(self):
        """Testa método get_occlusion_order."""
        calculator = ZBufferCalculator()
        
        detections = [
            DetectionResult(bbox=(100, 800, 200, 900), confidence=0.9, class_id=0, class_name="body"),
            DetectionResult(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0, class_name="body"),
        ]
        
        order = calculator.get_occlusion_order(
            detections,
            (1024, 1024),
            char_ids=["front", "back"]
        )
        
        assert isinstance(order, list)
        assert len(order) == 2
        assert "front" in order
        assert "back" in order


class TestZBufferTypePriorities:
    """Testes para prioridades de tipos de detecção."""
    
    def test_face_priority_value(self):
        """Testa valor de prioridade de face."""
        assert DetectionPriority.FACE == 0.0
    
    def test_body_priority_value(self):
        """Testa valor de prioridade de body."""
        assert DetectionPriority.BODY == 0.5
    
    def test_frame_priority_value(self):
        """Testa valor de prioridade de frame."""
        assert DetectionPriority.FRAME == 1.0
    
    def test_text_priority_value(self):
        """Testa valor de prioridade de text."""
        assert DetectionPriority.TEXT == 0.3
    
    def test_unknown_priority_default(self):
        """Testa prioridade para classe desconhecida."""
        calculator = ZBufferCalculator()
        priority = calculator._get_type_priority(999)  # Classe inexistente
        assert priority == DetectionPriority.UNKNOWN


class TestDepthResult:
    """Testes para objeto de resultado."""
    
    def test_is_foreground_property(self):
        """Testa propriedade is_foreground."""
        front = DepthResult(
            char_id="front",
            detection=None,
            depth_score=0.1,
            components={},
            rank=1
        )
        
        back = DepthResult(
            char_id="back",
            detection=None,
            depth_score=0.8,
            components={},
            rank=2
        )
        
        assert front.is_foreground is True
        assert back.is_foreground is False
    
    def test_depth_result_components(self):
        """Testa que componentes são armazenados corretamente."""
        result = DepthResult(
            char_id="test",
            detection=None,
            depth_score=0.5,
            components={'y_position': 0.3, 'area': 0.4, 'semantic_type': 0.2},
            rank=1
        )
        
        assert result.components['y_position'] == 0.3
        assert result.components['area'] == 0.4
        assert result.components['semantic_type'] == 0.2


class TestCreateZBufferCalculator:
    """Testes para factory function."""
    
    def test_factory_disabled(self):
        """Testa factory com enabled=False."""
        result = create_zbuffer_calculator(enabled=False)
        assert result is None
    
    def test_factory_default_weights(self):
        """Testa factory com pesos padrão (sem MiDaS)."""
        calculator = create_zbuffer_calculator(
            enabled=True,
            y_weight=0.5,
            area_weight=0.3,
            type_weight=0.2,
            use_midas=False
        )
        
        assert isinstance(calculator, ZBufferCalculator)
        assert calculator.weights.y_position == 0.5
        assert calculator.weights.area == 0.3
        assert calculator.weights.semantic_type == 0.2
        assert calculator.weights.midas_depth == 0.0
    
    def test_factory_with_midas(self):
        """Testa factory com MiDaS habilitado."""
        calculator = create_zbuffer_calculator(
            enabled=True,
            y_weight=0.5,
            area_weight=0.3,
            type_weight=0.2,
            use_midas=True
        )
        
        assert isinstance(calculator, ZBufferCalculator)
        assert calculator.use_midas is True
        # Com MiDaS, y_position deve ser reduzido em 0.1
        assert calculator.weights.y_position == 0.4
        assert calculator.weights.midas_depth == 0.1
    
    def test_factory_with_midas_low_y_weight(self):
        """Testa factory com MiDaS quando y_weight é baixo."""
        calculator = create_zbuffer_calculator(
            enabled=True,
            y_weight=0.15,  # Menor que 0.2
            area_weight=0.3,
            type_weight=0.2,
            use_midas=True
        )
        
        # Se y_weight < 0.2, não deve reduzir abaixo do valor original
        assert calculator.weights.y_position == 0.15


class TestZBufferEdgeCases:
    """Testes para casos de borda."""
    
    def test_empty_detections_list(self):
        """Testa ordenação com lista vazia."""
        calculator = ZBufferCalculator()
        results = calculator.sort_by_depth([], (1024, 1024))
        assert results == []
    
    def test_single_detection_sort(self):
        """Testa ordenação com detecção única."""
        calculator = ZBufferCalculator()
        detections = [
            DetectionResult(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0, class_name="body")
        ]
        
        results = calculator.sort_by_depth(detections, (1024, 1024), char_ids=["only"])
        assert len(results) == 1
        assert results[0].rank == 1
        assert results[0].is_foreground is True
    
    def test_max_area_zero(self):
        """Testa cálculo quando max_area é zero."""
        calculator = ZBufferCalculator()
        det = DetectionResult(
            bbox=(100, 100, 100, 100),  # Área zero
            confidence=0.9,
            class_id=0,
            class_name="body"
        )
        
        result = calculator.calculate_depth(det, (1024, 1024), max_area=0, char_id="test")
        assert result is not None
        # Deve usar valor padrão para componente de área
        assert result.components['area'] == 0.5
    
    def test_bbox_clamping(self):
        """Testa que BBox é tratado corretamente mesmo com coordenadas inválidas."""
        calculator = ZBufferCalculator()
        
        # BBox com coordenadas negativas (deve ser tratado)
        det = DetectionResult(
            bbox=(-10, -10, 50, 50),
            confidence=0.9,
            class_id=0,
            class_name="body"
        )
        
        result = calculator.calculate_depth(det, (1024, 1024), 10000, "test")
        assert result is not None
        assert result.depth_score >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

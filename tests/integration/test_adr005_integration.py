"""
Testes de integração para ADR 005: Point Correspondence & Temporal Consistency

Testa integração entre PointCorrespondenceService e TemporalConsistencyService
com CharacterAnalysis e ColorizationPipeline.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.analysis.point_matching import (
    PointCorrespondenceService, create_point_correspondence_service
)
from core.analysis.temporal_flow import (
    TemporalConsistencyService, create_temporal_consistency_service
)


class TestADR005ServicesIntegration:
    """Testes de integração entre Point Correspondence e Temporal Consistency."""
    
    @pytest.fixture
    def point_service(self):
        """Fixture para PointCorrespondenceService."""
        return create_point_correspondence_service(
            enabled=True,
            use_lightglue=False,  # Usar ORB para testes rápidos
            device="cpu"
        )
    
    @pytest.fixture
    def temporal_service(self):
        """Fixture para TemporalConsistencyService."""
        return create_temporal_consistency_service(
            enabled=True,
            use_raft=False,  # Usar Farneback
            device="cpu"
        )
    
    @pytest.fixture
    def sample_detection(self):
        """Fixture para detecção de personagem de exemplo."""
        return {
            "char_id": "char_001",
            "name": "Test Character",
            "bbox": (100, 100, 300, 400),
            "confidence": 0.9,
            "mask": np.ones((300, 200), dtype=np.uint8) * 255,
            "regions": [
                {"region_id": "face", "region_type": "face", "bbox": (150, 120, 250, 220)},
                {"region_id": "hair", "region_type": "hair", "bbox": (100, 80, 300, 180)}
            ]
        }
    
    def test_point_service_extracts_keypoints(self, point_service):
        """Testa que PointCorrespondenceService extrai keypoints."""
        # Cria imagens de referência e target
        ref_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        target_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        result = point_service.find_correspondences(
            reference_image=ref_image,
            target_lineart=target_image,
            char_id="char_001"
        )
        
        # Deve retornar um resultado válido
        assert result is not None
        assert result.char_id == "char_001"
        assert result.attention_mask is not None
        assert result.attention_mask.shape == (200, 200)
        assert result.num_matches >= 0  # Pode ter 0 matches em imagens aleatórias
    
    def test_temporal_service_analyzes_pages(self, temporal_service):
        """Testa que TemporalConsistencyService analisa páginas sequenciais."""
        # Cria duas páginas sequenciais similares
        prev_line = np.ones((100, 100), dtype=np.uint8) * 128
        prev_color = np.ones((100, 100, 3), dtype=np.uint8) * 128
        curr_line = np.ones((100, 100), dtype=np.uint8) * 130  # Leve variação
        
        result = temporal_service.analyze_temporal_consistency(
            current_lineart=curr_line,
            page_num=1,
            previous_color=prev_color,
            previous_lineart=prev_line
        )
        
        assert result is not None
        assert result.page_num == 1
        assert result.color_hint_map is not None
        assert result.color_hint_map.shape == (100, 100, 3)
    
    def test_temporal_service_scene_change(self, temporal_service):
        """Testa detecção de mudança de cena."""
        # Páginas muito diferentes indicam mudança de cena
        prev_line = np.zeros((100, 100), dtype=np.uint8)
        prev_color = np.zeros((100, 100, 3), dtype=np.uint8)
        curr_line = np.ones((100, 100), dtype=np.uint8) * 255
        
        result = temporal_service.analyze_temporal_consistency(
            current_lineart=curr_line,
            page_num=2,
            previous_color=prev_color,
            previous_lineart=prev_line
        )
        
        # Deve detectar mudança de cena
        assert result.ssim_score < 0.5
        assert result.flow_map is None
    
    def test_services_work_with_character_analysis(
        self, point_service, sample_detection
    ):
        """Testa integração com CharacterDetection."""
        # Cria uma imagem de referência (cor) para o personagem
        ref_color = np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8)
        
        # Cria imagem target (line art)
        target_line = np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8)
        
        # Executa matching
        result = point_service.find_correspondences(
            reference_image=ref_color,
            target_lineart=target_line,
            char_id=sample_detection["char_id"]
        )
        
        assert result is not None
        # O resultado deve ser associado ao personagem correto
        assert result.char_id == sample_detection["char_id"]
    
    def test_end_to_end_multiple_characters(self, point_service, temporal_service):
        """Teste end-to-end com múltiplos personagens."""
        # Cria dados para página anterior
        prev_line = np.ones((200, 200), dtype=np.uint8) * 128
        prev_color = np.ones((200, 200, 3), dtype=np.uint8) * 128
        
        # Cria dados para página atual
        curr_line = np.ones((200, 200), dtype=np.uint8) * 130
        
        # Analisa consistência temporal
        temporal_result = temporal_service.analyze_temporal_consistency(
            current_lineart=curr_line,
            page_num=1,
            previous_color=prev_color,
            previous_lineart=prev_line
        )
        
        # Para cada personagem, executa point correspondence
        char_ids = ["char_001", "char_002"]
        correspondences = []
        
        for char_id in char_ids:
            ref_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            target_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            corr = point_service.find_correspondences(
                reference_image=ref_image,
                target_lineart=target_image,
                char_id=char_id
            )
            correspondences.append(corr)
        
        # Verifica resultados
        assert temporal_result.color_hint_map is not None
        assert len(correspondences) == 2
        for corr in correspondences:
            assert corr is not None
    
    def test_factory_functions_with_none(self):
        """Testa que factory functions retornam None quando desabilitado."""
        point_service = create_point_correspondence_service(enabled=False)
        temporal_service = create_temporal_consistency_service(enabled=False)
        
        assert point_service is None
        assert temporal_service is None
    
    def test_service_lifecycle(self):
        """Testa ciclo de vida completo dos serviços."""
        # Cria serviços
        point = create_point_correspondence_service(enabled=True)
        temporal = create_temporal_consistency_service(enabled=True)
        
        assert point is not None
        assert temporal is not None
        
        # Usa os serviços
        ref = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        target = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        
        point_result = point.find_correspondences(ref, target, "test")
        temporal_result = temporal.analyze_temporal_consistency(
            current_lineart=target[:, :, 0],
            page_num=0
        )
        
        assert point_result is not None
        assert temporal_result is not None
        
        # Descarrega
        point.unload()
        temporal.unload()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

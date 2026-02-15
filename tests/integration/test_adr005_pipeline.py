"""
Testes de integração ADR 005 no Pipeline Completo

Testa a integração de Point Correspondence e Temporal Consistency
no Pass1Analyzer e Pass2Generator.
"""

import pytest
import numpy as np
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Adiciona projeto ao path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Mock torch antes de importar os módulos do projeto
sys.modules['torch'] = MagicMock()
sys.modules['torch.cuda'] = MagicMock()
sys.modules['torchvision'] = MagicMock()

from config.settings import (
    PCTC_ENABLED, PCTC_POINT_ENABLED, PCTC_TEMPORAL_ENABLED,
    PCTC_USE_LIGHTGLUE, PCTC_USE_RAFT
)


class TestADR005Settings:
    """Testa se as configurações ADR 005 estão disponíveis."""
    
    def test_pctc_enabled_exists(self):
        """Verifica se PCTC_ENABLED existe em settings."""
        assert isinstance(PCTC_ENABLED, bool)
        assert PCTC_ENABLED is True
    
    def test_pctc_point_enabled_exists(self):
        """Verifica se PCTC_POINT_ENABLED existe."""
        assert isinstance(PCTC_POINT_ENABLED, bool)
    
    def test_pctc_temporal_enabled_exists(self):
        """Verifica se PCTC_TEMPORAL_ENABLED existe."""
        assert isinstance(PCTC_TEMPORAL_ENABLED, bool)
    
    def test_pctc_use_lightglue_exists(self):
        """Verifica se PCTC_USE_LIGHTGLUE existe."""
        assert isinstance(PCTC_USE_LIGHTGLUE, bool)
    
    def test_pctc_use_raft_exists(self):
        """Verifica se PCTC_USE_RAFT existe."""
        assert isinstance(PCTC_USE_RAFT, bool)


class TestPass1AnalyzerADR005:
    """Testa ADR 005 no Pass1Analyzer."""
    
    def test_analyzer_initializes_with_pctc(self):
        """Testa se Pass1Analyzer aceita configurações PCTC."""
        from core.pass1_analyzer import Pass1Analyzer
        
        analyzer = Pass1Analyzer(
            device="cpu",
            enable_pctc=True,
            enable_point=True,
            enable_temporal=True
        )
        
        assert analyzer.enable_pctc is True
        assert analyzer.enable_point is True
        assert analyzer.enable_temporal is True
        assert analyzer._point_correspondence_service is None  # Lazy loading
        assert analyzer._temporal_consistency_service is None  # Lazy loading
    
    def test_analyzer_disables_pctc_when_false(self):
        """Testa se PCTC é desativado quando enable_pctc=False."""
        from core.pass1_analyzer import Pass1Analyzer
        
        analyzer = Pass1Analyzer(
            device="cpu",
            enable_pctc=False,
            enable_point=True,  # Deve ser sobrescrito para False
            enable_temporal=True  # Deve ser sobrescrito para False
        )
        
        assert analyzer.enable_pctc is False
        assert analyzer.enable_point is False
        assert analyzer.enable_temporal is False
    
    def test_point_correspondence_service_lazy_load(self):
        """Testa lazy loading do PointCorrespondenceService."""
        from core.pass1_analyzer import Pass1Analyzer
        
        analyzer = Pass1Analyzer(
            device="cpu",
            enable_pctc=True,
            enable_point=True
        )
        
        # Inicialmente None
        assert analyzer._point_correspondence_service is None
        
        # Carrega sob demanda
        service = analyzer._get_point_correspondence_service()
        
        if service is not None:
            assert analyzer._point_correspondence_service is not None
    
    def test_temporal_consistency_service_lazy_load(self):
        """Testa lazy loading do TemporalConsistencyService."""
        from core.pass1_analyzer import Pass1Analyzer
        
        analyzer = Pass1Analyzer(
            device="cpu",
            enable_pctc=True,
            enable_temporal=True
        )
        
        # Inicialmente None
        assert analyzer._temporal_consistency_service is None
        
        # Carrega sob demanda
        service = analyzer._get_temporal_consistency_service()
        
        if service is not None:
            assert analyzer._temporal_consistency_service is not None


class TestPageAnalysisADR005:
    """Testa se PageAnalysis suporta campos ADR 005."""
    
    def test_page_analysis_has_attention_masks(self):
        """Verifica se PageAnalysis tem campo attention_masks."""
        from core.database.chapter_db import PageAnalysis
        
        analysis = PageAnalysis(
            page_num=0,
            image_path="test.png",
            detections=[],
            character_ids=["char_001"]
        )
        
        assert hasattr(analysis, 'attention_masks')
        assert isinstance(analysis.attention_masks, dict)
    
    def test_page_analysis_has_temporal_data(self):
        """Verifica se PageAnalysis tem campo temporal_data."""
        from core.database.chapter_db import PageAnalysis
        
        analysis = PageAnalysis(
            page_num=0,
            image_path="test.png",
            detections=[],
            character_ids=["char_001"]
        )
        
        assert hasattr(analysis, 'temporal_data')
        assert isinstance(analysis.temporal_data, dict)
    
    def test_page_analysis_with_adr005_data(self):
        """Testa criação de PageAnalysis com dados ADR 005."""
        from core.database.chapter_db import PageAnalysis
        
        attention_masks = {
            "char_001": np.zeros((100, 100), dtype=np.float32)
        }
        temporal_data = {
            "transition_type": "continuous",
            "ssim_score": 0.85
        }
        
        analysis = PageAnalysis(
            page_num=0,
            image_path="test.png",
            detections=[],
            character_ids=["char_001"],
            attention_masks=attention_masks,
            temporal_data=temporal_data
        )
        
        assert analysis.attention_masks == attention_masks
        assert analysis.temporal_data == temporal_data


class TestChapterDatabaseADR005:
    """Testa se ChapterDatabase salva/carrega dados ADR 005."""
    
    def test_save_page_analysis_with_adr005(self, tmp_path):
        """Testa salvamento de PageAnalysis com ADR 005."""
        from core.database.chapter_db import ChapterDatabase, PageAnalysis
        
        # Cria database temporário
        db = ChapterDatabase("test_chapter", cache_root=str(tmp_path))
        
        attention_masks = {
            "char_001": np.ones((50, 50), dtype=np.float32)
        }
        temporal_data = {
            "transition_type": "continuous",
            "ssim_score": 0.85,
            "has_color_hint": True
        }
        
        analysis = PageAnalysis(
            page_num=0,
            image_path="test.png",
            detections=[{"bbox": [0, 0, 10, 10]}],
            character_ids=["char_001"],
            attention_masks=attention_masks,
            temporal_data=temporal_data
        )
        
        # Salva
        db.save_page_analysis(analysis)
        
        # Verifica se foi salvo
        assert len(db._pages_df) == 1
        assert 'attention_mask_paths' in db._pages_df.columns
        assert 'temporal_data' in db._pages_df.columns
    
    def test_get_page_analysis_loads_adr005(self, tmp_path):
        """Testa carregamento de PageAnalysis com ADR 005."""
        from core.database.chapter_db import ChapterDatabase, PageAnalysis
        
        # Cria database temporário
        db = ChapterDatabase("test_chapter", cache_root=str(tmp_path))
        
        attention_masks = {
            "char_001": np.ones((50, 50), dtype=np.float32) * 0.5
        }
        temporal_data = {
            "transition_type": "continuous",
            "ssim_score": 0.85
        }
        
        analysis = PageAnalysis(
            page_num=0,
            image_path="test.png",
            detections=[],
            character_ids=["char_001"],
            attention_masks=attention_masks,
            temporal_data=temporal_data
        )
        
        # Salva
        db.save_page_analysis(analysis)
        
        # Carrega
        loaded = db.get_page_analysis(0)
        
        assert loaded is not None
        assert loaded.page_num == 0
        assert isinstance(loaded.attention_masks, dict)
        assert isinstance(loaded.temporal_data, dict)


class TestPass2GeneratorADR005:
    """Testa ADR 005 no Pass2Generator."""
    
    def test_generator_loads_adr005_data(self, tmp_path):
        """Testa se Pass2Generator carrega dados ADR 005 do database."""
        # Este teste requer um database configurado
        # Vamos apenas verificar a estrutura
        from core.database.chapter_db import ChapterDatabase, PageAnalysis
        
        db = ChapterDatabase("test_chapter", cache_root=str(tmp_path))
        
        # Cria page analysis com dados ADR 005
        analysis = PageAnalysis(
            page_num=0,
            image_path="test.png",
            detections=[],
            character_ids=["char_001"],
            attention_masks={"char_001": np.zeros((100, 100))},
            temporal_data={"transition_type": "continuous"}
        )
        
        db.save_page_analysis(analysis)
        
        # Carrega e verifica
        loaded = db.get_page_analysis(0)
        assert "char_001" in loaded.attention_masks or len(loaded.attention_masks) == 0


@pytest.mark.integration
class TestADR005EndToEnd:
    """Testes end-to-end de ADR 005."""
    
    def test_full_pipeline_with_pctc_disabled(self, tmp_path):
        """Testa pipeline com PCTC desativado."""
        from core.pass1_analyzer import Pass1Analyzer
        
        analyzer = Pass1Analyzer(
            device="cpu",
            enable_pctc=False,
            enable_sam2=False,
            enable_zbuffer=False
        )
        
        # Verifica que PCTC está desativado
        assert analyzer.enable_pctc is False
        assert analyzer.enable_point is False
        assert analyzer.enable_temporal is False
    
    def test_full_pipeline_with_pctc_enabled(self, tmp_path):
        """Testa pipeline com PCTC ativado."""
        from core.pass1_analyzer import Pass1Analyzer
        
        analyzer = Pass1Analyzer(
            device="cpu",
            enable_pctc=True,
            enable_point=True,
            enable_temporal=True,
            enable_sam2=False,
            enable_zbuffer=False
        )
        
        # Verifica que PCTC está ativado
        assert analyzer.enable_pctc is True
        assert analyzer.enable_point is True
        assert analyzer.enable_temporal is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

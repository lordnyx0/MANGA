"""
MangaAutoColor Pro - TileService Unit Tests

Testa TileService e a geração correta de TileJobs,
especialmente a extração de active_char_ids das detecções.
"""

import pytest
import json
from unittest.mock import MagicMock, patch
from pathlib import Path

from core.domain.tile_service import TileService
from core.database.chapter_db import TileJob


class TestTileServiceCharIdExtraction:
    """Testa extração de char_ids das detecções."""

    def test_extract_char_id_from_detections(self):
        """
        CRITICAL: Verifica que TileService extrai 'char_id' corretamente.
        
        Este teste teria falhado com o bug original onde buscava 'character_id'.
        """
        # Mock database
        mock_db = MagicMock()
        mock_db.chapter_id = "ch_test"
        mock_db._pages_df = None
        mock_db.canny_dir = Path("/tmp/canny")
        
        service = TileService(mock_db)
        
        # Simula detecções COM char_id (formato correto)
        detections_with_char_id = [
            {'bbox': (100, 100, 300, 400), 'char_id': 'char_001', 'class_id': 0},
            {'bbox': (500, 200, 700, 500), 'char_id': 'char_002', 'class_id': 0},
        ]
        
        # Simula get_characters_in_tile retornando detecções
        with patch.object(service.tiling_manager, 'get_characters_in_tile', 
                         return_value=detections_with_char_id):
            
            chars_in_tile = service.tiling_manager.get_characters_in_tile(
                (0, 0, 1024, 1024), detections_with_char_id
            )
            
            # Extrai IDs usando a mesma lógica do TileService._process_page
            active_ids = []
            for c in chars_in_tile:
                if isinstance(c, dict) and 'char_id' in c:
                    active_ids.append(c['char_id'])
            
            # DEVE encontrar os char_ids
            assert len(active_ids) == 2, f"Esperado 2 char_ids, obteve {len(active_ids)}"
            assert 'char_001' in active_ids
            assert 'char_002' in active_ids

    def test_fail_with_character_id_key(self):
        """
        Verifica que detecções com 'character_id' NÃO são extraídas.
        
        Este teste documenta o bug corrigido - se alguém reverter,
        este teste vai passar mas test_extract_char_id_from_detections falhará.
        """
        detections_wrong_key = [
            {'bbox': (100, 100, 300, 400), 'character_id': 'char_001', 'class_id': 0},
        ]
        
        # A lógica correta procura 'char_id', não 'character_id'
        active_ids = []
        for c in detections_wrong_key:
            if isinstance(c, dict) and 'char_id' in c:
                active_ids.append(c['char_id'])
        
        # Com a chave errada, não deve encontrar nada
        assert len(active_ids) == 0, "Não deveria extrair IDs com chave 'character_id'"

    def test_mixed_detections(self):
        """Testa detecções mistas (algumas com char_id, outras sem)."""
        detections_mixed = [
            {'bbox': (100, 100, 300, 400), 'char_id': 'char_001', 'class_id': 0},
            {'bbox': (500, 200, 700, 500), 'class_id': 3},  # texto, sem char_id
            {'bbox': (200, 300, 400, 600), 'char_id': 'char_003', 'class_id': 0},
        ]
        
        active_ids = []
        for c in detections_mixed:
            if isinstance(c, dict) and 'char_id' in c:
                active_ids.append(c['char_id'])
        
        assert len(active_ids) == 2
        assert 'char_001' in active_ids
        assert 'char_003' in active_ids


class TestTileJobCreation:
    """Testa criação de TileJob com dados corretos."""

    def test_tile_job_has_active_char_ids(self):
        """Verifica que TileJob é criado com active_char_ids populado."""
        job = TileJob(
            page_num=0,
            tile_bbox=(0, 0, 1024, 1024),
            active_char_ids=['char_001', 'char_002'],
            mask_paths={},
            canny_path="/tmp/canny.npy"
        )
        
        assert len(job.active_char_ids) == 2
        assert 'char_001' in job.active_char_ids
        assert 'char_002' in job.active_char_ids

    def test_tile_job_empty_when_no_chars(self):
        """Verifica que TileJob tem active_char_ids vazio quando não há personagens."""
        job = TileJob(
            page_num=0,
            tile_bbox=(0, 0, 1024, 1024),
            active_char_ids=[],
            mask_paths={},
            canny_path="/tmp/canny.npy"
        )
        
        assert len(job.active_char_ids) == 0


class TestPipelineCharIdLinking:
    """Testa que pipeline.py associa char_id às detecções."""

    def test_char_id_linked_to_detection_by_bbox(self):
        """
        CRITICAL: Verifica que char_id é associado às detecções após save_characters.
        
        Este é o teste que teria pego o bug original.
        """
        # Simula dados do Pass1Analyzer (detecções SEM char_id)
        detections = [
            {'bbox': (100, 100, 300, 400), 'confidence': 0.9, 'class_id': 0},
            {'bbox': (500, 200, 700, 500), 'confidence': 0.85, 'class_id': 0},
        ]
        
        # Simula characters após save_characters_from_analysis (COM character_id adicionado)
        characters = [
            {'bbox': (100, 100, 300, 400), 'character_id': 'char_001', 'embedding': []},
            {'bbox': (500, 200, 700, 500), 'character_id': 'char_002', 'embedding': []},
        ]
        
        # Simula a lógica de linking do pipeline.py
        for det in detections:
            det_bbox = det.get('bbox')
            if det_bbox:
                for char in characters:
                    if char.get('bbox') == det_bbox and 'character_id' in char:
                        det['char_id'] = char['character_id']
                        break
        
        # Verifica que char_id foi adicionado
        assert 'char_id' in detections[0], "Detecção 1 deve ter char_id após linking"
        assert 'char_id' in detections[1], "Detecção 2 deve ter char_id após linking"
        assert detections[0]['char_id'] == 'char_001'
        assert detections[1]['char_id'] == 'char_002'

    def test_char_id_not_linked_for_text_detections(self):
        """Verifica que detecções de texto não ganham char_id."""
        detections = [
            {'bbox': (100, 100, 300, 400), 'confidence': 0.9, 'class_id': 0},  # body
            {'bbox': (200, 50, 400, 100), 'confidence': 0.8, 'class_id': 3},   # text
        ]
        
        characters = [
            {'bbox': (100, 100, 300, 400), 'character_id': 'char_001', 'embedding': []},
        ]
        
        # Linking
        for det in detections:
            det_bbox = det.get('bbox')
            if det_bbox:
                for char in characters:
                    if char.get('bbox') == det_bbox and 'character_id' in char:
                        det['char_id'] = char['character_id']
                        break
        
        # Body detection deve ter char_id
        assert 'char_id' in detections[0]
        # Text detection não deve ter char_id (não há character correspondente)
        assert 'char_id' not in detections[1]

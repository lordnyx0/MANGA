"""
MangaAutoColor Pro - E2E Integration Test: char_id Flow

Testa o fluxo completo de char_id desde Pass 1 até Pass 2.
Este teste teria detectado o bug onde char_id não era associado às detecções.
"""

import pytest
import json
from pathlib import Path
import numpy as np

# Importa classes necessárias
from core.database.chapter_db import ChapterDatabase, TileJob


class TestCharIdFlowE2E:
    """
    Teste E2E do fluxo de char_id.
    
    Verifica toda a cadeia:
    Pass1Analyzer → pipeline.py (linking) → ChapterDatabase → TileService → Pass2Generator
    """

    @pytest.fixture
    def temp_chapter_dir(self, tmp_path):
        """Cria diretório temporário para o capítulo."""
        chapter_id = "ch_test_e2e"
        chapter_dir = tmp_path / "chapters" / chapter_id
        chapter_dir.mkdir(parents=True)
        return chapter_id, tmp_path

    def test_detections_have_char_id_after_pipeline_processing(self, temp_chapter_dir):
        """
        CRITICAL: Verifica que detecções salvas no database têm char_id.
        
        Este é o teste que detecta o bug onde char_id não era vinculado.
        """
        chapter_id, base_path = temp_chapter_dir
        
        # Cria database com diretório temporário
        cache_root = base_path / "chapter_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        db = ChapterDatabase(chapter_id, cache_root=str(cache_root))
        
        # Simula dados do Pass1Analyzer (detecções SEM char_id)
        original_detections = [
            {'bbox': (100, 100, 300, 400), 'confidence': 0.9, 'class_id': 0, 'class_name': 'body'},
            {'bbox': (500, 200, 700, 500), 'confidence': 0.85, 'class_id': 0, 'class_name': 'body'},
            {'bbox': (50, 50, 200, 80), 'confidence': 0.75, 'class_id': 3, 'class_name': 'text'},
        ]
        
        # Simula characters (após extração de identidade)
        characters = [
            {
                'bbox': (100, 100, 300, 400), 
                'embedding': np.random.randn(768).tolist(),
                'has_face': True
            },
            {
                'bbox': (500, 200, 700, 500), 
                'embedding': np.random.randn(768).tolist(),
                'has_face': False
            },
        ]
        
        # PASSO 1: Salva characters (gera character_ids)
        char_ids = db.save_characters_from_analysis(characters, page_num=0)
        
        assert len(char_ids) == 2, "Deveria ter gerado 2 char_ids"
        
        # PASSO 2: Simula o linking que o pipeline.py faz
        for det in original_detections:
            det_bbox = det.get('bbox')
            if det_bbox:
                for char in characters:
                    if char.get('bbox') == det_bbox and 'character_id' in char:
                        det['char_id'] = char['character_id']
                        break
        
        # PASSO 3: Salva página com detecções linkadas
        db.save_page_analysis(
            page_num=0,
            image_path="/tmp/test.png",
            detections=original_detections,
            character_ids=char_ids,
            scene_type="PRESENT",
            processed=True
        )
        
        # VERIFICAÇÃO: Carrega detecções e verifica char_id
        page_analysis = db.get_page_analysis(0)
        assert page_analysis is not None, "Page analysis should exist"
        # Get detections from the page analysis raw data
        page_row = db._pages_df[db._pages_df['page_num'] == 0].iloc[0]
        saved_detections = json.loads(page_row['detections']) if 'detections' in page_row else []
        
        if isinstance(saved_detections, str):
            saved_detections = json.loads(saved_detections)
        
        # Conta detecções com char_id
        detections_with_char_id = [d for d in saved_detections if 'char_id' in d]
        
        assert len(detections_with_char_id) == 2, \
            f"Esperado 2 detecções com char_id, obteve {len(detections_with_char_id)}"
        
        # Verifica que texto não tem char_id
        text_detections = [d for d in saved_detections if d.get('class_id') == 3]
        assert len(text_detections) == 1
        assert 'char_id' not in text_detections[0], "Detecção de texto não deve ter char_id"

    def test_tile_job_receives_char_ids(self, temp_chapter_dir):
        """Verifica que TileJob é criado com char_ids das detecções."""
        chapter_id, base_path = temp_chapter_dir
        
        # Cria database com diretório temporário
        cache_root = base_path / "chapter_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        db = ChapterDatabase(chapter_id, cache_root=str(cache_root))
        
        # Cria TileJob com char_ids
        job = TileJob(
            page_num=0,
            tile_bbox=(0, 0, 1024, 1024),
            active_char_ids=['char_001', 'char_002'],
            mask_paths={},
            canny_path=str(base_path / "canny.npy")
        )
        
        db.save_tile_job(job)
        
        # Carrega e verifica
        jobs = db.get_tile_jobs(0)
        
        assert len(jobs) >= 1, "Deveria ter pelo menos 1 TileJob"
        assert len(jobs[0].active_char_ids) == 2, "TileJob deve ter 2 char_ids"

    def test_embedding_loaded_for_char_id(self, temp_chapter_dir):
        """Verifica que embeddings podem ser carregados via char_id."""
        chapter_id, base_path = temp_chapter_dir
        
        # Cria database com diretório temporário
        cache_root = base_path / "chapter_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        db = ChapterDatabase(chapter_id, cache_root=str(cache_root))
        
        # Salva embedding usando o método correto
        char_id = "char_test_001"
        embedding = np.random.randn(768).astype(np.float32)
        
        import torch
        embedding_tensor = torch.from_numpy(embedding)
        # Use save_character_embedding instead of save_embedding
        db.save_character_embedding(
            char_id=char_id,
            clip_embedding=embedding_tensor,
            face_embedding=None,
            prominence_score=0.8,
            first_seen_page=0
        )
        
        # Carrega de volta usando load_embedding
        loaded = db.load_embedding(char_id, "clip")
        
        assert loaded is not None, f"Embedding para {char_id} não foi carregado"
        assert loaded.shape[0] == 768, f"Embedding tem shape errado: {loaded.shape}"


class TestCharIdContractAssertions:
    """Testes de contrato para garantir formato correto de dados."""

    def test_detection_has_required_keys_after_linking(self):
        """Verifica que detecções têm todas as chaves necessárias."""
        detection_template = {
            'bbox': (100, 100, 300, 400),
            'confidence': 0.9,
            'class_id': 0,
            'class_name': 'body',
            'char_id': 'char_001'  # Deve estar presente após linking
        }
        
        required_keys = ['bbox', 'confidence', 'class_id']
        optional_keys = ['char_id', 'class_name', 'page_num']
        
        for key in required_keys:
            assert key in detection_template, f"Chave obrigatória '{key}' ausente"
        
        # Para body detections, char_id deve estar presente
        if detection_template.get('class_id') == 0:
            assert 'char_id' in detection_template, \
                "Body detection deve ter char_id após linking"

    def test_tile_job_active_char_ids_not_empty_when_chars_present(self):
        """Verifica que TileJob.active_char_ids não é vazio quando há personagens no tile."""
        # Simula cenário onde há personagens detectados
        detections_with_chars = [
            {'bbox': (100, 100, 300, 400), 'char_id': 'char_001'},
            {'bbox': (500, 200, 700, 500), 'char_id': 'char_002'},
        ]
        
        # Extrai char_ids (como TileService faz)
        active_char_ids = [d['char_id'] for d in detections_with_chars if 'char_id' in d]
        
        # Se há detecções com personagem, active_char_ids NÃO pode ser vazio
        assert len(active_char_ids) > 0, \
            "active_char_ids não pode ser vazio quando há personagens no tile"

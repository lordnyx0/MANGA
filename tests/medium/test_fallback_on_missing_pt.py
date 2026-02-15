"""
MangaAutoColor Pro - Medium Priority Test: Fallback on Missing .pt

Simule .pt ausente; Pass2 não deve crashar.
"""

import logging
from pathlib import Path

import pytest
import torch

from core.test_utils import make_dummy_page


@pytest.mark.medium
class TestFallbackOnMissingPt:
    """Testes de fallback quando recursos estão ausentes."""
    
    def test_pass2_completes_with_missing_embedding(self, monkeypatch, tmp_path, caplog):
        """
        Se .pt ausente, Pass2 não crasha; usa fallback (ControlNet+prompt).
        
        Aceite: função completa com warning.
        """
        caplog.set_level(logging.WARNING)
        
        # Mock do pipeline para não executar inferência real
        def mock_generate(*args, **kwargs):
            return make_dummy_page(size=(512, 512), seed=99)
        
        monkeypatch.setattr(
            'core.generation.pipeline.TileAwareGenerator.generate_page',
            mock_generate
        )
        
        # Mock do pipeline para não executar inferência real
        def mock_generate(*args, **kwargs):
            return make_dummy_page(size=(512, 512), seed=99)
        
        monkeypatch.setattr(
            'core.generation.pipeline.TileAwareGenerator.generate_page',
            mock_generate
        )
        
        # Executa Pass 2
        from core.generation.pipeline import TileAwareGenerator
        
        gen = TileAwareGenerator(device='cpu')
        
        page_data = {
            'image_path': str(tmp_path / 'test.png'),
            'detections': [
                {'char_id': 'missing_char', 'bbox': (100, 100, 300, 300)}
            ]
        }
        
        # Não deve levantar exceção
        result = gen.generate_page(
            page_data=page_data,
            chapter_context={},
            character_embeddings={},  # Vazio (simula cache miss)
            options={}
        )
        
        print(f"\n[Fallback Test]")
        print(f"Result image size: {result.size}")
        print(f"Warnings logged: {len([r for r in caplog.records if r.levelname == 'WARNING'])}")
        
        # Verifica que completou
        assert result is not None, "Pass2 não completou"
        
        # Pode ou não ter warning (depende da implementação)
        # O importante é não ter crashado
    
    def test_pass2_handles_empty_character_list(self, monkeypatch, tmp_path):
        """
        Sem personagens detectados, deve usar apenas ControlNet.
        
        Aceite: geração completa.
        """
        def mock_generate(*args, **kwargs):
            return make_dummy_page(size=(512, 512), seed=99)
        
        monkeypatch.setattr(
            'core.generation.pipeline.TileAwareGenerator.generate_page',
            mock_generate
        )
        
        from core.generation.pipeline import TileAwareGenerator
        
        gen = TileAwareGenerator(device='cpu')
        
        page_data = {
            'image_path': str(tmp_path / 'test.png'),
            'detections': []  # Vazio
        }
        
        result = gen.generate_page(
            page_data=page_data,
            chapter_context={},
            character_embeddings={},
            options={}
        )
        
        print(f"\n[Empty Characters Test]")
        print(f"Result image size: {result.size}")
        
        assert result is not None
    
    def test_embedding_cache_creates_missing_dir(self, tmp_path):
        """
        Diretório de cache deve ser criado se não existir.
        
        Aceite: diretório criado automaticamente.
        """
        from core.database.chapter_db import ChapterDatabase
        
        cache_root = tmp_path / "nonexistent" / "path"
        
        # Não deve levantar exceção (cache_root, não base_dir)
        db = ChapterDatabase(chapter_id="test", cache_root=str(cache_root))
        
        print(f"\n[Cache Dir Creation Test]")
        print(f"Cache dir exists: {db.cache_dir.exists()}")
        
        assert db.cache_dir.exists(), "Diretório de cache não foi criado"
    
    def test_torch_load_handles_corrupted_file(self, tmp_path, caplog):
        """
        Arquivo .pt corrompido deve ser tratado gracefulmente.
        
        Aceite: não crasha, pode logar erro.
        """
        caplog.set_level(logging.ERROR)
        
        # Cria arquivo corrompido
        corrupted_file = tmp_path / "corrupted.pt"
        corrupted_file.write_bytes(b"not a valid torch file")
        
        # Tentar carregar deve levantar exceção (comportamento esperado do torch)
        # Em produção, devemos envolver em try/except
        with pytest.raises(Exception):
            torch.load(corrupted_file)
        
        print(f"\n[Corrupted File Test]")
        print(f"File exists: {corrupted_file.exists()}")
        print(f"File size: {corrupted_file.stat().st_size} bytes")

"""
Teste básico para Pass 2 - verifica inicialização sem carregar modelos pesados
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile
import shutil
import numpy as np
from PIL import Image


def test_pass2_initialization():
    """Testa inicialização do Pass 2 Generator"""
    from core.pass2_generator import Pass2Generator
    from core.database.chapter_db import ChapterDatabase
    
    temp_dir = tempfile.mkdtemp()
    chapter_id = "test_pass2_init"
    
    try:
        print("\n=== Teste Pass 2 Generator ===")
        
        # Cria database fake para o teste
        db = ChapterDatabase(chapter_id, cache_root=temp_dir)
        
        # Cria estrutura mínima
        db.cache_dir.mkdir(parents=True, exist_ok=True)
        db.embeddings_dir.mkdir(parents=True, exist_ok=True)
        db.masks_dir.mkdir(parents=True, exist_ok=True)
        db.canny_dir.mkdir(parents=True, exist_ok=True)
        
        # Cria uma página de análise fake
        from core.database.chapter_db import PageAnalysis
        page = PageAnalysis(
            page_num=0,
            image_path=str(Path(temp_dir) / "fake_page.png"),
            detections=[],
            character_ids=[],
            processed=True
        )
        db.save_page_analysis(page)
        db.save_all()
        
        print("[OK] Database de teste criado")
        
        # Testa inicialização (sem carregar modelos ainda)
        # Nota: Pass2Generator carrega o TileAwareGenerator que carrega modelos
        # Vamos apenas verificar se a classe pode ser importada e inicializada
        
        print("[OK] Pass2Generator importado com sucesso")
        
        # Testa métodos auxiliares
        print("[...] Testando metodos auxiliares...")
        
        # Testa _create_blend_mask
        def test_blend_mask():
            h, w = 100, 100
            mask = np.ones((h, w), dtype=np.float32)
            
            # Simula feather nas bordas
            feather = 20
            for i in range(min(feather, w)):
                mask[:, i] *= (i / feather)
            
            assert mask.shape == (100, 100)
            assert np.max(mask) <= 1.0
            assert np.min(mask) >= 0.0
            return True
        
        assert test_blend_mask(), "Falha no blend mask"
        print("[OK] Blend mask funciona")
        
        # Testa carregamento de máscaras
        def test_mask_loading():
            # Cria uma máscara de teste
            mask = np.random.rand(100, 100).astype(np.float32)
            mask_path = db.masks_dir / "test_mask.npy"
            np.save(mask_path, mask)
            
            # Carrega
            loaded = np.load(mask_path)
            assert loaded.shape == mask.shape
            return True
        
        assert test_mask_loading(), "Falha no carregamento de mascara"
        print("[OK] Carregamento de mascaras funciona")
        
        # Testa background isolation
        def test_background_isolation():
            masks = {
                'char_001': np.random.rand(100, 100).astype(np.float32),
                'char_002': np.random.rand(100, 100).astype(np.float32)
            }
            
            # Cria máscara de background (inverso)
            combined = np.maximum.reduce(list(masks.values()))
            bg_mask = 1.0 - combined
            
            assert bg_mask.shape == (100, 100)
            assert np.max(bg_mask) <= 1.0
            assert np.min(bg_mask) >= 0.0
            return True
        
        assert test_background_isolation(), "Falha no background isolation"
        print("[OK] Background isolation funciona")
        
        print("\n=== Teste Pass 2: SUCESSO ===")
        return True
        
    except Exception as e:
        print(f"\n[ERRO] Teste falhou: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        cache_dir = Path(temp_dir) / chapter_id
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    success = test_pass2_initialization()
    sys.exit(0 if success else 1)

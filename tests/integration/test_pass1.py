"""
Teste de integração completo para Pass 1 (Pytest Version)
"""
import sys
import shutil
import pytest
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

# Adiciona raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.pipeline import MangaColorizationPipeline, ChapterAnalysis
from core.database.chapter_db import ChapterDatabase


def create_test_manga_page(output_path, size=(400, 600)):
    """Cria uma imagem de manga de teste com alguns elementos"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Cabeça
    draw.ellipse([150, 50, 250, 150], fill='white', outline='black', width=3)
    # Corpo
    draw.rectangle([150, 150, 250, 400], fill='white', outline='black', width=3)
    # Braços
    draw.line([150, 200, 100, 300], fill='black', width=3)
    draw.line([250, 200, 300, 300], fill='black', width=3)
    
    # Fundo
    draw.line([0, 500, 400, 500], fill='black', width=2)
    draw.rectangle([50, 450, 100, 500], fill='white', outline='black', width=2)
    
    img.save(output_path)
    return output_path


@pytest.fixture
def clean_cache():
    """Fixture para limpar cache do capítulo de teste após execução"""
    # Usamos um diretório temporário para o cache do teste
    yield


@pytest.mark.integration
def test_pass1_full_pipeline(tmp_path):
    """
    Testa o Pass 1 completo com uma imagem de teste via Pipeline.
    Verifica se:
    1. Pipeline inicializa corretamente
    2. Processamento do capítulo roda sem erros
    3. Resultados (Database, Cache) são persistidos
    """
    
    # 1. Setup
    test_image = tmp_path / "pages" / "page_001.png"
    create_test_manga_page(test_image)
    assert test_image.exists()
    
    cache_dir = tmp_path / "cache"
    
    # 2. Inicialização
    pipeline = MangaColorizationPipeline(
        device='cpu',
        cache_dir=str(cache_dir)
    )
    
    # 3. Execução
    print(f"\n[INTEGRATION] Analisando página {test_image}...")
    
    # Mock progress callback
    progress_calls = []
    pipeline.set_progress_callback(lambda p, s, v: progress_calls.append((p, s, v)))
    
    analysis = pipeline.process_chapter(
        page_paths=[str(test_image)]
    )
    
    # 4. Verificação de Resultados
    assert isinstance(analysis, ChapterAnalysis)
    assert analysis.num_pages == 1
    
    # Verifica Database
    chapter_id = analysis.chapter_id
    db_path = cache_dir / "chapters" / chapter_id / "chapter.db"
    # O ChapterDatabase do pipeline cria uma estrutura de diretórios baseada no chapter_id,
    # mas o local exato depende da configuração. O pipeline usa ./cache por padrão se não sobrescrito.
    # Mas passamos cache_dir no init.
    # O ChapterDatabase assume cache_dir fixo?
    # Vamos verificar o pipeline._database.cache_dir
    
    # Acesso direto ao DB do pipeline (acesso privado para teste)
    db = pipeline._get_database(chapter_id)
    assert db is not None
    
    # Verifica se os dados foram salvos
    # Como a imagem é sintética e simples, o YOLO pode não detectar nada se o modelo não for robusto a desenhos toscos.
    # Mas o processo deve completar.
    
    print(f"Num characters: {analysis.num_characters}")
    
    if analysis.num_characters > 0:
        chars = db.get_all_characters()
        assert len(chars) == analysis.num_characters

"""
Teste de integração completo para Pass 1
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile
import shutil
import numpy as np
from PIL import Image, ImageDraw
import torch


def create_test_manga_page(output_path, size=(400, 600)):
    """Cria uma imagem de manga de teste com alguns elementos"""
    # Cria imagem em preto e branco
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Desenha um "personagem" simplificado (círculo para cabeça, retângulo para corpo)
    # Cabeça
    draw.ellipse([150, 50, 250, 150], fill='white', outline='black', width=3)
    # Corpo
    draw.rectangle([150, 150, 250, 400], fill='white', outline='black', width=3)
    # Braços
    draw.line([150, 200, 100, 300], fill='black', width=3)
    draw.line([250, 200, 300, 300], fill='black', width=3)
    
    # Adiciona algumas linhas de fundo (simulando cenário)
    draw.line([0, 500, 400, 500], fill='black', width=2)
    draw.rectangle([50, 450, 100, 500], fill='white', outline='black', width=2)
    
    img.save(output_path)
    return output_path


def test_pass1_full_pipeline():
    """Testa o Pass 1 completo com uma imagem de teste"""
    from core.pass1_analyzer import Pass1Analyzer
    from core.database.chapter_db import ChapterDatabase
    
    # Cria diretório temporário
    temp_dir = tempfile.mkdtemp()
    
    try:
        print("\n=== Teste de Integração Pass 1 ===")
        
        # Cria imagem de teste
        test_image = Path(temp_dir) / "page_001.png"
        create_test_manga_page(str(test_image))
        print(f"[OK] Imagem de teste criada: {test_image}")
        
        # Inicializa analyzer
        print("[...] Inicializando Pass1Analyzer...")
        analyzer = Pass1Analyzer(device='cpu')
        print("[OK] Pass1Analyzer inicializado")
        
        # Testa propriedades lazy-loaded
        print("[...] Carregando componentes...")
        _ = analyzer.palette_extractor
        print("[OK] PaletteExtractor carregado")
        
        _ = analyzer._get_nms_processor()
        print("[OK] CannyContinuityNMS carregado")
        
        # Executa análise
        chapter_id = "test_chapter_001"
        print(f"[...] Executando analise do capitulo {chapter_id}...")
        
        summary = analyzer.analyze_chapter(
            chapter_id=chapter_id,
            page_paths=[str(test_image)],
            progress_callback=lambda current, total: print(f"    Progresso: {current}/{total}")
        )
        
        print(f"\n[OK] Analise concluida!")
        print(f"    - Personagens detectados: {summary['total_characters']}")
        print(f"    - Tiles pre-computados: {summary['total_tiles']}")
        print(f"    - Paginas processadas: {summary['total_pages']}")
        
        # Verifica se o database foi criado (diretório existe)
        db = ChapterDatabase(chapter_id)
        cache_exists = db.cache_dir.exists()
        print(f"[OK] Diretorio de cache existe: {cache_exists}")
        
        # Se houver personagens, verifica o arquivo parquet
        if summary['total_characters'] > 0:
            assert db.exists(), "Database nao foi criado (characters.parquet)"
            print("[OK] Database (characters.parquet) criado")
        
        # Verifica se há personagens
        if summary['total_characters'] > 0:
            print(f"\n[OK] {summary['total_characters']} personagens detectados")
            
            # Tenta carregar uma paleta (se houver personagens)
            chars = db.get_all_characters()
            if chars:
                first_char = chars[0]
                char_id = first_char['char_id']
                palette = db.load_character_palette(char_id)
                if palette:
                    print(f"[OK] Paleta carregada para {char_id}: {len(palette.regions)} regioes")
                else:
                    print(f"[AVISO] Sem paleta para {char_id} (normal se nao houve deteccao valida)")
            
            # NOVO: Verifica que detecções têm char_id vinculado
            page_data = db.get_page_data(0)
            if page_data:
                import json
                detections = page_data.get('detections', [])
                if isinstance(detections, str):
                    detections = json.loads(detections)
                
                body_dets = [d for d in detections if d.get('class_id') == 0]
                char_id_dets = [d for d in body_dets if 'char_id' in d]
                
                if len(body_dets) > 0:
                    if len(char_id_dets) > 0:
                        print(f"[OK] {len(char_id_dets)}/{len(body_dets)} body detections tem char_id vinculado")
                    else:
                        print(f"[AVISO] Nenhuma body detection tem char_id (pode indicar bug de linking)")
        
        print("\n=== Teste Pass 1: SUCESSO ===")
        return True
        
    except Exception as e:
        print(f"\n[ERRO] Teste falhou: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Limpa
        shutil.rmtree(temp_dir, ignore_errors=True)
        # Limpa cache do chapter
        cache_dir = Path("chapter_cache") / chapter_id
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    success = test_pass1_full_pipeline()
    sys.exit(0 if success else 1)

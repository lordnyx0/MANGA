"""
MangaAutoColor Pro - CLI (Command Line Interface)

Interface de linha de comando para uso do sistema sem GUI.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional
import time

# Adiciona raiz do projeto
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.logging.setup import setup_logging
# Configura logging antes de importar outros módulos que podem usar loggers
setup_logging(verbose=True)

from core.pipeline import MangaColorizationPipeline, GenerationOptions
from config.settings import STYLE_PRESETS, DEVICE, DTYPE
from core.exceptions import AnalysisError, GenerationError, ModelLoadError, ResourceError


def print_progress(page_num: int, stage: str, progress: float):
    """Callback de progresso para CLI."""
    bar_length = 30
    filled = int(bar_length * progress / 100)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"\r  Página {page_num+1}: [{bar}] {progress:.1f}% - {stage}", end='', flush=True)
    if progress >= 100:
        print()  # Nova linha ao completar


def analyze_command(args):
    """Comando: analyze (Pass 1)"""
    print(f"[ANÁLISE] Iniciando Pass 1...")
    print(f"  Device: {DEVICE}")
    print(f"  Dtype: {DTYPE}")
    print(f"  Páginas: {len(args.input)}")
    
    pipeline = MangaColorizationPipeline(
        device=DEVICE,
        dtype=DTYPE,
        enable_xformers=True,
        enable_cpu_offload=True
    )
    
    pipeline.set_progress_callback(print_progress)
    
    try:
        start_time = time.time()
        analysis = pipeline.process_chapter(args.input)
        elapsed = time.time() - start_time
        
        print(f"\n[OK] Análise completa em {elapsed:.1f}s")
        print(f"  Personagens detectados: {analysis.num_characters}")
        print(f"  Tempo estimado de geração: {analysis.estimated_generation_time:.0f}s")
        
        # Lista personagens
        if analysis.characters:
            print("\n  Personagens:")
            for i, char in enumerate(analysis.characters[:10]):
                print(f"    {i+1}. {char.get('id', f'char_{i}')} "
                      f"({char.get('appearances', 1)} aparições)")
        
        return 0
        
    except AnalysisError as e:
        print(f"\n[ERRO] Falha na análise (Pass 1): {e}")
        return 1
    except ResourceError as e:
        print(f"\n[ERRO] Recursos insuficientes: {e}")
        return 2
    except Exception as e:
        print(f"\n[ERRO] Erro inesperado na análise: {e}")
        import traceback
        traceback.print_exc()
        return 1


def generate_command(args):
    """Comando: generate (Pass 2)"""
    print(f"[GERAÇÃO] Iniciando Pass 2...")
    print(f"  Chapter: {args.chapter_id}")
    print(f"  Estilo: {args.style}")
    print(f"  Qualidade: {args.quality}")
    
    pipeline = MangaColorizationPipeline(
        device=DEVICE,
        dtype=DTYPE,
        enable_xformers=True,
        enable_cpu_offload=True
    )
    
    pipeline.set_progress_callback(print_progress)
    
    options = GenerationOptions(
        style_preset=args.style,
        quality_mode=args.quality,
        seed=args.seed,
        preserve_original_text=not args.no_text_preserve,
        apply_narrative_transforms=not args.no_narrative
    )
    
    # Cria diretório de saída
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Se não especificou páginas, gera todas
        pages = args.pages if args.pages else None
        
        start_time = time.time()
        
        if pages:
            # Gera páginas específicas
            for page_num in pages:
                result = pipeline.generate_page(page_num - 1, options)  # 0-based
                output_path = output_dir / f"page_{page_num:03d}_colored.png"
                result.save(output_path)
                print(f"  Salvo: {output_path}")
        else:
            # Gera todas
            results = pipeline.generate_all(str(output_dir), options)
            print(f"\n[OK] {len(results)} páginas geradas")
        
        elapsed = time.time() - start_time
        print(f"\nTempo total: {elapsed:.1f}s")
        print(f"Saída: {output_dir.absolute()}")
        
        return 0
        
    except GenerationError as e:
        print(f"\n[ERRO] Falha na geração (Pass 2): {e}")
        return 1
    except ModelLoadError as e:
        print(f"\n[ERRO] Falha ao carregar modelo '{e.model_name}': {e}")
        return 2
    except ResourceError as e:
        print(f"\n[ERRO] Memória insuficiente: {e}")
        print("Sugestão: Tente --quality fast ou reduza o tamanho da imagem.")
        return 2
    except Exception as e:
        print(f"\n[ERRO] Erro inesperado na geração: {e}")
        import traceback
        traceback.print_exc()
        return 1


def full_command(args):
    """Comando: full (Pass 1 + Pass 2)"""
    print(f"[FULL PIPELINE] Executando Pass 1 + Pass 2...")
    
    # Pass 1
    if analyze_command(args) != 0:
        return 1
    
    print("\n" + "=" * 60)
    
    # Pass 2
    args.chapter_id = "auto"  # Usa chapter_id automático
    return generate_command(args)


def list_styles_command(args):
    """Comando: list-styles"""
    print("Estilos disponíveis:")
    print()
    
    for style_name, style_config in STYLE_PRESETS.items():
        print(f"  {style_name}")
        if style_config.get('prompt_addition'):
            print(f"    Prompt: {style_config['prompt_addition']}")
        if style_config.get('negative_addition'):
            print(f"    Negative: {style_config['negative_addition']}")
        print()
    
    return 0


def info_command(args):
    """Comando: info (informações do sistema)"""
    from config.settings import get_device_properties, TILE_SIZE, MAX_REF_PER_TILE
    
    device_info = get_device_properties()
    
    print("Informações do Sistema")
    print("=" * 40)
    print(f"Device: {device_info.get('name', 'CPU')}")
    print(f"VRAM: {device_info.get('total_memory_gb', 0):.1f} GB")
    print(f"CUDA: {device_info.get('major', 0)}.{device_info.get('minor', 0)}")
    print(f"PyTorch: {DEVICE} / {DTYPE}")
    print()
    print("Configurações:")
    print(f"  Tile size: {TILE_SIZE}")
    print(f"  Max chars/tile: {MAX_REF_PER_TILE}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="manga-autocolor",
        description="MangaAutoColor Pro - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Análise apenas (Pass 1)
  python cli.py analyze page_*.png

  # Geração apenas (Pass 2) - requer análise prévia
  python cli.py generate --chapter-id chapter_001 --output ./output

  # Pipeline completo
  python cli.py full page_*.png --output ./output --style vibrant

  # Geração de páginas específicas
  python cli.py generate --chapter-id chapter_001 --pages 1 5 10 --output ./output
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponíveis")
    
    # Comando: analyze
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Pass 1: Análise do capítulo"
    )
    analyze_parser.add_argument(
        "input",
        nargs="+",
        help="Arquivos de imagem das páginas"
    )
    analyze_parser.set_defaults(func=analyze_command)
    
    # Comando: generate
    generate_parser = subparsers.add_parser(
        "generate",
        help="Pass 2: Geração colorizada"
    )
    generate_parser.add_argument(
        "--chapter-id",
        required=True,
        help="ID do capítulo (do Pass 1)"
    )
    generate_parser.add_argument(
        "--pages",
        nargs="+",
        type=int,
        help="Números das páginas a gerar (default: todas)"
    )
    generate_parser.add_argument(
        "--output",
        "-o",
        default="./output",
        help="Diretório de saída"
    )
    generate_parser.add_argument(
        "--style",
        default="default",
        choices=list(STYLE_PRESETS.keys()),
        help="Estilo de colorização"
    )
    generate_parser.add_argument(
        "--quality",
        default="balanced",
        choices=["fast", "balanced", "high"],
        help="Modo de qualidade"
    )
    generate_parser.add_argument(
        "--seed",
        type=int,
        help="Seed para reprodutibilidade"
    )
    generate_parser.add_argument(
        "--no-text-preserve",
        action="store_true",
        help="Não preservar texto original"
    )
    generate_parser.add_argument(
        "--no-narrative",
        action="store_true",
        help="Não aplicar transformações narrativas"
    )
    generate_parser.set_defaults(func=generate_command)
    
    # Comando: full
    full_parser = subparsers.add_parser(
        "full",
        help="Pipeline completo (Pass 1 + Pass 2)"
    )
    full_parser.add_argument(
        "input",
        nargs="+",
        help="Arquivos de imagem das páginas"
    )
    full_parser.add_argument(
        "--output",
        "-o",
        default="./output",
        help="Diretório de saída"
    )
    full_parser.add_argument(
        "--style",
        default="default",
        choices=list(STYLE_PRESETS.keys()),
        help="Estilo de colorização"
    )
    full_parser.add_argument(
        "--quality",
        default="balanced",
        choices=["fast", "balanced", "high"],
        help="Modo de qualidade"
    )
    full_parser.add_argument(
        "--seed",
        type=int,
        help="Seed para reprodutibilidade"
    )
    full_parser.set_defaults(func=full_command)
    
    # Comando: list-styles
    styles_parser = subparsers.add_parser(
        "list-styles",
        help="Lista estilos disponíveis"
    )
    styles_parser.set_defaults(func=list_styles_command)
    
    # Comando: info
    info_parser = subparsers.add_parser(
        "info",
        help="Informações do sistema"
    )
    info_parser.set_defaults(func=info_command)
    
    # Parse args
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Executa comando
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

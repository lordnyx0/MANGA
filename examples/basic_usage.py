"""
MangaAutoColor Pro - Exemplo de Uso Básico

Este exemplo demonstra o uso básico da API Python.
"""

from pathlib import Path
from core.pipeline import MangaColorizationPipeline, GenerationOptions


def main():
    # Inicializa o pipeline
    print("Inicializando pipeline...")
    pipeline = MangaColorizationPipeline(
        device="cuda",  # ou "cpu"
        enable_xformers=True,
        enable_cpu_offload=True
    )
    
    # Define caminhos das páginas
    chapter_pages = [
        "./input/page_001.png",
        "./input/page_002.png",
        "./input/page_003.png",
    ]
    
    # Passo 1: Análise
    print("\n[Passo 1] Analisando capítulo...")
    analysis = pipeline.process_chapter(chapter_pages)
    
    print(f"  Páginas: {analysis.num_pages}")
    print(f"  Personagens: {analysis.num_characters}")
    print(f"  Tempo estimado: {analysis.estimated_generation_time:.0f}s")
    
    # Passo 2: Geração
    print("\n[Passo 2] Gerando páginas colorizadas...")
    
    options = GenerationOptions(
        style_preset="vibrant",
        quality_mode="balanced",
        seed=42
    )
    
    # Gera todas as páginas
    results = pipeline.generate_all("./output", options)
    
    print(f"\n[OK] {len(results)} páginas geradas em ./output")


if __name__ == "__main__":
    main()

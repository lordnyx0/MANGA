"""
MangaAutoColor Pro - Módulo de Geração

Exporta:
- TileAwareGenerator: Gerador com processamento por tiles
- TileGenerationResult: Resultado de geração de tile
"""

from core.generation.pipeline import TileAwareGenerator
from core.generation.types import TileGenerationResult

__all__ = ['TileAwareGenerator', 'TileGenerationResult']

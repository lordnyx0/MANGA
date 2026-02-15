"""
MangaAutoColor Pro - Módulo de Identidade

Exporta:
- HybridIdentitySystem: Extrator híbrido de identidades (CLIP + ArcFace)
- IdentityCache: Cache persistente de embeddings
- PaletteExtractor: Extrator de paletas de cores
- PaletteManager: Gerenciador de paletas com consistência temporal
"""

from .hybrid_encoder import (
    HybridIdentitySystem, 
    IdentityFeatures,
    IdentityCache
)
from .palette_manager import (
    PaletteExtractor,
    PaletteManager,
    CharacterPalette,
    ColorRegion,
    generate_prompt_from_palette
)

__all__ = [
    'HybridIdentitySystem',
    'IdentityFeatures',
    'IdentityCache',
    'PaletteExtractor',
    'PaletteManager',
    'CharacterPalette',
    'ColorRegion',
    'generate_prompt_from_palette'
]

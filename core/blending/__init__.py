"""
MangaAutoColor Pro - Módulo de Blending

Exporta:
- LatentBlender: Blending no espaço latente e pixel
- BlendRegion: Região para multi-band blending
- Funções utilitárias de blending
"""

from .latent_blender import (
    LatentBlender,
    BlendRegion,
    blend_images_average,
    create_transition_mask
)

__all__ = [
    'LatentBlender',
    'BlendRegion',
    'blend_images_average',
    'create_transition_mask'
]

"""
MangaAutoColor Pro - Tipos de Dados para Geração
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from PIL import Image

@dataclass
class TileGenerationResult:
    """Resultado da geração de um tile"""
    tile_id: str
    image: Image.Image
    bbox: tuple
    metadata: Dict[str, Any]

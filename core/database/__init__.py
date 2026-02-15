"""
MangaAutoColor Pro - Módulo de Database (Two-Pass)

Exporta:
- ChapterDatabase: Persistência híbrida FAISS + Parquet + .pt
- CharacterRecord: Registro de personagem
- TileJob: Job de processamento de tile
- PageAnalysis: Análise de página
"""

from .chapter_db import (
    ChapterDatabase,
    CharacterRecord,
    TileJob,
    PageAnalysis
)

__all__ = [
    'ChapterDatabase',
    'CharacterRecord',
    'TileJob',
    'PageAnalysis'
]

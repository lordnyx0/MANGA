"""
MangaAutoColor Pro - Exceções do Domínio
Centraliza todas as exceções personalizadas do sistema.
"""

from typing import Optional

class MangaColorError(Exception):
    """Exceção base para todo o domínio MangaAutoColor."""
    pass


class AnalysisError(MangaColorError):
    """Erro durante a etapa de análise (Pass 1)."""
    def __init__(self, message: str, page_num: Optional[int] = None):
        super().__init__(message)
        self.page_num = page_num


class GenerationError(MangaColorError):
    """Erro durante a etapa de geração (Pass 2)."""
    def __init__(self, message: str, page_num: Optional[int] = None):
        super().__init__(message)
        self.page_num = page_num


class ModelLoadError(MangaColorError):
    """Erro ao carregar modelos (DL/ML)."""
    def __init__(self, message: str, model_name: str):
        super().__init__(message)
        self.model_name = model_name


class ResourceError(MangaColorError):
    """Erro relacionado a recursos (VRAM, RAM, Disco)."""
    pass

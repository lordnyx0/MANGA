"""
MangaAutoColor Pro - Configuração de Logging
Fornece configuração centralizada para o módulo logging padrão do Python.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(name: str = "MangaAutoColor", verbose: bool = False, log_file: Optional[str] = None):
    """
    Configura o logger principal.
    
    Args:
        name: Nome do logger
        verbose: Se True, define nível para DEBUG
        log_file: Caminho opcional para arquivo de log
    """
    logger = logging.getLogger(name)
    
    # Evita duplicação de handlers se chamado múltiplas vezes
    if logger.handlers:
        return logger
        
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    
    # Formato
    console_format = logging.Formatter(
        '[%(levelname)s] %(message)s'
    )
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(console_format)
    logger.addHandler(ch)
    
    # File Handler (opcional)
    if log_file:
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)  # Arquivo sempre detalhado
        fh.setFormatter(file_format)
        logger.addHandler(fh)
        
    # Redireciona warnings do Python para o logger
    logging.captureWarnings(True)
    
    return logger

def get_logger(name: str):
    """Retorna um logger filho com o namespace correto."""
    return logging.getLogger(f"MangaAutoColor.{name}")

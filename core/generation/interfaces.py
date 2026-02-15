
from abc import ABC, abstractmethod
from PIL import Image
from typing import Dict, List, Optional, Any, Union
import numpy as np

class ColorizationEngine(ABC):
    """
    Interface abstrata para motores de colorização.
    Permite trocar entre implementações (ex: SDXL, SD1.5, Flux) sem afetar o resto do sistema.
    """
    
    @abstractmethod
    def load_models(self):
        """Carrega os modelos necessários na memória (VRAM/RAM)."""
        pass
        
    @abstractmethod
    def offload_models(self):
        """Descarrega modelos para economizar VRAM quando ocioso."""
        pass
        
    @abstractmethod
    def generate_page(
        self, 
        page_image: Image.Image,
        options: Dict[str, Any]
    ) -> Image.Image:
        """
        Gera uma página colorizada completa.
        
        Args:
            page_image: Imagem original (PIL)
            options: Dicionário de opções de geração
            
        Returns:
            Imagem colorizada (PIL)
        """
        pass
        
    @abstractmethod
    def generate_region(
        self,
        line_art: Image.Image,
        mask: Image.Image,
        reference_image: Optional[Image.Image],
        prompt: str,
        negative_prompt: str,
        seed: int
    ) -> Image.Image:
        """
        Gera colorização para uma região específica (inpainting).
        
        Args:
            line_art: Imagem base de line art (para condicionamento)
            mask: Máscara da região a colorir
            reference_image: Imagem de referência visual (IP-Adapter)
            prompt: Prompt textual positivo
            negative_prompt: Prompt textual negativo
            seed: Seed para reprodutibilidade
            
        Returns:
            Imagem da região colorizada (apenas a região, ou full image com região alterada)
        """
        pass

"""
MangaAutoColor Pro - Compatibilidade Pipeline V2/V3 (Realtime Adapter)

Este módulo fornece a classe TileAwareGenerator como um adapter para o SD15LineartEngine (V3),
permitindo que a API existente (projetada para V2/SDXL) continue funcionando
sem alterações profundas.
"""

import torch
from PIL import Image
from typing import List, Dict, Optional, Any
import numpy as np

from config.settings import DEVICE, DTYPE
from core.generation.engines.sd15_lineart_engine import SD15LineartEngine
from core.logging.setup import get_logger

logger = get_logger("TileAwareGeneratorAdapter")

class TileAwareGenerator:
    """
    Adapter que faz a ponte entre a API Realtime (SDXL-Lightning based)
    e o novo motor SD15LineartEngine.
    """
    
    def __init__(self, device: str = DEVICE, dtype=DTYPE, enable_offload: bool = True):
        self.device = device
        self.dtype = dtype
        self.engine = SD15LineartEngine(device=device, dtype=dtype)
        self.is_loaded = False
        
    def load_models(self):
        """Carrega o motor V3."""
        if not self.is_loaded:
            self.engine.load_models()
            self.is_loaded = True
            
    def unload(self):
        """Libera VRAM."""
        self.engine.offload_models()
        self.is_loaded = False

    def generate_image(
        self,
        image: Image.Image,
        character_embeddings: Dict[str, Any],
        detections: List[Dict],
        options: Any
    ) -> Image.Image:
        """
        Gera imagem colorizada usando o motor V3.
        
        Args:
            image: Imagem de entrada (linha/PB)
            character_embeddings: Dict de embeddings (ignorado no V3 Adapter por incompatibilidade)
            detections: Lista de detecções para encontrar a referência visual
            options: GenerationOptions (pydantic model) ou dict
        """
        if not self.is_loaded:
            self.load_models()
            
        # 1. Converte options para dict se for Pydantic model
        if hasattr(options, 'dict'):
            opts = options.dict()
        else:
            opts = dict(options) if options else {}

        # 2. Referência Visual (IP-Adapter)
        # O motor V3 espera uma IMAGEM de referência colorida.
        # Desativamos a extração de crop P&B da própria página, pois isso
        # "suja" o IP-Adapter com cinzas, resultando em cores lavadas.
        # Se não houver referência real, deixamos como None.
        reference_image = None

        # 3. Configurações Específicas V3
        opts['reference_image'] = reference_image
        
        # Override de passos para modo Realtime (se solicitado 4 steps, forçamos um mínimo aceitável para SD1.5)
        # SD1.5 Lightning/LCM pode funcionar em 4 steps, mas o modelo base sd1.5 precisa de ~20.
        # Se o usuário pediu 4 (preset realtime), vamos tentar usar turbo/lightning settings ou aceitar que vai ficar ruim?
        # Por enquanto, vamos respeitar o pedido, mas o SD15LineartEngine usa DDIM.
        # Se steps < 10, DDIM pode falhar em qualidade. 
        # Vamos logar e manter.
        if opts.get('num_inference_steps', 20) < 10:
            logger.warning("Solicitado steps < 10. SD1.5 pode gerar artefatos. Considere aumentar.")

        # 4. Geração (Color Layer)
        # O engine V3 já trata prompt, ip-adapter e controlnet
        color_layer = self.engine.generate_page(image, opts)
        
        # 5. Composição (Passo Multiply V3)
        # Garante que o traço original seja preservado
        # Passa detections para realizar o "Bubble Masking" (limpeza de balões)
        result = self.engine.compose_final(image, color_layer, detections=detections)
        
        return result

"""
Scene Detector - Detecção de contexto narrativo em páginas de mangá.

Analisa a composição visual da página para inferir o tipo de cena:
- present: Cena normal no presente
- flashback: Cena de flashback/memória
- dream: Cena de sonho
- nightmare: Cena de pesadelo
"""

import numpy as np
from PIL import Image
from typing import Dict, Any
from core.constants import SceneType


class SceneDetector:
    """
    Detector de tipo de cena baseado em análise visual.
    
    Usa heurísticas de:
    - Tonalidade geral da imagem (escura/clara)
    - Presença de efeitos visuais (brilhos, sombras)
    - Layout e composição
    """
    
    def __init__(self):
        self.scene_types = ['present', 'flashback', 'dream', 'nightmare']
    
    def detect(self, image: Image.Image) -> str:
        """
        Detecta o tipo de cena a partir da imagem.
        
        Args:
            image: PIL Image da página
            
        Returns:
            str: Tipo de cena ('present', 'flashback', 'dream', 'nightmare')
        """
        # Converte para numpy array
        img_array = np.array(image.convert('RGB'))
        
        # Análise de histograma
        gray = np.mean(img_array, axis=2)
        
        # Métricas básicas
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Heurísticas simples
        # Flashback: alta luminosidade, baixo contraste (branco e preto)
        # Nightmare: baixa luminosidade, alto contraste
        # Dream: média luminosidade, baixo contraste
        # Present: distribuição normal
        
        if mean_brightness > 200 and std_brightness < 40:
            return SceneType.FLASHBACK
        elif mean_brightness < 60 and std_brightness > 60:
            return SceneType.NIGHTMARE
        elif 100 < mean_brightness < 180 and std_brightness < 50:
            return SceneType.DREAM
        else:
            return SceneType.PRESENT
    
    def detect_batch(self, images: list) -> list:
        """
        Detecta tipos de cena para múltiplas imagens.
        
        Args:
            images: Lista de PIL Images
            
        Returns:
            list: Lista de tipos de cena
        """
        return [self.detect(img) for img in images]

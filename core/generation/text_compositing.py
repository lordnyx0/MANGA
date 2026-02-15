"""
MangaAutoColor Pro - Text Compositing Inteligente

Evita o "Efeito Carta de Resgate" através de:
1. Filtros de validação de texto real
2. Suavização de bordas (feather)
3. Detecção de falsos positivos
"""

import numpy as np
from PIL import Image, ImageFilter
from typing import List, Dict, Tuple, Optional
import cv2


class SmartTextCompositing:
    """
    Compositing inteligente de texto para mangá.
    
    Diferente do compositing ingênuo, este sistema:
    - Valida se a região realmente contém texto
    - Rejeita SFX, narrações em caixas pretas, riscos
    - Aplica feather nas bordas para transição suave
    """
    
    def __init__(
        self,
        min_area: int = 100,      # Área mínima (pixels)
        max_area: int = 50000,    # Área máxima (evita caixas enormes)
        min_aspect: float = 0.3,  # Proporção mínima altura/largura
        max_aspect: float = 3.0,  # Proporção máxima
        feather_radius: int = 3,  # Suavização das bordas
        confidence_threshold: float = 0.5  # Confiança mínima do detector
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.feather_radius = feather_radius
        self.confidence_threshold = confidence_threshold
    
    def apply(
        self,
        generated_image: Image.Image,
        original_image: Image.Image,
        detections: List[Dict]
    ) -> Tuple[Image.Image, int]:
        """
        Aplica compositing inteligente.
        
        Returns:
            Tuple de (imagem_resultado, número_de_regiões_processadas)
        """
        result = generated_image.copy()
        text_count = 0
        
        for det in detections:
            # Verifica se é texto
            if not self._is_text_detection(det):
                continue
            
            # Valida a região
            bbox = det.get('bbox')
            if not self._validate_bbox(bbox, original_image.size):
                continue
            
            x1, y1, x2, y2 = [int(c) for c in bbox]
            
            # Recorta região da original
            try:
                text_region = original_image.crop((x1, y1, x2, y2))
            except Exception:
                continue
            
            # Valida conteúdo (evita caixas pretas/brancas sólidas)
            if not self._validate_content(text_region):
                print(f"[SmartTextCompositing] Rejeitado: região sem texto legível")
                continue
            
            # Aplica feather para suavizar bordas
            if self.feather_radius > 0:
                text_region = self._apply_feather(text_region)
            
            # Cola na imagem gerada
            try:
                result.paste(text_region, (x1, y1))
                text_count += 1
                print(f"[SmartTextCompositing] Texto restaurado em ({x1}, {y1}, {x2}, {y2})")
            except Exception as e:
                print(f"[SmartTextCompositing] Erro ao colar: {e}")
        
        return result, text_count
    
    def _is_text_detection(self, det: Dict) -> bool:
        """Verifica se a detecção é do tipo texto."""
        class_id = det.get('class_id', -1)
        class_name = det.get('class_name', '')
        confidence = det.get('confidence', 0.0)
        
        # Deve ser classe texto com confiança suficiente
        is_text = (class_id == 3 or class_name in ['text', 'texto'])
        confident = confidence >= self.confidence_threshold
        
        return is_text and confident
    
    def _validate_bbox(self, bbox: Optional[Tuple], image_size: Tuple[int, int]) -> bool:
        """Valida se o bbox tem proporções de texto."""
        if bbox is None:
            return False
        
        try:
            x1, y1, x2, y2 = [int(c) for c in bbox]
        except (ValueError, TypeError):
            return False
        
        img_w, img_h = image_size
        
        # Coordenadas válidas
        if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
            return False
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        # Área
        area = (x2 - x1) * (y2 - y1)
        if area < self.min_area or area > self.max_area:
            print(f"[SmartTextCompositing] Rejeitado: área {area} fora do intervalo")
            return False
        
        # Proporção (aspect ratio)
        width = x2 - x1
        height = y2 - y1
        aspect = height / width if width > 0 else 1.0
        
        if aspect < self.min_aspect or aspect > self.max_aspect:
            print(f"[SmartTextCompositing] Rejeitado: aspecto {aspect:.2f} não parece texto")
            return False
        
        return True
    
    def _validate_content(self, region: Image.Image) -> bool:
        """
        Valida se a região realmente contém texto.
        
        Rejeita:
        - Caixas pretas/brancas sólidas (narrações)
        - Regiões uniformes (sem variação)
        - Riscos verticais/horizontais simples
        """
        # Converte para numpy
        img_array = np.array(region.convert('L'))  # Grayscale
        
        # Verifica variação de intensidade
        std_dev = np.std(img_array)
        if std_dev < 10:  # Muito uniforme (quase sólido)
            return False
        
        # Verifica se tem contraste de texto (preto no branco ou vice-versa)
        mean_intensity = np.mean(img_array)
        
        # Evita caixas muito escuras (narrações em preto) ou muito claras
        if mean_intensity < 30 or mean_intensity > 225:
            return False
        
        # Análise de bordas (texto tem muitas bordas verticais)
        sobelx = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_array, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Texto tem bordas significativas
        edge_density = np.sum(edge_magnitude > 50) / edge_magnitude.size
        if edge_density < 0.05:  # Muito poucas bordas
            print(f"[SmartTextCompositing] Rejeitado: densidade de bordas {edge_density:.3f} muito baixa")
            return False
        
        # Verifica se tem estrutura de linhas (características de texto)
        # Conta componentes conectados
        _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Texto típico tem vários componentes pequenos (letras)
        small_components = sum(1 for i in range(1, num_labels) 
                              if 10 < stats[i, cv2.CC_STAT_AREA] < 1000)
        
        if small_components < 3:  # Poucos componentes = provavelmente não é texto
            print(f"[SmartTextCompositing] Rejeitado: apenas {small_components} componentes pequenos")
            return False
        
        return True
    
    def _apply_feather(self, region: Image.Image) -> Image.Image:
        """Aplica feather nas bordas para transição suave."""
        if self.feather_radius <= 0:
            return region
        
        # Cria máscara com alpha
        rgba = region.convert('RGBA')
        
        # Aplica blur na máscara para suavizar bordas
        feathered = rgba.filter(ImageFilter.GaussianBlur(self.feather_radius))
        
        return feathered


# Função de conveniência para uso no pipeline
def apply_smart_text_compositing(
    generated_image: Image.Image,
    original_image: Image.Image,
    detections: List[Dict],
    enable: bool = True
) -> Image.Image:
    """
    Aplica text compositing inteligente.
    
    Args:
        generated_image: Imagem colorizada pela IA
        original_image: Imagem original em P&B
        detections: Lista de detecções
        enable: Se False, retorna a imagem original sem compositing
        
    Returns:
        Imagem processada
    """
    if not enable:
        return generated_image
    
    compositor = SmartTextCompositing()
    result, count = compositor.apply(generated_image, original_image, detections)
    
    if count > 0:
        print(f"[SmartTextCompositing] Total: {count} regiões de texto restauradas")
    
    return result

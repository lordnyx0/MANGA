"""
MangaAutoColor Pro - Image Operations Utilities
Funções comuns de processamento de imagem e geometria para evitar duplicação.
"""

from typing import Tuple, List, Union
import numpy as np
import cv2
from PIL import Image

def clamp_bbox(
    bbox: Tuple[int, int, int, int], 
    image_shape: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Garante que bbox esteja dentro dos limites da imagem.
    
    Args:
        bbox: (x1, y1, x2, y2)
        image_shape: (width, height) ou (height, width, channels)
        
    Returns:
        bbox ajustado (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox
    
    if len(image_shape) == 2:
        max_w, max_h = image_shape
    elif len(image_shape) == 3:
        max_h, max_w = image_shape[:2]
    else:
        raise ValueError("image_shape deve ser (w, h) ou shape numpy")
        
    x1 = max(0, min(x1, max_w))
    y1 = max(0, min(y1, max_h))
    x2 = max(0, min(x2, max_w))
    y2 = max(0, min(y2, max_h))
    
    return (x1, y1, x2, y2)


def calculate_context_bbox(
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    inflation_factor: float = 1.5
) -> Tuple[int, int, int, int]:
    """
    Infla um bounding box por um fator, mantendo o centro.
    Útil para capturar contexto ao redor de um personagem (para IP-Adapter).
    
    Args:
        bbox: (x1, y1, x2, y2)
        image_shape: (width, height)
        inflation_factor: Fator de multiplicação (ex: 1.5 = 150%)
        
    Returns:
        bbox inflado e clampar (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox
    
    # Centro e dimensões atuais
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    w = x2 - x1
    h = y2 - y1
    
    # Novas dimensões
    new_w = int(w * inflation_factor)
    new_h = int(h * inflation_factor)
    
    # Novas coordenadas
    new_x1 = cx - new_w // 2
    new_y1 = cy - new_h // 2
    new_x2 = cx + new_w // 2
    new_y2 = cy + new_h // 2
    
    return clamp_bbox((new_x1, new_y1, new_x2, new_y2), image_shape)


def create_context_crop(
    image: Union[np.ndarray, Image.Image],
    bbox: Tuple[int, int, int, int],
    inflation_factor: float = 1.5
) -> Union[np.ndarray, Image.Image]:
    """
    Retorna o crop da imagem correspondente ao bbox com contexto.
    Suporta tanto numpy array quanto PIL Image.
    """
    is_pil = isinstance(image, Image.Image)
    
    if is_pil:
        w, h = image.size
        # PIL usa (width, height)
        shape = (w, h)
    else:
        h, w = image.shape[:2]
        # Numpy usa (height, width) mas calculate_context_bbox espera (w, h)
        shape = (w, h)
        
    ctx_bbox = calculate_context_bbox(bbox, shape, inflation_factor)
    x1, y1, x2, y2 = ctx_bbox
    
    if is_pil:
        return image.crop((x1, y1, x2, y2))
    else:
        return image[y1:y2, x1:x2]


def extract_canny_edges(
    image: np.ndarray, 
    low_threshold: int = 50, 
    high_threshold: int = 150
) -> np.ndarray:
    """
    Extrai arestas usando Canny.
    Se a imagem for RGB, converte para Grayscale primeiro.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
        
    return cv2.Canny(gray, low_threshold, high_threshold)


def create_gaussian_mask(
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    sigma_factor: float = 4.0
) -> np.ndarray:
    """
    Cria máscara gaussiana centrada no bbox.
    
    Args:
        bbox: (x1, y1, x2, y2)
        image_shape: (height, width) - Nota: Ordem numpy
        
    Returns:
        Máscara 2D float32 (0-1)
    """
    img_h, img_w = image_shape
    
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    
    sigma_x = (x2 - x1) / sigma_factor
    sigma_y = (y2 - y1) / sigma_factor
    sigma = max(sigma_x, sigma_y, 20)  # Min sigma para evitar singularidade
    
    y, x = np.ogrid[:img_h, :img_w]
    dist_sq = ((x - cx) / sigma) ** 2 + ((y - cy) / sigma) ** 2
    mask = np.exp(-dist_sq / 2)
    
    return mask.astype(np.float32)


def create_blend_mask(
    tile_size: Tuple[int, int],
    overlap: int,
    image_size: Optional[Tuple[int, int]] = None,
    tile_bbox: Optional[Tuple[int, int, int, int]] = None
) -> np.ndarray:
    """
    Cria máscara para blending de tiles (feathered/gaussiana).
    Versão aprimorada com scipy para transições mais suaves.
    """
    from scipy.ndimage import gaussian_filter
    
    # Se receber tile_bbox e image_size, calcula se precisa de feather em cada lado
    # (Não aplica feather na borda externa da página)
    h, w = tile_size
    mask = np.ones((h, w), dtype=np.float32)
    
    if overlap <= 0:
        return mask
        
    x = np.linspace(0, 1, overlap)
    left = x
    right = x[::-1]
    
    # Se temos contexto de página, sabemos onde não aplicar feather
    if image_size and tile_bbox:
        img_w, img_h = image_size
        tx1, ty1, tx2, ty2 = tile_bbox
        
        if tx1 > 0: mask[:, :overlap] *= left[np.newaxis, :]
        if tx2 < img_w: mask[:, -overlap:] *= right[np.newaxis, :]
        if ty1 > 0: mask[:overlap, :] *= left[:, np.newaxis]
        if ty2 < img_h: mask[-overlap:, :] *= right[:, np.newaxis]
    else:
        # Modo simples: feather em todos os lados
        mask[:, :overlap] *= left[np.newaxis, :]
        mask[:, -overlap:] *= right[np.newaxis, :]
        mask[:overlap, :] *= left[:, np.newaxis]
        mask[-overlap:, :] *= right[:, np.newaxis]
    
    # Suaviza a transição
    mask = gaussian_filter(mask, sigma=overlap/4)
    return np.clip(mask, 0.0, 1.0)

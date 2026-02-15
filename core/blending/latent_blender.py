"""
MangaAutoColor Pro - Blending no Espaço Latente

Implementa blending suave de imagens geradas usando operações no espaço latente
do VAE. Baseado no conceito de Blended Latent Diffusion, mas otimizado para
composição de tiles e personagens.

Técnicas:
- Gaussian Mask Blending: Máscaras suaves para transições
- Multi-band Blending: Preservação de detalhes em diferentes frequências
- Chroma Isolation: Separação de luminância/crominância para blending
- Poisson Blending: Harmonização de bordas (opcional)

Referências:
- Blended Latent Diffusion: arxiv.org/abs/2206.02779
- Multi-band Blending: Burt and Adelson, 1983
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Union, Dict
import cv2
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass

try:
    from skimage.color import rgb2lab, lab2rgb
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

from config.settings import (
    FEATHER_WIDTH, GAUSSIAN_SIGMA, GAUSSIAN_KERNEL_SIZE,
    POISSON_BLEND_ENABLED, POISSON_BLEND_ITERATIONS,
    VERBOSE
)


@dataclass
class BlendRegion:
    """
    Região para blending.
    
    Attributes:
        image: Imagem RGB (H, W, 3)
        mask: Máscara de peso (H, W) em [0, 1]
        bbox: Posição na imagem final (x1, y1, x2, y2)
        priority: Prioridade para sobreposição
    """
    image: np.ndarray
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    priority: int = 0


class LatentBlender:
    """
    Blender de imagens no espaço latente e pixel.
    
    Fornece múltiplas estratégias de blending para diferentes cenários:
    - Tile blending: União de tiles com overlap
    - Character blending: Composição de personagens sobre background
    - Poisson blending: Harmonização final de bordas
    
    Args:
        feather_width: Largura da borda suave
        use_multiband: Usar multi-band blending
    """
    
    def __init__(
        self,
        feather_width: int = FEATHER_WIDTH,
        use_multiband: bool = True
    ):
        self.feather_width = feather_width
        self.use_multiband = use_multiband
        
        if VERBOSE:
            print(f"[LatentBlender] Inicializado (feather={feather_width}, "
                  f"multiband={use_multiband})")
    
    def blend_tiles(
        self,
        image_shape: Tuple[int, int],
        tiles: List[Tuple[np.ndarray, Tuple[int, int, int, int]]]
    ) -> np.ndarray:
        """
        Blending de tiles em uma imagem completa.
        
        Args:
            image_shape: (h, w) da imagem final
            tiles: Lista de (imagem, bbox)
            
        Returns:
            Imagem blended
        """
        img_h, img_w = image_shape
        
        # Acumuladores
        accumulator = np.zeros((img_h, img_w, 3), dtype=np.float32)
        weight_map = np.zeros((img_h, img_w), dtype=np.float32)
        
        for tile_img, (x1, y1, x2, y2) in tiles:
            h, w = tile_img.shape[:2]
            
            # Garante dimensões
            x2 = min(x1 + w, img_w)
            y2 = min(y1 + h, img_h)
            w = x2 - x1
            h = y2 - y1
            
            tile_crop = tile_img[:h, :w]
            
            # Cria máscara com feather nas bordas
            mask = self._create_feathered_mask((h, w), (x1, y1, x2, y2), (img_w, img_h))
            
            # Acumula
            for c in range(3):
                accumulator[y1:y2, x1:x2, c] += tile_crop[:, :, c] * mask
            weight_map[y1:y2, x1:x2] += mask
        
        # Normaliza
        weight_map = np.maximum(weight_map, 1e-8)
        result = accumulator / weight_map[:, :, np.newaxis]
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _create_feathered_mask(
        self,
        tile_shape: Tuple[int, int],
        tile_bbox: Tuple[int, int, int, int],
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Cria máscara com feathering nas bordas do tile.
        
        Args:
            tile_shape: (h, w) do tile
            tile_bbox: Posição (x1, y1, x2, y2)
            image_size: (w, h) da imagem completa
            
        Returns:
            Máscara com feather
        """
        h, w = tile_shape
        mask = np.ones((h, w), dtype=np.float32)
        
        x1, y1, x2, y2 = tile_bbox
        img_w, img_h = image_size
        
        feather = self.feather_width
        
        # Borda esquerda
        if x1 > 0:
            for i in range(min(feather, w)):
                mask[:, i] *= (i / feather) if feather > 0 else 1.0
        
        # Borda direita
        if x2 < img_w:
            for i in range(min(feather, w)):
                mask[:, w-1-i] *= (i / feather) if feather > 0 else 1.0
        
        # Borda superior
        if y1 > 0:
            for i in range(min(feather, h)):
                mask[i, :] *= (i / feather) if feather > 0 else 1.0
        
        # Borda inferior
        if y2 < img_h:
            for i in range(min(feather, h)):
                mask[h-1-i, :] *= (i / feather) if feather > 0 else 1.0
        
        return mask
    
    def blend_with_mask(
        self,
        background: np.ndarray,
        foreground: np.ndarray,
        mask: np.ndarray,
        feather_edges: bool = True
    ) -> np.ndarray:
        """
        Blending de foreground sobre background usando máscara.
        
        Args:
            background: Imagem de fundo
            foreground: Imagem da frente
            mask: Máscara de blending [0, 1]
            feather_edges: Aplicar feather nas bordas da máscara
            
        Returns:
            Imagem blended
        """
        # Garante mesmas dimensões
        h, w = background.shape[:2]
        foreground = cv2.resize(foreground, (w, h))
        mask = cv2.resize(mask, (w, h))
        
        # Aplica feather na máscara se solicitado
        if feather_edges:
            mask = self._apply_gaussian_blur(mask, GAUSSIAN_SIGMA)
        
        # Normaliza máscara
        mask = np.clip(mask, 0, 1)
        
        # Blending alpha
        mask_3ch = mask[:, :, np.newaxis]
        result = background * (1 - mask_3ch) + foreground * mask_3ch
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _apply_gaussian_blur(
        self,
        mask: np.ndarray,
        sigma: float
    ) -> np.ndarray:
        """Aplica blur gaussiano em máscara"""
        return gaussian_filter(mask, sigma=sigma)
    
    def create_gaussian_mask(
        self,
        shape: Tuple[int, int],
        center: Optional[Tuple[int, int]] = None,
        sigma: Optional[float] = None
    ) -> np.ndarray:
        """
        Cria máscara gaussiana 2D.
        
        Args:
            shape: (h, w) da máscara
            center: Centro da gaussiana (padrão: centro da imagem)
            sigma: Desvio padrão (padrão: baseado no tamanho)
            
        Returns:
            Máscara gaussiana normalizada [0, 1]
        """
        h, w = shape
        
        if center is None:
            center = (w // 2, h // 2)
        
        if sigma is None:
            sigma = min(w, h) / 4
        
        # Coordenadas
        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)
        
        # Gaussiana 2D
        cx, cy = center
        mask = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        
        # Normaliza
        mask = mask / mask.max()
        
        return mask.astype(np.float32)
    
    def multiband_blend(
        self,
        regions: List[BlendRegion],
        output_shape: Tuple[int, int],
        num_bands: int = 6
    ) -> np.ndarray:
        """
        Multi-band blending para preservação de detalhes.
        
        Decompõe cada região em laplacian pyramids e combina
        por banda de frequência para transições suaves.
        
        Args:
            regions: Lista de regiões para blending
            output_shape: (h, w) da saída
            num_bands: Número de bandas de frequência
            
        Returns:
            Imagem blended
        """
        img_h, img_w = output_shape
        
        # Acumuladores por banda
        band_accumulators = [np.zeros((img_h, img_w, 3)) for _ in range(num_bands)]
        band_weights = [np.zeros((img_h, img_w)) for _ in range(num_bands)]
        
        for region in regions:
            x1, y1, x2, y2 = region.bbox
            h = y2 - y1
            w = x2 - x1
            
            # Redimensiona imagem e máscara para o bbox
            img_resized = cv2.resize(region.image, (w, h))
            mask_resized = cv2.resize(region.mask, (w, h))
            
            # Laplacian pyramid
            pyramid = self._laplacian_pyramid(img_resized, num_bands)
            mask_pyramid = self._gaussian_pyramid(mask_resized, num_bands)
            
            # Acumula por banda
            for i, (band, mask_band) in enumerate(zip(pyramid, mask_pyramid)):
                # Redimensiona banda se necessário
                if band.shape[0] != h or band.shape[1] != w:
                    band = cv2.resize(band, (w, h))
                if mask_band.shape[0] != h or mask_band.shape[1] != w:
                    mask_band = cv2.resize(mask_band, (w, h))
                
                # Expand mask para 3 canais
                mask_3ch = np.stack([mask_band] * 3, axis=-1)
                
                # Acumula
                band_accumulators[i][y1:y2, x1:x2] += band * mask_3ch
                band_weights[i][y1:y2, x1:x2] += mask_band
        
        # Reconstrói imagem
        result_bands = []
        for acc, weight in zip(band_accumulators, band_weights):
            weight = np.maximum(weight, 1e-8)
            weight_3ch = np.stack([weight] * 3, axis=-1)
            result_bands.append(acc / weight_3ch)
        
        # Reconstrói da pirâmide laplaciana
        result = self._reconstruct_from_laplacian(result_bands)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _gaussian_pyramid(
        self,
        image: np.ndarray,
        levels: int
    ) -> List[np.ndarray]:
        """Constrói pirâmide gaussiana"""
        pyramid = [image]
        current = image.copy()
        
        for _ in range(levels - 1):
            current = cv2.pyrDown(current)
            pyramid.append(current)
        
        return pyramid
    
    def _laplacian_pyramid(
        self,
        image: np.ndarray,
        levels: int
    ) -> List[np.ndarray]:
        """Constrói pirâmide laplaciana"""
        gaussian = self._gaussian_pyramid(image, levels)
        laplacian = []
        
        for i in range(levels - 1):
            size = (gaussian[i].shape[1], gaussian[i].shape[0])
            ge = cv2.pyrUp(gaussian[i + 1], dstsize=size)
            
            # Garante mesmas dimensões
            if ge.shape != gaussian[i].shape:
                ge = cv2.resize(ge, (gaussian[i].shape[1], gaussian[i].shape[0]))
            
            laplacian.append(gaussian[i] - ge)
        
        laplacian.append(gaussian[-1])
        
        return laplacian
    
    def _reconstruct_from_laplacian(
        self,
        pyramid: List[np.ndarray]
    ) -> np.ndarray:
        """Reconstrói imagem da pirâmide laplaciana"""
        result = pyramid[-1]
        
        for i in range(len(pyramid) - 2, -1, -1):
            size = (pyramid[i].shape[1], pyramid[i].shape[0])
            result = cv2.pyrUp(result, dstsize=size)
            
            # Garante mesmas dimensões
            if result.shape != pyramid[i].shape:
                result = cv2.resize(result, (pyramid[i].shape[1], pyramid[i].shape[0]))
            
            result = result + pyramid[i]
        
        return result
    
    def chroma_blend(
        self,
        background: np.ndarray,
        foreground: np.ndarray,
        mask: np.ndarray,
        preserve_luma: bool = True
    ) -> np.ndarray:
        """
        Blending preservando luminância do background.
        
        Útil para manter consistência de iluminação enquanto
        aplica cores do foreground.
        
        Args:
            background: Imagem de fundo
            foreground: Imagem da frente
            mask: Máscara de blending
            preserve_luma: Preservar luminância
            
        Returns:
            Imagem blended
        """
        if not SKIMAGE_AVAILABLE or not preserve_luma:
            return self.blend_with_mask(background, foreground, mask)
        
        # Converte para Lab
        bg_lab = rgb2lab(background / 255.0)
        fg_lab = rgb2lab(foreground / 255.0)
        
        # Blending apenas nos canais de cor (a, b)
        mask_blur = self._apply_gaussian_blur(mask, GAUSSIAN_SIGMA)
        mask_3ch = np.stack([mask_blur] * 3, axis=-1)
        
        result_lab = bg_lab.copy()
        
        # Mantém luminância do background, mistura crominância
        result_lab[:, :, 1:] = (
            bg_lab[:, :, 1:] * (1 - mask_3ch[:, :, 1:]) +
            fg_lab[:, :, 1:] * mask_3ch[:, :, 1:]
        )
        
        # Converte de volta
        result = lab2rgb(result_lab) * 255
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def poisson_blend(
        self,
        background: np.ndarray,
        foreground: np.ndarray,
        mask: np.ndarray,
        iterations: int = POISSON_BLEND_ITERATIONS
    ) -> np.ndarray:
        """
        Poisson blending para harmonização de bordas.
        
        Preserva gradientes do foreground enquanto mantém
        consistência com o background nas bordas.
        
        Args:
            background: Imagem de fundo
            foreground: Imagem da frente
            mask: Máscara binária da região
            iterations: Iterações do solver
            
        Returns:
            Imagem harmonizada
        """
        if not POISSON_BLEND_ENABLED:
            return self.blend_with_mask(background, foreground, mask)
        
        # Implementação simplificada de Poisson blending
        # Versão completa usaria o método de Jacobi ou FFT
        
        result = background.copy().astype(np.float32)
        fg = foreground.astype(np.float32)
        
        # Máscara binária
        mask_binary = (mask > 0.5).astype(np.float32)
        
        # Região de interesse
        roi = mask_binary[:, :, np.newaxis]
        
        # Iterações de suavização
        for _ in range(iterations // 10):  # Reduzido para performance
            # Laplaciano
            laplacian = cv2.Laplacian(result, cv2.CV_32F)
            
            # Atualiza apenas na região da máscara
            result = result * (1 - roi) + (result + 0.1 * laplacian) * roi
            
            # Mantém valores do foreground como guia
            result = result * (1 - roi * 0.1) + fg * (roi * 0.1)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def create_differential_mask(
        self,
        base_mask: np.ndarray,
        strength_center: float = 1.0,
        strength_edge: float = 0.0,
        decay_type: str = "gaussian"
    ) -> np.ndarray:
        """
        Cria máscara diferencial com força variável.
        
        Implementação do conceito de Differential Diffusion:
        - Centro: força máxima (1.0)
        - Bordas: decaimento suave para força mínima (0.0)
        
        Args:
            base_mask: Máscara binária base
            strength_center: Força no centro
            strength_edge: Força nas bordas
            decay_type: Tipo de decaimento ("gaussian", "linear")
            
        Returns:
            Máscara diferencial
        """
        from scipy.ndimage import distance_transform_edt
        
        # Distância da borda
        distance = distance_transform_edt(base_mask)
        
        # Normaliza distância
        max_dist = distance.max()
        if max_dist > 0:
            distance = distance / max_dist
        
        # Aplica decaimento
        if decay_type == "gaussian":
            # Decaimento gaussiano
            mask = strength_edge + (strength_center - strength_edge) * \
                   np.exp(-((1 - distance) ** 2) / (2 * 0.3 ** 2))
        else:
            # Decaimento linear
            mask = strength_edge + (strength_center - strength_edge) * distance
        
        return np.clip(mask, 0, 1).astype(np.float32)


def blend_images_average(
    images: List[np.ndarray],
    weights: Optional[List[float]] = None
) -> np.ndarray:
    """
    Blending simples por média ponderada.
    
    Args:
        images: Lista de imagens
        weights: Pesos para cada imagem
        
    Returns:
        Imagem média
    """
    if weights is None:
        weights = [1.0 / len(images)] * len(images)
    
    result = np.zeros_like(images[0], dtype=np.float32)
    
    for img, weight in zip(images, weights):
        result += img.astype(np.float32) * weight
    
    return np.clip(result, 0, 255).astype(np.uint8)


def create_transition_mask(
    shape: Tuple[int, int],
    direction: str = "horizontal",
    position: float = 0.5,
    width: float = 0.1
) -> np.ndarray:
    """
    Cria máscara para transição suave entre duas imagens.
    
    Args:
        shape: (h, w) da máscara
        direction: "horizontal" ou "vertical"
        position: Posição da transição (0-1)
        width: Largura da zona de transição (0-1)
        
    Returns:
        Máscara de transição [0, 1]
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.float32)
    
    if direction == "horizontal":
        center = int(w * position)
        trans_width = int(w * width)
        
        x = np.arange(w)
        mask = np.clip((x - (center - trans_width)) / (2 * trans_width), 0, 1)
        mask = np.tile(mask, (h, 1))
    else:
        center = int(h * position)
        trans_width = int(h * width)
        
        y = np.arange(h)
        mask = np.clip((y - (center - trans_width)) / (2 * trans_width), 0, 1)
        mask = np.tile(mask[:, np.newaxis], (1, w))
    
    return mask

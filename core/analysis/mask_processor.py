"""
MangaAutoColor Pro - Mask Processor (ADR 004)

Processamento de máscaras para segmentação SAM:
- Operações morfológicas (close, erode, dilate)
- Resolução de oclusões (Z-Ordering)
- Suavização de bordas para IP-Adapter

Pipeline de Processamento (ordem correta):
1. Subtração Booleana (Hard Cut) - resolve sobreposições
2. Dilatação mínima (1px) na máscara de fundo - garante overlap
3. Gaussian Blur suave (sigma=0.5) - anti-hard-edge

Evita o problema de "aura" entre personagens ao garantir que
a dilatação ocorra antes do blur.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from core.logging.setup import get_logger
from config.settings import MASK_MIN_AREA_RATIO, MASK_MAX_COMPONENTS

logger = get_logger("MaskProcessor")


class MaskOperations(Enum):
    """Operações morfológicas suportadas."""
    CLOSE = "close"       # Fechamento (dilate + erode) - remove buracos
    ERODE = "erode"       # Erosão - encolhe máscara
    DILATE = "dilate"     # Dilatação - expande máscara
    OPEN = "open"         # Abertura (erode + dilate) - remove ruído


@dataclass
class ProcessedMask:
    """Máscara processada com metadados."""
    char_id: str
    mask: np.ndarray  # Máscara binária (0 ou 255)
    mask_float: np.ndarray  # Máscara float (0.0 a 1.0) para IP-Adapter
    depth_rank: int  # Ordem de profundidade
    was_occluded: bool  # Se teve área subtraída por oclusão
    overlap_pixels: int  # Pixels de overlap com outros personagens


class MaskProcessor:
    """
    Processador de máscaras SAM para uso no Regional IP-Adapter.
    
    Responsabilidades:
    1. Aplicar operações morfológicas (close, erode)
    2. Resolver oclusões via Z-Ordering (subtração booleana)
    3. Suavizar bordas para evitar hard edges no IP-Adapter
    
    Args:
        close_kernel_size: Tamanho do kernel para morphological close
        erosion_pixels: Pixels para erosão em contatos diretos
        blur_sigma: Sigma do gaussian blur para bordas
    """
    
    def __init__(
        self,
        close_kernel_size: int = 3,
        erosion_pixels: int = 2,
        blur_sigma: float = 0.5,
        overlap_dilation: int = 1  # Dilatação para garantir overlap mínimo
    ):
        self.close_kernel_size = close_kernel_size
        self.erosion_pixels = erosion_pixels
        self.blur_sigma = blur_sigma
        self.overlap_dilation = overlap_dilation
        
        logger.debug(f"MaskProcessor inicializado "
                    f"(close={close_kernel_size}, erode={erosion_pixels}, "
                    f"blur={blur_sigma})")
    

    def _is_mask_quality_valid(self, mask: np.ndarray) -> bool:
        h, w = mask.shape[:2]
        area_ratio = float(np.mean(mask > 0))
        if area_ratio < MASK_MIN_AREA_RATIO:
            return False

        num_labels, _ = cv2.connectedComponents((mask > 0).astype(np.uint8))
        # connectedComponents conta background como label 0
        components = max(0, num_labels - 1)
        if components > MASK_MAX_COMPONENTS:
            return False

        return True

    def process_masks(
        self,
        segmentation_results: Dict[str, 'SegmentationResult'],
        depth_order: List[str]
    ) -> Dict[str, ProcessedMask]:
        """
        Processa máscaras completas: morfologia + oclusão + suavização.
        
        Args:
            segmentation_results: Dict de char_id -> SegmentationResult (com RLE)
            depth_order: Lista de char_ids ordenada da FRENTE para o FUNDO
            
        Returns:
            Dict de char_id -> ProcessedMask prontas para IP-Adapter
        """
        if not segmentation_results:
            return {}
        
        # 1. Decodifica máscaras RLE
        raw_masks = {}
        for char_id, seg_result in segmentation_results.items():
            raw_masks[char_id] = seg_result.mask
        
        # 2. Aplica morphological close para limpar bordas + gate de qualidade
        closed_masks = {}
        for char_id, mask in raw_masks.items():
            candidate = self.apply_morphological_close(mask, self.close_kernel_size)
            if not self._is_mask_quality_valid(candidate):
                logger.warning(f"Máscara rejeitada por qualidade: {char_id}")
                continue
            closed_masks[char_id] = candidate
        
        # 3. Resolve oclusões via Z-Ordering
        occlusion_masks = self.compute_occlusion_masks(closed_masks, depth_order)
        
        # 4. Aplica erosão em regiões de contato (anti-aliasing)
        eroded_masks = {}
        for char_id in depth_order:
            if char_id not in occlusion_masks:
                continue
            mask = occlusion_masks[char_id]
            
            # Verifica se houve oclusão (perdeu área)
            original_area = np.sum(closed_masks[char_id] > 0)
            current_area = np.sum(mask > 0)
            was_occluded = current_area < original_area
            overlap_pixels = original_area - current_area
            
            # Aplica erosão apenas se houver contato com outros
            if was_occluded and self.erosion_pixels > 0:
                mask = self.apply_erosion(mask, self.erosion_pixels)
            
            eroded_masks[char_id] = (mask, was_occluded, overlap_pixels)
        
        # 5. Aplica dilatação mínima em máscaras de fundo para garantir overlap
        dilated_masks = self._apply_overlap_dilation(eroded_masks, depth_order)
        
        # 6. Suaviza bordas com Gaussian Blur
        processed = {}
        for char_id in depth_order:
            if char_id not in dilated_masks:
                continue
            mask, was_occluded, overlap_pixels = dilated_masks[char_id]
            
            # Converte para float e aplica blur
            mask_float = mask.astype(np.float32) / 255.0
            
            if self.blur_sigma > 0:
                mask_float = self.apply_gaussian_blur(mask_float, self.blur_sigma)
            
            processed[char_id] = ProcessedMask(
                char_id=char_id,
                mask=mask,
                mask_float=mask_float,
                depth_rank=depth_order.index(char_id) + 1,
                was_occluded=was_occluded,
                overlap_pixels=overlap_pixels
            )
        
        logger.info(f"Processadas {len(processed)} máscaras")
        return processed
    
    def compute_occlusion_masks(
        self,
        masks: Dict[str, np.ndarray],
        depth_order: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Resolve sobreposições usando Z-Ordering.
        
        Fórmula (ADR 004):
        M_i_final = M_i ∩ (¬⋃_{j∈Front(i)} M_j)
        
        Personagens mais à frente (menor depth_rank) subtraem
        sua área dos personagens ao fundo.
        
        Args:
            masks: Dict de char_id -> máscara binária
            depth_order: Ordem da frente para o fundo
            
        Returns:
            Máscaras com oclusões resolvidas
        """
        result = {}
        front_union = None  # União booleana das máscaras à frente

        for char_id in depth_order:
            if char_id not in masks:
                continue

            mask_bool = masks[char_id] > 0

            # Subtração booleana robusta: remove área ocupada por personagens à frente
            if front_union is not None:
                mask_bool = np.logical_and(mask_bool, np.logical_not(front_union))

            result[char_id] = (mask_bool.astype(np.uint8) * 255)

            # Atualiza união de primeiro plano com máscara ORIGINAL (não subtraída)
            front_mask_bool = masks[char_id] > 0
            front_union = front_mask_bool if front_union is None else np.logical_or(front_union, front_mask_bool)

        return result
    
    def _apply_overlap_dilation(
        self,
        masks: Dict[str, Tuple[np.ndarray, bool, int]],
        depth_order: List[str]
    ) -> Dict[str, Tuple[np.ndarray, bool, int]]:
        """
        Aplica dilatação mínima em máscaras de fundo para garantir overlap.
        
        Isso evita gaps entre personagens após a subtração de oclusão.
        Aplica apenas em personagens que NÃO são o primeiro plano.
        """
        if self.overlap_dilation <= 0:
            return masks
        
        result = {}
        
        front_union = None

        for idx, char_id in enumerate(depth_order):
            if char_id not in masks:
                continue

            mask, was_occluded, overlap_pixels = masks[char_id]
            mask_bool = mask > 0

            # Aplica dilatação em personagens de fundo (não o primeiro),
            # mas evita invadir área já ocupada pelo foreground.
            if idx > 0:  # Não é o personagem mais à frente
                dilated = self.apply_dilation((mask_bool.astype(np.uint8) * 255), self.overlap_dilation) > 0
                if front_union is not None:
                    dilated = np.logical_and(dilated, np.logical_not(front_union))
                mask_bool = np.logical_or(mask_bool, dilated)

            result[char_id] = (mask_bool.astype(np.uint8) * 255, was_occluded, overlap_pixels)

            original_bool = masks[char_id][0] > 0
            front_union = original_bool if front_union is None else np.logical_or(front_union, original_bool)

        return result
    
    # =========================================================================
    # Operações Morfológicas Individuais
    # =========================================================================
    
    def apply_morphological_close(
        self,
        mask: np.ndarray,
        kernel_size: int = 3
    ) -> np.ndarray:
        """
        Aplica fechamento morfológico (dilate + erode).
        
        Remove pequenos buracos na máscara e suaviza bordas.
        Útil para liminar máscaras SAM.
        """
        if kernel_size <= 0:
            return mask
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (kernel_size, kernel_size)
        )
        
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    def apply_morphological_open(
        self,
        mask: np.ndarray,
        kernel_size: int = 3
    ) -> np.ndarray:
        """
        Aplica abertura morfológica (erode + dilate).
        
        Remove pequenos ruídos e artefatos isolados.
        """
        if kernel_size <= 0:
            return mask
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )
        
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    def apply_erosion(
        self,
        mask: np.ndarray,
        pixels: int = 2
    ) -> np.ndarray:
        """
        Aplica erosão (encolhe a máscara).
        
        Usado em regiões de contato para evitar aliasing.
        """
        if pixels <= 0:
            return mask
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (pixels * 2 + 1, pixels * 2 + 1)
        )
        
        return cv2.erode(mask, kernel, iterations=1)
    
    def apply_dilation(
        self,
        mask: np.ndarray,
        pixels: int = 1
    ) -> np.ndarray:
        """
        Aplica dilatação (expande a máscara).
        
        Usado para garantir overlap mínimo entre personagens.
        """
        if pixels <= 0:
            return mask
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (pixels * 2 + 1, pixels * 2 + 1)
        )
        
        return cv2.dilate(mask, kernel, iterations=1)
    
    def apply_gaussian_blur(
        self,
        mask: np.ndarray,
        sigma: float = 0.5
    ) -> np.ndarray:
        """
        Aplica Gaussian Blur na máscara.
        
        Evita hard edges que causam artefatos no IP-Adapter.
        Sigma pequeno (0.5) mantém precisão mas suaviza transições.
        """
        if sigma <= 0:
            return mask
        
        # Kernel size proporcional ao sigma (mínimo 3)
        ksize = int(6 * sigma) | 1  # Garante ímpar
        ksize = max(3, ksize)
        
        return cv2.GaussianBlur(mask, (ksize, ksize), sigma)
    
    # =========================================================================
    # Utilidades
    # =========================================================================
    
    @staticmethod
    def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calcula Intersection over Union entre duas máscaras."""
        intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
        union = np.logical_or(mask1 > 0, mask2 > 0).sum()
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def compute_overlap_area(mask1: np.ndarray, mask2: np.ndarray) -> int:
        """Calcula área de sobreposição entre duas máscaras."""
        return int(np.logical_and(mask1 > 0, mask2 > 0).sum())
    
    @staticmethod
    def create_background_mask(
        character_masks: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Cria máscara de background (inverso dos personagens).
        
        Usada para Background Isolation no IP-Adapter.
        """
        if not character_masks:
            return np.array([])
        
        # Combina todas as máscaras
        masks_list = list(character_masks.values())
        combined = np.maximum.reduce(masks_list)
        
        # Background é o inverso
        background = 255 - combined
        
        return background


def create_mask_processor(
    close_kernel: int = 3,
    erosion_pixels: int = 2,
    blur_sigma: float = 0.5
) -> MaskProcessor:
    """Factory function para criar processador de máscaras."""
    return MaskProcessor(
        close_kernel_size=close_kernel,
        erosion_pixels=erosion_pixels,
        blur_sigma=blur_sigma
    )

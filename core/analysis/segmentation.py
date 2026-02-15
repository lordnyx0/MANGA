"""
MangaAutoColor Pro - SAM 2.1 Tiny Segmenter (ADR 004)

Implementa segmentação semântica densa usando SAM 2.1 Tiny (35MB).
Substitui Bounding Boxes por máscaras precisas para eliminar color bleeding.

Armazenamento: Usa RLE encoding diretamente no Parquet (não .npy avulsos)
para evitar overhead de I/O com muitos arquivos pequenos.

Baseado em:
- SAM 2.1 (Meta AI): https://github.com/facebookresearch/sam2
- ADR 004: Segmentação Semântica com Z-Buffer Hierárquico
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import cv2
import warnings

from core.detection.yolo_detector import DetectionResult
from core.logging.setup import get_logger

logger = get_logger("SAM2Segmenter")


@dataclass
class SegmentationResult:
    """
    Resultado da segmentação para um personagem.
    
    A máscara é armazenada em formato RLE (Run-Length Encoding) para
    eficiência de espaço e I/O. Isso evita criar milhares de arquivos .npy.
    """
    char_id: str
    rle_mask: str  # Máscara codificada em RLE
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) na imagem original
    mask_shape: Tuple[int, int]  # (height, width) da máscara
    confidence: float  # Confiança média dos pixels da máscara
    area_pixels: int  # Área em pixels da máscara
    
    @property
    def mask(self) -> np.ndarray:
        """Decodifica máscara RLE para numpy array (lazy loading)."""
        return RLECodec.decode(self.rle_mask, self.mask_shape[0], self.mask_shape[1])
    
    @classmethod
    def from_mask(
        cls,
        char_id: str,
        mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
        confidence: float = 1.0
    ) -> 'SegmentationResult':
        """Cria resultado a partir de máscara numpy."""
        rle = RLECodec.encode(mask)
        area = int(np.sum(mask > 0))
        return cls(
            char_id=char_id,
            rle_mask=rle,
            bbox=bbox,
            mask_shape=mask.shape,
            confidence=confidence,
            area_pixels=area
        )


class RLECodec:
    """
    Run-Length Encoding para máscaras binárias compactas.
    
    Formato: "w,h,count,value,count,value,..."
    Exemplo: "100,100,500,1,500,0,..." (500px foreground, 500px background)
    
    Inspirado no COCO API mas simplificado para uso interno.
    """
    
    @staticmethod
    def encode(mask: np.ndarray) -> str:
        """
        Codifica máscara binária para string RLE.
        
        Args:
            mask: Array binário (H, W) com valores 0 ou 1/255
            
        Returns:
            String RLE compacta
        """
        if mask.size == 0:
            return "0,0"
        
        h, w = mask.shape
        # Flatten e binariza (0 ou 1)
        flat = (mask.flatten() > 0).astype(np.uint8)
        
        # RLE encoding otimizado
        if len(flat) == 0:
            return f"{w},{h}"
        
        # Encontra mudanças de valor
        changes = np.where(flat[:-1] != flat[1:])[0] + 1
        # Índices onde ocorrem mudanças: iniciais + mudanças + final
        indices = np.concatenate([[0], changes, [len(flat)]])
        
        # Calcula run lengths
        runs = np.diff(indices)
        values = flat[indices[:-1]]
        
        # Codifica: começa sempre com o valor do primeiro pixel
        # Se começar com 0, primeiro run é background
        encoded = [w, h]  # Header com dimensões
        
        for run, val in zip(runs, values):
            encoded.extend([int(run), int(val)])
        
        return ",".join(map(str, encoded))
    
    @staticmethod
    def decode(rle: str, height: Optional[int] = None, 
               width: Optional[int] = None) -> np.ndarray:
        """
        Decodifica string RLE para máscara binária.
        
        Args:
            rle: String RLE
            height, width: Dimensões (opcional se codificadas no RLE)
            
        Returns:
            Array binário (H, W) dtype=uint8 (0 ou 255)
        """
        if not rle or rle == "0,0":
            return np.array([])
        
        parts = list(map(int, rle.split(',')))
        
        if len(parts) < 2:
            return np.array([])
        
        # Extrai dimensões do header
        w, h = parts[0], parts[1]
        
        # Se dimensões fornecidas explicitamente, usa elas
        if height is not None and width is not None:
            h, w = height, width
        
        if w == 0 or h == 0:
            return np.array([])
        
        # Decodifica runs
        runs = parts[2:]
        
        # Reconstrói máscara
        mask = []
        for i in range(0, len(runs), 2):
            if i + 1 < len(runs):
                count, value = runs[i], runs[i + 1]
                mask.extend([value] * count)
        
        mask = np.array(mask, dtype=np.uint8)
        
        # Garante tamanho correto
        expected_size = w * h
        if len(mask) != expected_size:
            # Ajusta se necessário (trunca ou preenche)
            if len(mask) > expected_size:
                mask = mask[:expected_size]
            else:
                mask = np.pad(mask, (0, expected_size - len(mask)), mode='constant')
        
        mask = mask.reshape(h, w) * 255  # Escala para 0-255
        return mask.astype(np.uint8)


class SAM2Segmenter:
    """
    Segmentador SAM 2.1 Tiny para mangá.
    
    Gera máscaras precisas de personagens a partir de bounding boxes YOLO.
    Opera em CPU durante Pass 1 para preservar VRAM para Pass 2.
    
    Fallback: Se SAM 2.1 não disponível, retorna máscaras BBox retangulares.
    
    Args:
        model_size: Tamanho do modelo ('tiny', 'small', 'base', 'large')
        device: Dispositivo ('cpu' recomendado para Pass 1)
        enabled: Se False, sempre usa fallback BBox
    """
    
    MODEL_CONFIGS = {
        'tiny': {
            'repo_id': 'facebook/sam2-hiera-tiny',
            'config': 'sam2_hiera_t.yaml',
            'size_mb': 35
        },
        'small': {
            'repo_id': 'facebook/sam2-hiera-small',
            'config': 'sam2_hiera_s.yaml',
            'size_mb': 80
        },
        'base': {
            'repo_id': 'facebook/sam2-hiera-base-plus',
            'config': 'sam2_hiera_b+.yaml',
            'size_mb': 180
        },
        'large': {
            'repo_id': 'facebook/sam2-hiera-large',
            'config': 'sam2_hiera_l.yaml',
            'size_mb': 400
        }
    }
    
    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cpu",
        enabled: bool = True
    ):
        self.model_size = model_size.lower()
        self.device = device
        self.enabled = enabled
        
        # Lazy loading
        self._predictor = None
        self._image_cache_id = None
        
        if not enabled:
            logger.info("SAM2Segmenter desabilitado (usará fallback BBox)")
        elif self.model_size not in self.MODEL_CONFIGS:
            raise ValueError(f"Model size '{model_size}' não suportado. "
                           f"Use: {list(self.MODEL_CONFIGS.keys())}")
        else:
            logger.info(f"SAM2Segmenter inicializado (size={model_size}, device={device})")
    
    def _load_model(self) -> bool:
        """Carrega modelo SAM 2 (lazy loading). Retorna True se sucesso."""
        if self._predictor is not None:
            return True
        
        if not self.enabled:
            return False
        
        try:
            # Tenta importar SAM 2
            from sam2.build_sam import build_sam2_hf
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            config = self.MODEL_CONFIGS[self.model_size]
            
            logger.info(f"Carregando SAM 2 {self.model_size} ({config['size_mb']}MB)...")
            
            # Build do modelo via HuggingFace
            # SAM2 v1.1.0: build_sam2_hf(model_id, **kwargs)
            model = build_sam2_hf(config['repo_id'])
            model = model.to(self.device)
            model.eval()
            
            # Cria predictor
            self._predictor = SAM2ImagePredictor(model)
            
            logger.info(f"SAM 2 {self.model_size} carregado com sucesso!")
            return True
            
        except ImportError as e:
            logger.warning(f"sam2 não instalado: {e}. Usando fallback BBox.")
            return False
        except Exception as e:
            logger.error(f"Erro ao carregar SAM 2: {e}")
            return False
    
    def segment(
        self,
        image: np.ndarray,
        detections: List[DetectionResult],
        char_ids: Optional[List[str]] = None
    ) -> Dict[str, SegmentationResult]:
        """
        Segmenta personagens na imagem.
        
        Args:
            image: Imagem numpy (H, W, 3) RGB
            detections: Lista de detecções YOLO
            char_ids: IDs opcionais para cada detecção (gera se None)
            
        Returns:
            Dict mapeando char_id -> SegmentationResult (máscara em RLE)
        """
        h, w = image.shape[:2]
        
        # Gera char_ids se não fornecidos
        if char_ids is None:
            char_ids = [f"char_{i:03d}" for i in range(len(detections))]
        
        # Tenta carregar SAM
        sam_available = self._load_model()
        
        if not sam_available:
            logger.debug("SAM não disponível, usando fallback BBox")
            return self._fallback_bbox_masks(image, detections, char_ids)
        
        results = {}
        
        # Set image no predictor (cache de features)
        self._set_image(image)
        
        # Segmenta cada detecção
        for det, char_id in zip(detections, char_ids):
            try:
                result = self._segment_single(det, char_id, (h, w))
                if result is not None:
                    results[char_id] = result
            except Exception as e:
                logger.warning(f"Falha ao segmentar {char_id}: {e}")
                # Fallback para BBox
                result = self._fallback_single_bbox(det, char_id, (h, w))
                if result is not None:
                    results[char_id] = result
        
        logger.info(f"Segmentação completa: {len(results)}/{len(detections)} personagens")
        return results
    
    def _set_image(self, image: np.ndarray):
        """Set image no predictor com cache de features."""
        image_id = id(image)
        if self._image_cache_id == image_id:
            return
        
        self._predictor.set_image(image)
        self._image_cache_id = image_id
    
    def _segment_single(
        self,
        detection: DetectionResult,
        char_id: str,
        image_shape: Tuple[int, int]
    ) -> Optional[SegmentationResult]:
        """Segmenta um único personagem usando SAM."""
        h, w = image_shape
        x1, y1, x2, y2 = detection.bbox
        
        # Garante coordenadas válidas
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Ponto central como hint positivo
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Box como hint espacial
        input_box = np.array([x1, y1, x2, y2])
        
        try:
            # Predict com point e box
            masks, scores, _ = self._predictor.predict(
                point_coords=np.array([[cx, cy]]),
                point_labels=np.array([1]),  # 1 = foreground
                box=input_box[None, :],
                multimask_output=False,
            )
            
            if masks is None or len(masks) == 0:
                return None
            
            # Extrai melhor máscara
            mask = masks[0].astype(np.uint8)
            score = float(scores[0]) if scores is not None else 0.5
            
            return SegmentationResult.from_mask(
                char_id=char_id,
                mask=mask,
                bbox=(x1, y1, x2, y2),
                confidence=score
            )
            
        except Exception as e:
            logger.debug(f"Erro em predict SAM: {e}")
            return None
    
    def _fallback_bbox_masks(
        self,
        image: np.ndarray,
        detections: List[DetectionResult],
        char_ids: List[str]
    ) -> Dict[str, SegmentationResult]:
        """Fallback: Gera máscaras simples a partir de BBoxes."""
        h, w = image.shape[:2]
        results = {}
        
        for det, char_id in zip(detections, char_ids):
            result = self._fallback_single_bbox(det, char_id, (h, w))
            if result is not None:
                results[char_id] = result
        
        return results
    
    def _fallback_single_bbox(
        self,
        detection: DetectionResult,
        char_id: str,
        image_shape: Tuple[int, int]
    ) -> Optional[SegmentationResult]:
        """Fallback para uma única detecção (máscara retangular)."""
        h, w = image_shape
        x1, y1, x2, y2 = detection.bbox
        
        # Clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Cria máscara retangular simples
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        
        return SegmentationResult.from_mask(
            char_id=char_id,
            mask=mask,
            bbox=(x1, y1, x2, y2),
            confidence=detection.confidence
        )
    
    def unload(self):
        """Libera memória do modelo."""
        self._predictor = None
        self._image_cache_id = None
        
        import gc
        gc.collect()
        
        if self.device == "cuda":
            import torch
            torch.cuda.empty_cache()
        
        logger.info("SAM 2 descarregado")


def create_sam2_segmenter(
    enabled: bool = True,
    model_size: str = "tiny",
    device: str = "cpu"
) -> SAM2Segmenter:
    """Factory function para criar segmentador."""
    return SAM2Segmenter(model_size=model_size, device=device, enabled=enabled)

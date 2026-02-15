"""
MangaAutoColor Pro - Z-Buffer Hierárquico (ADR 004)

Calcula ordenação de profundidade relativa entre personagens usando
heurísticas visuais específicas para mangá/HQs.

Fórmula (ADR 004):
D(p) = w₁·H(y_center) + w₂·(1 - A_max/area(p)) + w₃·τ(type_p) + w₄·δ(p)

Onde:
- H(y_center): Posição vertical (quanto mais baixo = mais "na frente")
- area(p): Área da detecção
- τ(type_p): Prioridade semântica (face=0.0, body=0.5, frame=1.0)
- δ(p): Profundidade MiDaS (opcional)

Ajustes para Close-ups:
- Faces grandes (close-up) têm prioridade aumentada via área
- O peso da área (w₂) ajuda a compensar a posição Y em close-ups
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from core.detection.yolo_detector import DetectionResult
from core.logging.setup import get_logger

logger = get_logger("ZBuffer")


class DetectionPriority(float, Enum):
    """
    Prioridade semântica para ordenação de profundidade.
    
    Valores menores = mais à frente (menor profundidade).
    """
    FACE = 0.0       # Rosto deve estar na frente (close-up)
    BODY = 0.5       # Corpo no meio
    FRAME = 1.0      # Quadro/painel no fundo
    TEXT = 0.3       # Texto entre face e body
    UNKNOWN = 0.5


@dataclass
class ZBufferWeights:
    """Pesos configuráveis para a fórmula de profundidade."""
    y_position: float = 0.5      # w₁: Posição vertical
    area: float = 0.3            # w₂: Área relativa
    semantic_type: float = 0.2   # w₃: Prioridade semântica
    midas_depth: float = 0.0     # w₄: Profundidade MiDaS (se disponível)
    
    def validate(self):
        """Valida que pesos somam aproximadamente 1.0."""
        total = self.y_position + self.area + self.semantic_type + self.midas_depth
        if not (0.99 <= total <= 1.01):
            logger.warning(f"Pesos Z-Buffer somam {total:.2f}, esperado ~1.0")


@dataclass  
class DepthResult:
    """Resultado do cálculo de profundidade para uma detecção."""
    char_id: str
    detection: DetectionResult
    depth_score: float  # Quanto MENOR = mais à FRENTE
    components: Dict[str, float]  # Componentes individuais para debug
    rank: int = 0  # Ordem de profundidade (1 = mais à frente)
    
    @property
    def is_foreground(self) -> bool:
        """Retorna True se este personagem está em primeiro plano."""
        return self.rank == 1


class ZBufferCalculator:
    """
    Calculador de Z-Buffer Hierárquico para ordenação de personagens.
    
    Resolve sobreposições em cenas complexas ordenando personagens
    por profundidade relativa estimada.
    
    Args:
        weights: Pesos para a fórmula de profundidade
        use_midas: Se True, tenta usar estimativa de profundidade MiDaS
    """
    
    def __init__(
        self,
        weights: Optional[ZBufferWeights] = None,
        use_midas: bool = False
    ):
        self.weights = weights or ZBufferWeights()
        self.weights.validate()
        self.use_midas = use_midas
        self._midas_model = None
        
        logger.info(f"ZBufferCalculator inicializado (use_midas={use_midas})")
    
    def calculate_depth(
        self,
        detection: DetectionResult,
        image_size: Tuple[int, int],
        max_area: float,
        char_id: str = "unknown"
    ) -> DepthResult:
        """
        Calcula profundidade para uma única detecção.
        
        Args:
            detection: Detecção YOLO
            image_size: (width, height) da imagem
            max_area: Maior área entre todas as detecções (para normalização)
            char_id: ID do personagem
            
        Returns:
            DepthResult com score e componentes
        """
        img_w, img_h = image_size
        x1, y1, x2, y2 = detection.bbox
        
        # Centro Y normalizado (0 = topo, 1 = fundo)
        # Quanto maior Y_center = mais embaixo na imagem = mais "na frente"
        y_center = (y1 + y2) / 2
        y_normalized = y_center / img_h  # 0 a 1
        
        # Componente 1: Posição Y
        # Em mangá: mais baixo na página = mais à frente (menor profundidade)
        # Fórmula: H(y_center) = 1 - y_normalized
        y_component = 1.0 - y_normalized
        
        # Componente 2: Área relativa
        # Fórmula: min(1.0, area(p) / A_max) invertido
        # Maior área = menor depth score (mais à frente)
        area = (x2 - x1) * (y2 - y1)
        if max_area > 0 and area > 0:
            # Limita a 1.0 (caso area > max_area devido a arredondamentos)
            area_ratio = min(1.0, area / max_area)
            area_component = 1.0 - area_ratio  # 0 para maior área, 1 para menor
        else:
            area_component = 0.5
        
        # Componente 3: Prioridade semântica do tipo
        type_priority = self._get_type_priority(detection.class_id)
        
        # Componente 4: Profundidade MiDaS (se habilitado)
        midas_component = 0.0
        if self.use_midas:
            midas_component = self._estimate_midas_depth(detection.bbox, image_size)
        
        # Calcula score final ponderado
        # Quanto MENOR o score = mais à FRENTE
        depth_score = (
            self.weights.y_position * y_component +
            self.weights.area * area_component +
            self.weights.semantic_type * type_priority +
            self.weights.midas_depth * midas_component
        )
        
        return DepthResult(
            char_id=char_id,
            detection=detection,
            depth_score=depth_score,
            components={
                'y_position': y_component,
                'area': area_component,
                'semantic_type': type_priority,
                'midas': midas_component,
                'raw_area': area,
                'max_area': max_area,
                'y_normalized': y_normalized
            }
        )
    
    def sort_by_depth(
        self,
        detections: List[DetectionResult],
        image_size: Tuple[int, int],
        char_ids: Optional[List[str]] = None
    ) -> List[DepthResult]:
        """
        Ordena detecções por profundidade (da frente para o fundo).
        
        Args:
            detections: Lista de detecções YOLO
            image_size: (width, height) da imagem
            char_ids: IDs dos personagens (gera se None)
            
        Returns:
            Lista de DepthResult ordenada por profundidade
            (índice 0 = mais à frente, índice N = mais ao fundo)
        """
        if not detections:
            return []
        
        if char_ids is None:
            char_ids = [f"char_{i:03d}" for i in range(len(detections))]
        
        # Calcula área máxima para normalização
        max_area = 0
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            area = (x2 - x1) * (y2 - y1)
            max_area = max(max_area, area)
        
        # Calcula profundidade para cada detecção
        results = []
        for det, char_id in zip(detections, char_ids):
            result = self.calculate_depth(det, image_size, max_area, char_id)
            results.append(result)
        
        # Ordena por depth_score (menor = mais à frente)
        results.sort(key=lambda x: x.depth_score)
        
        # Atribui ranks
        for rank, result in enumerate(results, start=1):
            result.rank = rank
        
        # Log da ordenação
        if results:
            logger.debug("Ordenação Z-Buffer:")
            for r in results:
                logger.debug(f"  Rank {r.rank}: {r.char_id} "
                           f"(score={r.depth_score:.3f}, "
                           f"y={r.components['y_normalized']:.2f}, "
                           f"area={r.components['raw_area']})")
        
        return results
    
    def get_occlusion_order(
        self,
        detections: List[DetectionResult],
        image_size: Tuple[int, int],
        char_ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Retorna lista de char_ids ordenados da frente para o fundo.
        
        Útil para aplicação sequencial de máscaras.
        """
        results = self.sort_by_depth(detections, image_size, char_ids)
        return [r.char_id for r in results]
    
    def _get_type_priority(self, class_id: int) -> float:
        """Retorna prioridade semântica baseada na classe YOLO."""
        # Mapeamento de classes YOLO (Manga109)
        class_map = {
            0: DetectionPriority.BODY,   # body
            1: DetectionPriority.FACE,   # face
            2: DetectionPriority.FRAME,  # frame
            3: DetectionPriority.TEXT,   # text
        }
        return float(class_map.get(class_id, DetectionPriority.UNKNOWN))
    
    def _estimate_midas_depth(
        self,
        bbox: Tuple[int, int, int, int],
        image_size: Tuple[int, int]
    ) -> float:
        """
        Estima profundidade usando MiDaS Small (se disponível).
        
        Retorna valor 0-1 onde 0 = próximo, 1 = longe.
        """
        if self._midas_model is None:
            # TODO: Implementar carregamento do MiDaS Small
            return 0.5  # Valor neutro se não disponível
        
        # Placeholder - implementação real requer MiDaS
        return 0.5
    
    def _load_midas(self):
        """Carrega modelo MiDaS Small (lazy loading)."""
        if not self.use_midas or self._midas_model is not None:
            return
        
        try:
            # TODO: Implementar carregamento do MiDaS Small
            # import torch
            # self._midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            pass
        except Exception as e:
            logger.warning(f"Não foi possível carregar MiDaS: {e}")
            self._midas_model = None


def create_zbuffer_calculator(
    enabled: bool = True,
    y_weight: float = 0.5,
    area_weight: float = 0.3,
    type_weight: float = 0.2,
    use_midas: bool = False
) -> Optional[ZBufferCalculator]:
    """
    Factory function para criar calculador Z-Buffer.
    
    Sem MiDaS: y=0.5, area=0.3, type=0.2 (soma=1.0)
    Com MiDaS: y=0.4, area=0.3, type=0.2, midas=0.1 (soma=1.0)
    """
    if not enabled:
        return None
    
    if use_midas:
        # Redistribui peso: tira 0.1 do y_position para midas
        weights = ZBufferWeights(
            y_position=y_weight - 0.1 if y_weight >= 0.2 else y_weight,
            area=area_weight,
            semantic_type=type_weight,
            midas_depth=0.1
        )
    else:
        weights = ZBufferWeights(
            y_position=y_weight,
            area=area_weight,
            semantic_type=type_weight,
            midas_depth=0.0
        )
    
    return ZBufferCalculator(weights=weights, use_midas=use_midas)

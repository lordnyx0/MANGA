"""
MangaAutoColor Pro - Analysis Components (ADR 004 & 005)

Módulos para análise no Pass 1:
- segmentation: SAM 2.1 Tiny para segmentação semântica (ADR 004)
- z_buffer: Z-Buffer Hierárquico para ordenação de profundidade (ADR 004)
- mask_processor: Pós-processamento de máscaras (morfologia, RLE) (ADR 004)
- point_matching: Correspondência de keypoints com LightGlue (ADR 005)
- temporal_flow: Consistência temporal com RAFT/AdaIN (ADR 005)
"""

from .segmentation import SAM2Segmenter, SegmentationResult
from .z_buffer import ZBufferCalculator, ZBufferWeights
from .mask_processor import MaskProcessor, MaskOperations
__all__ = [
    # ADR 004
    'SAM2Segmenter',
    'SegmentationResult',
    'ZBufferCalculator',
    'ZBufferWeights',
    'MaskProcessor',
    'MaskOperations',
]

"""
MangaAutoColor Pro - Módulo de Detecção

Exporta:
- YOLODetector: Detecção de personagens com YOLOv8
- CannyContinuityNMS: NMS customizado com continuidade de bordas
"""

from .yolo_detector import YOLODetector, DetectionResult
from .nms_custom import CannyContinuityNMS, apply_nms_numpy

__all__ = [
    'YOLODetector',
    'DetectionResult',
    'CannyContinuityNMS',
    'apply_nms_numpy'
]

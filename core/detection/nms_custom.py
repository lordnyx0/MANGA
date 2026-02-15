"""
MangaAutoColor Pro - NMS Customizado com Canny Continuity

Implementa Non-Maximum Suppression (NMS) com heurística de continuidade de bordas.
Em mangás, personagens próximos podem ser parte da mesma figura se houver
continuidade nas linhas (Canny edges), evitando merges incorretos.

Baseado no conceito de que em arte de mangá, contornos contínuos indicam
conexão entre partes de um personagem.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter

from config.settings import DETECTION_IOU_THRESHOLD, VERBOSE


@dataclass
class RawDetection:
    """Detecção bruta antes do NMS"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int = 0
    class_name: str = 'unknown'
    
    def to_dict(self) -> Dict:
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'class_name': self.class_name
        }


class CannyContinuityNMS:
    """
    NMS com consideração de continuidade de bordas (Canny edges).
    
    Em mangás, múltiplos bounding boxes sobrepostos podem representar:
    1. Partes diferentes do mesmo personagem (devem ser unidas)
    2. Personagens diferentes próximos (devem ser separados)
    
    A heurística de Canny continuity resolve isso verificando se há
    continuidade nas bordas entre as detecções candidatas.
    
    Args:
        iou_threshold: Threshold de IoU para considerar overlap
        canny_threshold: Threshold de continuidade de bordas (0-1)
        min_merge_score: Score mínimo para merge de detecções
    """
    
    def __init__(
        self,
        iou_threshold: float = DETECTION_IOU_THRESHOLD,
        canny_threshold: float = 0.3,
        min_merge_score: float = 0.5
    ):
        self.iou_threshold = iou_threshold
        self.canny_threshold = canny_threshold
        self.min_merge_score = min_merge_score
        
        if VERBOSE:
            print(f"[CannyContinuityNMS] Inicializado (iou_threshold={iou_threshold}, "
                  f"canny_threshold={canny_threshold})")
    
    def merge_by_canny_continuity(
        self,
        detections: List[Dict],
        canny_edges: np.ndarray
    ) -> List[Dict]:
        """
        Aplica NMS com merge baseado em continuidade de bordas.
        
        Args:
            detections: Lista de detecções brutas (dict com 'bbox', 'confidence')
            canny_edges: Mapa de bordas Canny (H, W) binário
            
        Returns:
            Lista de detecções mescladas
        """
        if not detections:
            return []
        
        if len(detections) == 1:
            return detections
        
        # Converte para RawDetection
        raw_dets = []
        for det in detections:
            bbox = det['bbox']
            if isinstance(bbox, list):
                bbox = tuple(bbox)
            raw_dets.append(RawDetection(
                bbox=bbox,
                confidence=det.get('confidence', 0.5),
                class_id=det.get('class_id', 0),
                class_name=det.get('class_name', 'unknown')
            ))
        
        # Ordena por confiança (decrescente)
        raw_dets.sort(key=lambda x: x.confidence, reverse=True)
        
        # Lista de detecções processadas
        merged = []
        suppressed = set()
        
        for i, det_i in enumerate(raw_dets):
            if i in suppressed:
                continue
            
            # Grupo de detecções para merge
            merge_group = [det_i]
            
            for j, det_j in enumerate(raw_dets[i+1:], start=i+1):
                if j in suppressed:
                    continue
                
                # Calcula IoU
                iou = self._calculate_iou(det_i.bbox, det_j.bbox)
                
                if iou > self.iou_threshold:
                    # Verifica continuidade de bordas
                    continuity = self._check_canny_continuity(
                        det_i.bbox, det_j.bbox, canny_edges
                    )
                    
                    # Merge se houver continuidade suficiente
                    if continuity > self.canny_threshold:
                        merge_group.append(det_j)
                        suppressed.add(j)
            
            # Mescla o grupo
            merged_det = self._merge_detections(merge_group, canny_edges)
            merged.append(merged_det.to_dict())
        
        if VERBOSE:
            print(f"[CannyContinuityNMS] {len(detections)} -> {len(merged)} detecções "
                  f"({len(detections) - len(merged)} mescladas)")
        
        return merged
    
    def _calculate_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """
        Calcula Intersection over Union (IoU) entre dois bboxes.
        
        Args:
            bbox1: (x1, y1, x2, y2)
            bbox2: (x1, y1, x2, y2)
            
        Returns:
            IoU (0.0 a 1.0)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Interseção
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        
        # União
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _check_canny_continuity(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int],
        canny_edges: np.ndarray
    ) -> float:
        """
        Verifica continuidade de bordas Canny entre dois bboxes.
        
        A ideia é que se há uma linha contínua (edge) conectando as duas
        detecções, elas provavelmente são partes do mesmo personagem.
        
        Args:
            bbox1: Primeiro bbox
            bbox2: Segundo bbox
            canny_edges: Mapa de bordas Canny
            
        Returns:
            Score de continuidade (0.0 a 1.0)
        """
        h, w = canny_edges.shape
        
        # Calcula região de overlap expandida
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Bounding box da união
        ux1 = min(x1_1, x1_2)
        uy1 = min(y1_1, y1_2)
        ux2 = max(x2_1, x2_2)
        uy2 = max(y2_1, y2_2)
        
        # Expande um pouco para capturar conexões
        margin = 20
        ux1 = max(0, ux1 - margin)
        uy1 = max(0, uy1 - margin)
        ux2 = min(w, ux2 + margin)
        uy2 = min(h, uy2 + margin)
        
        # Extrai região
        region = canny_edges[uy1:uy2, ux1:ux2]
        
        if region.size == 0:
            return 0.0
        
        # Cria máscaras para cada bbox dentro da região
        mask1 = np.zeros_like(region, dtype=np.uint8)
        mask2 = np.zeros_like(region, dtype=np.uint8)
        
        # Coordenadas relativas à região
        r_x1_1 = max(0, x1_1 - ux1)
        r_y1_1 = max(0, y1_1 - uy1)
        r_x2_1 = min(region.shape[1], x2_1 - ux1)
        r_y2_1 = min(region.shape[0], y2_1 - uy1)
        
        r_x1_2 = max(0, x1_2 - ux1)
        r_y1_2 = max(0, y1_2 - uy1)
        r_x2_2 = min(region.shape[1], x2_2 - ux1)
        r_y2_2 = min(region.shape[0], y2_2 - uy1)
        
        mask1[r_y1_1:r_y2_1, r_x1_1:r_x2_1] = 1
        mask2[r_y1_2:r_y2_2, r_x1_2:r_x2_2] = 1
        
        # Verifica se há caminho de edges conectando as duas máscaras
        # Simplificação: conta edges na região entre os dois bboxes
        
        # Região intermediária (dilatada)
        from scipy.ndimage import binary_dilation
        dilated1 = binary_dilation(mask1, iterations=10)
        dilated2 = binary_dilation(mask2, iterations=10)
        
        # Região de interesse: entre as duas detecções
        between_mask = dilated1 & dilated2
        
        # Se há overlap significativo nas dilatações, estão próximos
        overlap_ratio = np.sum(between_mask) / min(np.sum(mask1), np.sum(mask2))
        
        if overlap_ratio > 0.5:
            # Bboxes já estão muito próximos/sobrepostos
            return 1.0
        
        # Conta edges na região de conexão potencial
        # Desenha linha entre centros e verifica cobertura
        c1_x = (r_x1_1 + r_x2_1) // 2
        c1_y = (r_y1_1 + r_y2_1) // 2
        c2_x = (r_x1_2 + r_x2_2) // 2
        c2_y = (r_y1_2 + r_y2_2) // 2
        
        # Linha entre centros
        line_mask = np.zeros_like(region, dtype=np.uint8)
        cv2.line(line_mask, (c1_x, c1_y), (c2_x, c2_y), 1, thickness=5)
        
        # Quanto da linha está coberta por edges?
        line_edges = line_mask & region
        edge_coverage = np.sum(line_edges) / (np.sum(line_mask) + 1e-8)
        
        # Também verifica densidade de edges na região entre bboxes
        gap_region = dilated1 | dilated2
        gap_edges = region[gap_region > 0]
        edge_density = np.sum(gap_edges > 0) / (len(gap_edges) + 1e-8)
        
        # Score combinado
        continuity = 0.5 * edge_coverage + 0.5 * edge_density
        
        return min(1.0, continuity)
    
    def _merge_detections(
        self,
        detections: List[RawDetection],
        canny_edges: np.ndarray
    ) -> RawDetection:
        """
        Mescla múltiplas detecções em uma única.
        
        A bbox resultante é o envelope (union) das detecções,
        e a confiança é a máxima entre elas.
        
        Args:
            detections: Lista de detecções para mesclar
            canny_edges: Mapa de bordas (para ajuste fino)
            
        Returns:
            Detecção mesclada
        """
        if len(detections) == 1:
            return detections[0]
        
        # Encontra bbox envelope
        x1 = min(d.bbox[0] for d in detections)
        y1 = min(d.bbox[1] for d in detections)
        x2 = max(d.bbox[2] for d in detections)
        y2 = max(d.bbox[3] for d in detections)
        
        # Confiança máxima
        max_conf = max(d.confidence for d in detections)
        
        # Classe (assume mesma classe)
        class_id = detections[0].class_id
        
        # Nome da classe (do primeiro, ou mapeia do ID)
        class_name_map = {0: 'body', 1: 'face', 2: 'frame', 3: 'text'}
        class_name = getattr(detections[0], 'class_name', None) or class_name_map.get(class_id, 'unknown')
        
        # Opcional: ajusta bordas para seguir edges do Canny
        # (poderia ser implementado para maior precisão)
        
        return RawDetection(
            bbox=(x1, y1, x2, y2),
            confidence=max_conf,
            class_id=class_id,
            class_name=class_name
        )
    
    def suppress_small_detections(
        self,
        detections: List[Dict],
        min_area: int = 1000,
        min_dimension: int = 30
    ) -> List[Dict]:
        """
        Remove detecções muito pequenas (prováveis falsos positivos).
        
        Args:
            detections: Lista de detecções
            min_area: Área mínima em pixels
            min_dimension: Dimensão mínima de altura/largura
            
        Returns:
            Lista filtrada
        """
        filtered = []
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            w = x2 - x1
            h = y2 - y1
            area = w * h
            
            if area >= min_area and w >= min_dimension and h >= min_dimension:
                filtered.append(det)
        
        if VERBOSE:
            suppressed = len(detections) - len(filtered)
            if suppressed > 0:
                print(f"[CannyContinuityNMS] {suppressed} detecções pequenas suprimidas")
        
        return filtered
    
    def split_large_detections(
        self,
        detections: List[Dict],
        canny_edges: np.ndarray,
        max_aspect_ratio: float = 3.0
    ) -> List[Dict]:
        """
        Divide detecções com aspect ratio muito grande (possíveis múltiplos personagens).
        
        Args:
            detections: Lista de detecções
            canny_edges: Mapa de bordas
            max_aspect_ratio: Aspect ratio máximo antes de split
            
        Returns:
            Lista possivelmente expandida
        """
        result = []
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            w = x2 - x1
            h = y2 - y1
            
            aspect_ratio = max(w, h) / (min(w, h) + 1e-8)
            
            if aspect_ratio > max_aspect_ratio:
                # Possível grupo de personagens - tenta dividir
                # Simplificação: divide ao meio
                
                if w > h:
                    # Divide horizontalmente
                    mid = (x1 + x2) // 2
                    det1 = det.copy()
                    det1['bbox'] = (x1, y1, mid, y2)
                    det2 = det.copy()
                    det2['bbox'] = (mid, y1, x2, y2)
                    result.extend([det1, det2])
                else:
                    # Divide verticalmente
                    mid = (y1 + y2) // 2
                    det1 = det.copy()
                    det1['bbox'] = (x1, y1, x2, mid)
                    det2 = det.copy()
                    det2['bbox'] = (x1, mid, x2, y2)
                    result.extend([det1, det2])
            else:
                result.append(det)
        
        return result


def apply_nms_numpy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5
) -> List[int]:
    """
    Implementação pura de NMS usando NumPy (fallback).
    
    Args:
        boxes: Array (N, 4) de bboxes [x1, y1, x2, y2]
        scores: Array (N,) de scores
        iou_threshold: Threshold de IoU
        
    Returns:
        Índices das detecções mantidas
    """
    if len(boxes) == 0:
        return []
    
    # Ordena por score
    indices = np.argsort(scores)[::-1]
    
    keep = []
    
    while len(indices) > 0:
        # Pega o de maior score
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Calcula IoU com os demais
        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]
        
        # Interseção
        x1 = np.maximum(current_box[0], other_boxes[:, 0])
        y1 = np.maximum(current_box[1], other_boxes[:, 1])
        x2 = np.minimum(current_box[2], other_boxes[:, 2])
        y2 = np.minimum(current_box[3], other_boxes[:, 3])
        
        inter_w = np.maximum(0, x2 - x1)
        inter_h = np.maximum(0, y2 - y1)
        inter_area = inter_w * inter_h
        
        # União
        area_current = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        area_others = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
        union_area = area_current + area_others - inter_area
        
        iou = inter_area / (union_area + 1e-8)
        
        # Remove os que têm IoU alta
        mask = iou <= iou_threshold
        indices = indices[1:][mask]
    
    return keep

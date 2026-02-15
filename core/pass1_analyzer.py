"""
MangaAutoColor Pro - Passo 1: Análise Completa
Analisa todas as páginas do capítulo para extrair informações globais.

ADR 004: Integra SAM 2.1 Tiny para segmentação semântica e Z-Buffer Hierárquico
para ordenação de profundidade, eliminando color bleeding em oclusões.

"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2
from dataclasses import dataclass, field

from config.settings import (
    DEVICE, DTYPE, CONTEXT_INFLATION_FACTOR,
    YOLO_CONFIDENCE,
    SAM2_ENABLED, SAM2_MODEL_SIZE, SAM2_DEVICE,
    ZBUFFER_ENABLED, ZBUFFER_WEIGHT_Y, ZBUFFER_WEIGHT_AREA, ZBUFFER_WEIGHT_TYPE
)
from core.constants import DetectionClass, SceneType
from core.utils.image_ops import create_context_crop, extract_canny_edges
from core.logging.setup import get_logger

logger = get_logger("Pass1Analyzer")


@dataclass
class Detection:
    """Detecção de personagem em uma página (ADR 004: agora com segmentação)"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    crop: np.ndarray
    context_crop: np.ndarray  # Com inflation de 150%
    page_num: int
    class_id: int = DetectionClass.BODY.value
    class_name: str = "body"
    # ADR 004: Novos campos para segmentação e profundidade
    char_id: Optional[str] = None
    sam_mask_rle: Optional[str] = None  # Máscara SAM em RLE
    depth_score: float = 0.0  # Score de profundidade (menor = mais à frente)
    depth_rank: int = 0  # Ordem de profundidade (1 = mais à frente)
    mask_shape: Tuple[int, int] = field(default_factory=lambda: (0, 0))


@dataclass
class PageAnalysis:
    """Resultado da análise de uma página (ADR 004/005)"""
    page_num: int
    image_path: str
    detections: List[Detection]
    characters: List[Dict]
    scene_type: str
    lineart: Optional[np.ndarray] = None
    text_mask: Optional[np.ndarray] = None
    # ADR 004: Ordenação de profundidade para o Pass 2
    depth_order: List[str] = field(default_factory=list)  # char_ids da frente para o fundo


class Pass1Analyzer:
    """
    Analisador do Passo 1 (ADR 004: SAM 2.1 + Z-Buffer, ADR 005: PCTC).
    
    Responsável por:
    - Detecção de personagens (YOLO + NMS customizado)
    - Segmentação semântica (SAM 2.1 Tiny)
    - Cálculo de profundidade (Z-Buffer Hierárquico)
    - Extração de identidades e paletas
    - Detecção de contexto narrativo
    """
    
    def __init__(
        self,
        device: str = DEVICE,
        dtype: torch.dtype = DTYPE,
        confidence_threshold: float = YOLO_CONFIDENCE,
        # ADR 004: Configurações de segmentação
        enable_sam2: bool = SAM2_ENABLED,
        sam2_model_size: str = SAM2_MODEL_SIZE,
        sam2_device: str = SAM2_DEVICE,
        enable_zbuffer: bool = ZBUFFER_ENABLED
    ):
        self.device = device
        self.dtype = dtype
        self.confidence_threshold = confidence_threshold
        
        # ADR 004: Configurações
        self.enable_sam2 = enable_sam2
        self.sam2_model_size = sam2_model_size
        self.sam2_device = sam2_device
        self.enable_zbuffer = enable_zbuffer
        
        # Componentes (lazy loading)
        self._yolo_detector = None
        self._nms_processor = None
        self._identity_encoder = None
        self._palette_extractor = None
        self._scene_detector = None
        # ADR 004: Novos componentes
        self._sam2_segmenter = None
        self._zbuffer_calculator = None
        self._mask_processor = None
        
        logger.info(f"Pass1Analyzer inicializado (device={device}, "
                   f"sam2={enable_sam2}, zbuffer={enable_zbuffer})")
    
    def _get_yolo_detector(self):
        """Lazy loading do YOLO"""
        if self._yolo_detector is None:
            from .detection.yolo_detector import YOLODetector
            self._yolo_detector = YOLODetector(
                device=self.device,
                conf_threshold=self.confidence_threshold
            )
        return self._yolo_detector
    
    def _get_nms_processor(self):
        """Lazy loading do NMS customizado"""
        if self._nms_processor is None:
            from .detection.nms_custom import CannyContinuityNMS
            self._nms_processor = CannyContinuityNMS()
        return self._nms_processor
    
    def _get_identity_encoder(self):
        """Lazy loading do encoder de identidade"""
        if self._identity_encoder is None:
            from .identity.hybrid_encoder import HybridIdentitySystem
            self._identity_encoder = HybridIdentitySystem(
                device=self.device,
                dtype=self.dtype
            )
        return self._identity_encoder
    
    def _get_palette_extractor(self):
        """Lazy loading do extrator de paleta"""
        if self._palette_extractor is None:
            from .identity.palette_manager import PaletteExtractor
            self._palette_extractor = PaletteExtractor()
        return self._palette_extractor
    
    @property
    def yolo_detector(self):
        return self._get_yolo_detector()

    @property
    def identity_encoder(self):
        return self._get_identity_encoder()

    @property
    def palette_extractor(self):
        return self._get_palette_extractor()

    def _get_scene_detector(self):
        """Lazy loading do detector de cenas"""
        if self._scene_detector is None:
            from .narrative.scene_detector import SceneDetector
            self._scene_detector = SceneDetector()
        return self._scene_detector
    
    # ADR 004: Novos componentes =================================================
    
    def _get_sam2_segmenter(self):
        """Lazy loading do SAM 2.1 Segmenter"""
        if self._sam2_segmenter is None and self.enable_sam2:
            from .analysis.segmentation import create_sam2_segmenter
            self._sam2_segmenter = create_sam2_segmenter(
                enabled=self.enable_sam2,
                model_size=self.sam2_model_size,
                device=self.sam2_device
            )
        return self._sam2_segmenter
    
    def _get_zbuffer_calculator(self):
        """Lazy loading do Z-Buffer Calculator"""
        if self._zbuffer_calculator is None and self.enable_zbuffer:
            from .analysis.z_buffer import create_zbuffer_calculator
            self._zbuffer_calculator = create_zbuffer_calculator(
                enabled=self.enable_zbuffer,
                y_weight=ZBUFFER_WEIGHT_Y,
                area_weight=ZBUFFER_WEIGHT_AREA,
                type_weight=ZBUFFER_WEIGHT_TYPE
            )
        return self._zbuffer_calculator
    
    def _get_mask_processor(self):
        """Lazy loading do Mask Processor"""
        if self._mask_processor is None:
            from .analysis.mask_processor import create_mask_processor
            self._mask_processor = create_mask_processor()
        return self._mask_processor
    

    
    def analyze_page(self, image_path: str, page_num: int) -> Dict:
        """
        Analisa uma única página do capítulo.
        
        ADR 004: Agora inclui segmentação SAM 2.1 e cálculo Z-Buffer.
        
        Args:
            image_path: Caminho para a imagem da página
            page_num: Número da página (0-based)
            
        Returns:
            Dict com dados da análise incluindo máscaras RLE e depth_order
        """
        logger.info(f"Analisando página {page_num}: {image_path}")
        
        # Carrega imagem
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        img_h, img_w = image_np.shape[:2]
        
        # 1. Detecção de personagens (YOLO)
        detections = self._detect_characters(image_np, page_num)
        
        # ADR 004: 2. Segmentação SAM 2.1 + Z-Buffer
        if detections and (self.enable_sam2 or self.enable_zbuffer):
            detections = self._apply_segmentation_and_depth(
                image_np, detections, (img_w, img_h)
            )
        
        # 3. Extração de identidades e paletas
        characters = []
        for det in detections:
            char_data = self._extract_character_data(det, image_np)
            characters.append(char_data)
        
        # 4. Detecção de contexto narrativo
        scene_type = self._detect_scene_type(image_np)
        
        # 5. Extrai lineart e máscara de texto
        lineart = self._extract_lineart(image_np)
        text_mask = self._detect_text_regions(image_np)
        
        # ADR 004: Extrai depth_order
        depth_order = [det.char_id for det in detections if det.char_id]
        
        return {
            'page_num': page_num,
            'image_path': image_path,
            'image_size': image.size,
            'detections': [
                {
                    'bbox': det.bbox,
                    'confidence': det.confidence,
                    'page_num': det.page_num,
                    'class_id': det.class_id,
                    'class_name': det.class_name,
                    # ADR 004: Inclui dados de segmentação
                    'char_id': det.char_id,
                    'sam_mask_rle': det.sam_mask_rle,
                    'mask_shape': det.mask_shape,
                    'depth_score': det.depth_score,
                    'depth_rank': det.depth_rank,
                }
                for det in detections
            ],
            'characters': characters,
            'scene_type': scene_type,
            'lineart': lineart,
            'text_mask': text_mask,
            # ADR 004: Ordenação de profundidade para Pass 2
            'depth_order': depth_order,
        }
    
    def _apply_segmentation_and_depth(
        self,
        image: np.ndarray,
        detections: List[Detection],
        image_size: Tuple[int, int]
    ) -> List[Detection]:
        """
        ADR 004: Aplica segmentação SAM e cálculo de profundidade.
        
        Args:
            image: Imagem numpy
            detections: Lista de detecções YOLO
            image_size: (width, height)
            
        Returns:
            Detecções atualizadas com char_id, sam_mask_rle, depth_score
        """
        # Filtra apenas detecções de personagens (não texto)
        char_detections = [d for d in detections 
                          if d.class_id in (DetectionClass.BODY.value, DetectionClass.FACE.value)]
        
        if not char_detections:
            return detections
        
        # Gera char_ids únicos
        for i, det in enumerate(char_detections):
            det.char_id = f"char_{det.page_num:03d}_{i:03d}"
        
        # 1. Segmentação SAM 2.1
        sam_results = {}
        if self.enable_sam2:
            try:
                segmenter = self._get_sam2_segmenter()
                if segmenter:
                    from core.detection.yolo_detector import DetectionResult
                    # Converte Detection para DetectionResult
                    yolo_dets = []
                    for det in char_detections:
                        yd = DetectionResult(
                            bbox=det.bbox,
                            confidence=det.confidence,
                            class_id=det.class_id,
                            class_name=det.class_name
                        )
                        yolo_dets.append(yd)
                    
                    char_ids = [d.char_id for d in char_detections]
                    sam_results = segmenter.segment(image, yolo_dets, char_ids)
                    logger.info(f"SAM segmentou {len(sam_results)} personagens")
            except Exception as e:
                logger.warning(f"SAM 2.1 falhou, usando BBox: {e}")
        
        # 2. Cálculo Z-Buffer
        depth_results = {}
        if self.enable_zbuffer:
            try:
                calculator = self._get_zbuffer_calculator()
                if calculator:
                    from core.detection.yolo_detector import DetectionResult
                    yolo_dets = []
                    for det in char_detections:
                        yd = DetectionResult(
                            bbox=det.bbox,
                            confidence=det.confidence,
                            class_id=det.class_id,
                            class_name=det.class_name
                        )
                        yolo_dets.append(yd)
                    
                    char_ids = [d.char_id for d in char_detections]
                    depth_ordered = calculator.sort_by_depth(yolo_dets, image_size, char_ids)
                    
                    for dr in depth_ordered:
                        depth_results[dr.char_id] = dr
                    
                    logger.info(f"Z-Buffer calculado para {len(depth_results)} personagens")
            except Exception as e:
                logger.warning(f"Z-Buffer falhou: {e}")
        
        # 3. Atualiza detecções com resultados
        for det in char_detections:
            # SAM
            if det.char_id in sam_results:
                seg_result = sam_results[det.char_id]
                det.sam_mask_rle = seg_result.rle_mask
                det.mask_shape = seg_result.mask_shape
            else:
                # Fallback: cria máscara BBox simples
                x1, y1, x2, y2 = det.bbox
                h, w = image.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255
                from .analysis.segmentation import SegmentationResult
                seg_result = SegmentationResult.from_mask(
                    det.char_id, mask, det.bbox
                )
                det.sam_mask_rle = seg_result.rle_mask
                det.mask_shape = seg_result.mask_shape
            
            # Z-Buffer
            if det.char_id in depth_results:
                dr = depth_results[det.char_id]
                det.depth_score = dr.depth_score
                det.depth_rank = dr.rank
        
        return detections
    

    
    def _detect_characters(
        self, 
        image: np.ndarray, 
        page_num: int
    ) -> List[Detection]:
        """
        Detecta personagens na imagem usando YOLO Manga109.
        
        Seguindo a arquitetura Two-Pass e os papers:
        - Body: Usado para IP-Adapter (identidade completa, cores, roupas)
        - Face: Usado para InsightFace (reconhecimento facial)
        
        Args:
            image: Imagem como array numpy (H, W, 3)
            page_num: Número da página
            
        Returns:
            Lista de Detection com body_crop e face_crop
        """
        # YOLO detection com grouping de body/face
        yolo = self._get_yolo_detector()
        
        # YOLO detection - captura TUDO primeiro
        yolo = self._get_yolo_detector()
        all_detections = yolo.detect(image)
        
        # Filtra detecções de Texto (class_id=3)
        text_detections = [d for d in all_detections if d.class_id == DetectionClass.TEXT.value]
        logger.debug(f"{len(text_detections)} balões de texto detectados")
        
        # Agrupa Personagens (Body+Face) reutilizando detections
        character_groups = yolo.group_body_face_pairs(image, detections=all_detections)
        
        logger.info(f"{len(character_groups)} personagens detectados (body+face)")
        
        # Cria objetos Detection para Personagens
        detections = []
        for group in character_groups:
            body_data = group.get('body')
            face_data = group.get('face')
            
            if body_data:
                body_crop, body_det = body_data
                x1, y1, x2, y2 = body_det.bbox
                
                # Context crop do body (para IP-Adapter) - inflado 150%
                body_context_crop = create_context_crop(
                    image, 
                    (x1, y1, x2, y2), 
                    CONTEXT_INFLATION_FACTOR
                )
                
                # Face crop (para InsightFace) - se existir
                face_crop = None

                if face_data:
                    face_crop, face_det = face_data
                
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=body_det.confidence,
                    crop=body_crop,  # Body crop principal
                    context_crop=body_context_crop,  # Body com contexto expandido
                    page_num=page_num,
                    class_id=DetectionClass.BODY.value,
                    class_name="body"
                ))
                
                # Armazena face_crop como atributo extra
                detections[-1].face_crop = face_crop
                detections[-1].has_face = face_data is not None
                
            elif face_data:
                # Apenas face (close-up), sem body
                face_crop, face_det = face_data
                x1, y1, x2, y2 = face_det.bbox
                
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=face_det.confidence,
                    crop=face_crop,  # Usa face como crop principal
                    context_crop=face_crop,  # Sem expansão para faces soltas
                    page_num=page_num,
                    class_id=DetectionClass.FACE.value,
                    class_name="face"
                ))
                detections[-1].face_crop = face_crop
                detections[-1].has_face = True
        
        # Adiciona detecções de Texto
        for text_det in text_detections:
            x1, y1, x2, y2 = text_det.bbox
            
            # Garante coordenadas válidas
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                text_crop = image[y1:y2, x1:x2]
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=text_det.confidence,
                    crop=text_crop,
                    context_crop=text_crop, # Texto não precisa de contexto expandido
                    page_num=page_num,
                    class_id=DetectionClass.TEXT.value,
                    class_name="text"
                ))

        logger.debug(f"{len(detections)} detecções finais (Chars+Text) para extração")
        return detections

    def _extract_lineart(self, image: np.ndarray) -> np.ndarray:
        """
        Extrai lineart da imagem.
        """
        return extract_canny_edges(image, 50, 150)
    
    def _extract_character_data(
        self, 
        detection: Detection,
        full_image: np.ndarray
    ) -> Dict:
        """
        Extrai dados do personagem seguindo a arquitetura Two-Pass.
        
        Estratégia (baseada nos papers):
        - Face Crop → InsightFace (reconhecimento facial fino)
        - Body Crop (context_crop) → IP-Adapter (identidade completa + cores)
        - Body Crop → Extração de paleta (roupas, acessórios)
        
        Args:
            detection: Detecção do personagem (com body/face)
            full_image: Imagem completa da página
            
        Returns:
            Dict com dados do personagem para cache
        """
        identity_encoder = self._get_identity_encoder()
        palette_extractor = self._get_palette_extractor()
        
        # --- PASSO 1: Extrai identidade facial (InsightFace) ---
        face_embedding = None
        face_method = None
        
        if hasattr(detection, 'face_crop') and detection.face_crop is not None:
            # Usa face para InsightFace (melhor reconhecimento facial)
            face_pil = Image.fromarray(detection.face_crop)
            face_embedding, face_method = identity_encoder.extract_identity(
                face_pil,
                timeout=5.0
            )
            logger.debug(f"Face extraída via {face_method}")
        
        # --- PASSO 2: Extrai embedding do corpo (IP-Adapter) ---
        # Usa context_crop (body expandido em 150%) para IP-Adapter
        body_context_pil = Image.fromarray(detection.context_crop)
        
        # Para IP-Adapter, usamos o encoder CLIP do corpo completo
        body_embedding, body_method = identity_encoder.extract_identity(
            body_context_pil,
            timeout=5.0
        )
        logger.debug(f"Body extraído via {body_method}")
        
        # --- PASSO 3: Extrai paleta de cores do corpo ---
        palette = palette_extractor.extract(body_context_pil)
        
        # Retorna ambos os embeddings separadamente
        # Pass 2 usará body_embedding para IP-Adapter
        
        # ADR 004: Inclui dados de segmentação e profundidade
        result = {
            'bbox': detection.bbox,
            'confidence': detection.confidence,
            'has_face': getattr(detection, 'has_face', False),
            # CLIP embedding (consolidado ou body) para FAISS
            'embedding': body_embedding.tolist() if hasattr(body_embedding, 'tolist') else body_embedding,
            'embedding_method': body_method,
            # Embeddings separados para Pass 2
            'body_embedding': body_embedding.tolist() if hasattr(body_embedding, 'tolist') else body_embedding,
            'face_embedding': face_embedding.tolist() if hasattr(face_embedding, 'tolist') else face_embedding,
            'palette': palette,
            'page_num': detection.page_num,
            # ADR 004: Dados de segmentação SAM e profundidade
            'char_id': detection.char_id,
            'sam_mask_rle': detection.sam_mask_rle,
            'mask_shape': detection.mask_shape,
            'depth_score': detection.depth_score,
            'depth_rank': detection.depth_rank,
        }
        
        return result
    
    def _detect_scene_type(self, image: np.ndarray) -> str:
        """
        Detecta tipo de cena (present, flashback, dream, etc).
        
        Args:
            image: Imagem da página
            
        Returns:
            Tipo de cena
        """
        scene_detector = self._get_scene_detector()
        
        # Converte para PIL se for numpy
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        return scene_detector.detect(image)

    def _detect_text_regions(self, image: np.ndarray) -> np.ndarray:
        """
        Detecta regiões de texto (balões, SFX).
        
        Args:
            image: Imagem da página
            
        Returns:
            Máscara de texto (0 = não texto, 255 = texto)
        """
        # Implementação simplificada
        # Em produção, usar modelo específico (PaddleOCR, etc)
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Threshold para detectar regiões brancas (fundo de balões)
        _, white_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        
        # Detecta componentes conectados
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            white_mask, 
            connectivity=8
        )
        
        # Filtra por tamanho (balões típicos)
        text_mask = np.zeros_like(gray)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if 100 < area < 50000:  # Filtro de área
                text_mask[labels == i] = 255
        
        return text_mask

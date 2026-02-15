"""
MangaAutoColor Pro - YOLO Detector (Manga109)

Detector de personagens em mangá usando YOLOv11
treinado no dataset Manga109.

Modelo: deepghs/manga109_yolo
Classes: body, face, frame, text

Arquitetura:
- Body: Corpo completo do personagem (para IP-Adapter)
- Face: Rosto do personagem (para Identity)
- Frame: Quadro/painel de mangá
- Text: Balão de fala/texto
"""

from pathlib import Path

import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import cv2
import os

# Desabilita auto-update do Ultralytics para evitar erros de permissão
os.environ['ULTRALYTICS_AUTOINSTALL'] = '0'
os.environ['YOLO_AUTOINSTALL'] = '0'
os.environ['ULTRALYTICS_OFFLINE'] = '1'  # Modo offline - não tenta instalar nada
from dataclasses import dataclass, field

try:
    from config.settings import (
        DEVICE, YOLO_CONFIDENCE, 
        CONTEXT_INFLATION_FACTOR, VERBOSE
    )
except ImportError:
    # Fallback para execução standalone
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    YOLO_CONFIDENCE = 0.3
    CONTEXT_INFLATION_FACTOR = 1.5
    VERBOSE = True

# Importa utilitários sempre (não dependem de config)
try:
    from core.utils.image_ops import calculate_context_bbox, create_gaussian_mask
except ImportError:
    # Se falhar aqui, define versões dummy ou relança erro
    # Mas como são utils puros, deve funcionar
    print("[AVISO] Não foi possível importar core.utils.image_ops")
    def calculate_context_bbox(bbox, shape, factor): return bbox
    def create_gaussian_mask(bbox, shape, factor): return np.ones(shape)


@dataclass
class DetectionResult:
    """Resultado de uma detecção de elemento em mangá"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str  # "body", "face", "frame", "text"
    detection_type: str = "character"  # "body", "face", "panel", "text"
    prominence_score: float = 0.0
    context_bbox: Optional[Tuple[int, int, int, int]] = None


class YOLODetector:
    """
    Detector YOLOv11 para elementos de mangá (Manga109 dataset).
    
    Modelo: deepghs/manga109_yolo
    
    Classes detectadas:
    - body: Corpo/personagem completo (ideal para IP-Adapter)
    - face: Rosto do personagem (ideal para Identity)
    - frame: Quadro/painel da página
    - text: Balão de fala/texto
    
    Raises:
        RuntimeError: Se o modelo não puder ser carregado
    """
    
    # Mapeamento de classes do modelo Manga109
    CLASS_NAMES = {
        0: "body",
        1: "face", 
        2: "frame",
        3: "text"
    }
    
    # Classes que representam personagens (para colorização)
    CHARACTER_CLASSES = {0, 1}  # body, face
    
    def __init__(
        self,
        model_path: str = "./data/models/manga109_yolo.onnx",
        device: str = DEVICE,
        conf_threshold: float = YOLO_CONFIDENCE
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.model = None
        self.class_map = {}  # Será populado após carregar modelo
        
        # Tenta carregar o modelo
        self._load_model()
        
        # Introspecção de classes
        self._introspect_classes()
        
        if VERBOSE:
            print(f"[YOLODetector] Inicializado (device={device}, conf_threshold={conf_threshold})")
    
    def _load_model(self):
        """
        Carrega o modelo YOLO Manga109.
        
        Suporta .onnx e .pt (PyTorch).
        
        Raises:
            RuntimeError: Se o modelo não for encontrado ou falhar ao carregar
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError(
                "Ultralytics não instalado. "
                "Execute: pip install ultralytics"
            )
        
        if not self.model_path.exists():
            raise RuntimeError(
                f"Modelo não encontrado: {self.model_path}\n\n"
                f"Baixe o modelo com:\n"
                f"  python scripts/download_models.py --models yolo_manga"
            )
        
        print(f"[YOLODetector] Carregando modelo: {self.model_path}")
        
        try:
            self.model = YOLO(str(self.model_path), task='detect')
            print(f"[YOLODetector] Modelo carregado com sucesso!")
        except Exception as e:
            raise RuntimeError(f"Falha ao carregar modelo: {e}")
    
    def _introspect_classes(self):
        """
        Introspecção das classes do modelo.
        Popula self.class_map com os nomes das classes.
        """
        if self.model is None:
            return
        
        try:
            # Tenta obter nomes das classes do modelo
            if hasattr(self.model, 'names'):
                names = self.model.names
                print(f"[YOLODetector] Classes do modelo:")
                for idx, name in names.items():
                    self.class_map[int(idx)] = str(name)
                    print(f"  [{idx}] {name}")
            else:
                # Fallback para mapeamento padrão Manga109
                self.class_map = self.CLASS_NAMES.copy()
                print(f"[YOLODetector] Usando mapeamento padrão Manga109:")
                for idx, name in self.class_map.items():
                    print(f"  [{idx}] {name}")
                    
        except Exception as e:
            print(f"[AVISO] Não foi possível introspectar classes: {e}")
            self.class_map = self.CLASS_NAMES.copy()
    
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Detecta elementos em uma imagem de mangá.
        
        Args:
            image: Imagem numpy array (BGR ou RGB)
            
        Returns:
            Lista de DetectionResult com body, face, frame, text
            
        Raises:
            RuntimeError: Se o modelo não estiver carregado
            ValueError: Se a imagem for inválida
        """
        if self.model is None:
            raise RuntimeError("Modelo YOLO não carregado. Reinicie o detector.")
        
        # Valida imagem
        if image is None or image.size == 0:
            raise ValueError("Imagem vazia ou inválida")
        
        # DEBUG: Log do range da imagem
        img_min = np.min(image)
        img_max = np.max(image)
        img_mean = np.mean(image)
        print(f"[DEBUG] YOLO Input Range: min={img_min}, max={img_max}, mean={img_mean:.2f}, shape={image.shape}")
        
        # Verifica se imagem está normalizada errada (0-1 em vez de 0-255)
        if img_max <= 1.0 and image.dtype != np.uint8:
            print(f"[DEBUG] Imagem parece estar normalizada (0-1). Convertendo para 0-255...")
            image = (image * 255).astype(np.uint8)
        
        # Garante que é uint8
        if image.dtype != np.uint8:
            print(f"[DEBUG] Convertendo dtype de {image.dtype} para uint8")
            image = image.astype(np.uint8)
        
        # Converte BGR para RGB se necessário (OpenCV carrega como BGR)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Executa detecção
        results = self.model(
            image_rgb,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )
        
        # Extrai detecções
        detections = []
        body_count = 0
        face_count = 0
        frame_count = 0
        text_count = 0
        
        for result in results:
            if result.boxes is None:
                continue
            
            for box in result.boxes:
                # Coordenadas
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Confiança
                conf = float(box.conf[0].cpu().numpy())
                
                # Classe
                cls = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0
                class_name = self.class_map.get(cls, f"class_{cls}")
                
                # Determina tipo de detecção
                if cls == 0:  # body
                    det_type = "body"
                    body_count += 1
                elif cls == 1:  # face
                    det_type = "face"
                    face_count += 1
                elif cls == 2:  # frame
                    det_type = "panel"
                    frame_count += 1
                elif cls == 3:  # text
                    det_type = "text"
                    text_count += 1
                else:
                    det_type = "unknown"
                
                detection = DetectionResult(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls,
                    class_name=class_name,
                    detection_type=det_type
                )
                
                # Calcula prominence (apenas para personagens)
                if cls in self.CHARACTER_CLASSES:
                    detection.prominence_score = self._calculate_prominence(
                        detection.bbox, image.shape[1], image.shape[0]
                    )
                    # Calcula bbox com contexto (inflado)
                    detection.context_bbox = calculate_context_bbox(
                        detection.bbox, 
                        (image.shape[1], image.shape[0]),
                        CONTEXT_INFLATION_FACTOR
                    )
                
                detections.append(detection)
        
        # Ordena por prominence (mais importantes primeiro)
        character_detections = [d for d in detections if d.class_id in self.CHARACTER_CLASSES]
        character_detections.sort(key=lambda x: x.prominence_score, reverse=True)
        
        # Reordena: body primeiro, depois face
        other_detections = [d for d in detections if d.class_id not in self.CHARACTER_CLASSES]
        detections = character_detections + other_detections
        
        print(f"[YOLODetector] Detecções encontradas:")
        print(f"  - Body (corpo): {body_count}")
        print(f"  - Face (rosto): {face_count}")
        print(f"  - Frame (quadro): {frame_count}")
        print(f"  - Text (texto): {text_count}")
        
        # Log das detecções de personagens (body/face)
        char_dets = [d for d in detections if d.class_id in self.CHARACTER_CLASSES][:5]
        for i, det in enumerate(char_dets):
            print(f"  [{i+1}] {det.class_name}: bbox={det.bbox}, conf={det.confidence:.2f}, prominence={det.prominence_score:.2f}")
        
        return detections
    
    def get_character_crops(
        self, 
        image: np.ndarray,
        detections: Optional[List[DetectionResult]] = None
    ) -> Dict[str, List[Tuple[np.ndarray, DetectionResult]]]:
        """
        Extrai crops separados para body e face.
        
        Seguindo os princípios de Differential Diffusion e IP-Adapter:
        - Body: Usado para IP-Adapter (identidade completa, melhor para cores/roupas)
        - Face: Usado para InsightFace/Identity (reconhecimento facial fino)
        
        Args:
            image: Imagem numpy array (RGB)
            detections: Lista opcional de detecções pré-calculadas
            
        Returns:
            Dict com 'bodies' e 'faces', cada um com lista de (crop, detection)
        """
        if detections is None:
            detections = self.detect(image)
        
        bodies = []
        faces = []
        
        for det in detections:
            if det.class_id == 0:  # body
                x1, y1, x2, y2 = det.context_bbox or det.bbox
                # Garante coordenadas válidas
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    crop = image[y1:y2, x1:x2]
                    bodies.append((crop, det))
            elif det.class_id == 1:  # face
                x1, y1, x2, y2 = det.bbox  # Face usa bbox original (mais preciso)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    crop = image[y1:y2, x1:x2]
                    faces.append((crop, det))
        
        # Ordena por prominence/confiança
        bodies.sort(key=lambda x: x[1].prominence_score, reverse=True)
        faces.sort(key=lambda x: x[1].confidence, reverse=True)
        
        return {'bodies': bodies, 'faces': faces}
    
    def get_best_character_crop(self, image: np.ndarray, prefer_face: bool = False) -> Optional[Tuple[np.ndarray, DetectionResult]]:
        """
        Retorna o melhor crop de personagem (método legado).
        
        Args:
            image: Imagem numpy array
            prefer_face: Se True, prioriza faces
            
        Returns:
            Tuple (crop, detection) ou None
        """
        crops = self.get_character_crops(image)
        
        if prefer_face and crops['faces']:
            return crops['faces'][0]
        
        if crops['bodies']:
            return crops['bodies'][0]
        
        if crops['faces']:
            return crops['faces'][0]
        
        return None
    
    def group_body_face_pairs(
        self, 
        image: np.ndarray,
        detections: Optional[List[DetectionResult]] = None
    ) -> List[Dict]:
        """
        Agrupa bodies e faces que pertencem ao mesmo personagem.
        
        Heurística: Se uma face está dentro de um body (ou próxima), 
        provavelmente pertencem ao mesmo personagem.
        
        Args:
            image: Imagem numpy array
            detections: Lista opcional de detecções pré-calculadas
            
        Returns:
            Lista de grupos {'body': (crop, det), 'face': (crop, det) ou None}
        """
        crops = self.get_character_crops(image, detections=detections)
        bodies = crops['bodies']
        faces = crops['faces']
        
        groups = []
        used_faces = set()
        
        for body_crop, body_det in bodies:
            bx1, by1, bx2, by2 = body_det.bbox
            
            # Procura face dentro deste body
            matching_face = None
            for i, (face_crop, face_det) in enumerate(faces):
                if i in used_faces:
                    continue
                    
                fx1, fy1, fx2, fy2 = face_det.bbox
                
                # Heurística 1: face está dentro do body?
                if (bx1 <= fx1 <= fx2 <= bx2) and (by1 <= fy1 <= fy2 <= by2):
                    matching_face = (face_crop, face_det)
                    used_faces.add(i)
                    break
                
                # Heurística 2: face está próxima ao topo do body (cabeça)
                # Dentro de 50px acima do body, com overlap horizontal
                if (abs(fy2 - by1) < 50 and 
                    max(bx1, fx1) < min(bx2, fx2)):  # Overlap horizontal
                    matching_face = (face_crop, face_det)
                    used_faces.add(i)
                    break
            
            groups.append({
                'body': (body_crop, body_det),
                'face': matching_face,
                'bbox': body_det.bbox,
                'prominence': body_det.prominence_score
            })
        
        # Adiciona faces órfãs (sem body) como personagens individuais
        for i, (face_crop, face_det) in enumerate(faces):
            if i not in used_faces:
                groups.append({
                    'body': None,
                    'face': (face_crop, face_det),
                    'bbox': face_det.bbox,
                    'prominence': face_det.confidence * 0.5  # Faces soltas têm menos prominence
                })
        
        # Ordena por prominence
        groups.sort(key=lambda x: x['prominence'], reverse=True)
        
        return groups
    
    def _calculate_prominence(
        self,
        bbox: Tuple[int, int, int, int],
        img_width: int,
        img_height: int
    ) -> float:
        """
        Calcula prominence score baseado em área e centralidade.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            img_width: Largura da imagem
            img_height: Altura da imagem
            
        Returns:
            Score de prominence (0-1)
        """
        x1, y1, x2, y2 = bbox
        
        # Área
        area = (x2 - x1) * (y2 - y1)
        img_area = img_width * img_height
        area_ratio = area / img_area
        
        # Centralidade (quanto mais central, maior)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        img_center_x = img_width / 2
        img_center_y = img_height / 2
        
        # Distância do centro (normalizada)
        dist_x = abs(center_x - img_center_x) / (img_width / 2)
        dist_y = abs(center_y - img_center_y) / (img_height / 2)
        centrality = 1.0 - (dist_x + dist_y) / 2
        
        # Prominence = área * centralidade
        prominence = area_ratio * centrality
        
        return min(prominence, 1.0)
    
    # inflate_bbox REMOVED (replaced by calculate_context_bbox)
    # create_gaussian_mask REMOVED (imported from image_ops)


# =============================================================================
# TESTE DE VALIDAÇÃO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("YOLODetector - Teste de Validação (Manga109)")
    print("=" * 60)
    
    # Cria imagem dummy de mangá (simula página em preto-e-branco)
    dummy = np.ones((512, 512, 3), dtype=np.uint8) * 255  # Fundo branco
    
    # Desenha alguns "personagens" (retângulos pretos simulando figuras)
    cv2.rectangle(dummy, (100, 100), (200, 300), (0, 0, 0), -1)  # Body-like
    cv2.rectangle(dummy, (130, 120), (170, 180), (50, 50, 50), -1)  # Face-like
    cv2.rectangle(dummy, (350, 150), (450, 400), (0, 0, 0), -1)  # Body 2
    cv2.rectangle(dummy, (300, 50), (500, 450), (200, 200, 200), 2)  # Frame
    
    # Salva imagem dummy para inspeção
    dummy_path = Path("./data/dummy_manga_test.png")
    dummy_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dummy_path), dummy)
    print(f"\nImagem dummy criada: {dummy_path}")
    
    # Testa detector
    print("\n[TESTE] Inicializando detector...")
    try:
        detector = YOLODetector()
        
        print("\n[TESTE] Executando detecção na imagem dummy...")
        detections = detector.detect(dummy)
        
        print("\n" + "=" * 60)
        print("RESULTADO DO TESTE:")
        print("=" * 60)
        
        if detections:
            for det in detections:
                print(f"  - {det.class_name.upper()} (conf: {det.confidence:.2f}): bbox={det.bbox}")
        else:
            print("  Nenhuma detecção na imagem dummy (esperado - é apenas um teste de forma)")
        
        # Testa get_best_character_crop
        print("\n[TESTE] Testando get_best_character_crop...")
        result = detector.get_best_character_crop(dummy)
        if result:
            crop, det = result
            print(f"  Melhor crop: {det.class_name} (shape: {crop.shape})")
        else:
            print("  Nenhum personagem encontrado (esperado na imagem dummy)")
        
        print("\n" + "=" * 60)
        print("[SUCESSO] Detector inicializado corretamente!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERRO] Falha no teste: {e}")
        import traceback
        traceback.print_exc()

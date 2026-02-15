"""
MangaAutoColor Pro - Test Utilities

Helpers para criação de dados de teste e utilitários de verificação.
Todos os dados são sintéticos (não usam imagens reais).
"""

import hashlib
import io
from typing import Tuple, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw


def make_dummy_page(size: Tuple[int, int] = (1024, 1024), seed: int = 42) -> Image.Image:
    """
    Cria uma imagem dummy sintética para testes.
    
    Gera uma imagem com formas geométricas simples (não é imagem real de personagem).
    
    Args:
        size: Tupla (width, height) da imagem
        seed: Seed para determinismo
        
    Returns:
        PIL Image em modo RGB
    """
    rng = np.random.RandomState(seed)
    width, height = size
    
    # Cria imagem base (fundo cinza claro)
    img = Image.new('RGB', size, (240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Desenha formas geométricas aleatórias (simula linhas de mangá)
    n_lines = rng.randint(10, 30)
    for _ in range(n_lines):
        x1 = rng.randint(0, width)
        y1 = rng.randint(0, height)
        x2 = rng.randint(0, width)
        y2 = rng.randint(0, height)
        draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 0), width=rng.randint(1, 4))
    
    # Adiciona alguns círculos (simula caracteres)
    n_circles = rng.randint(3, 8)
    for _ in range(n_circles):
        cx = rng.randint(100, width - 100)
        cy = rng.randint(100, height - 100)
        r = rng.randint(30, 100)
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], 
                     outline=(0, 0, 0), width=rng.randint(2, 5))
    
    return img


def make_dummy_canny(size: Tuple[int, int] = (1024, 1024), seed: int = 42) -> np.ndarray:
    """
    Cria um mapa de bordas Canny sintético.
    
    Args:
        size: Tupla (width, height)
        seed: Seed para determinismo
        
    Returns:
        Array numpy (H, W) uint8 com valores 0-255
    """
    rng = np.random.RandomState(seed)
    width, height = size
    
    # Cria imagem preta
    canny = np.zeros((height, width), dtype=np.uint8)
    
    # Adiciona linhas aleatórias (bordas)
    n_lines = rng.randint(15, 40)
    for _ in range(n_lines):
        x1 = rng.randint(0, width)
        y1 = rng.randint(0, height)
        x2 = rng.randint(0, width)
        y2 = rng.randint(0, height)
        
        # Desenha linha simples
        length = int(np.sqrt((x2-x1)**2 + (y2-y1)**2))
        if length > 0:
            for t in np.linspace(0, 1, length):
                x = int(x1 + t * (x2 - x1))
                y = int(y1 + t * (y2 - y1))
                if 0 <= x < width and 0 <= y < height:
                    canny[y, x] = 255
    
    return canny


def make_dummy_embedding(dim: int = 768, seed: int = 42) -> torch.Tensor:
    """
    Cria um embedding dummy normalizado.
    
    Args:
        dim: Dimensão do embedding (padrão CLIP: 768)
        seed: Seed para determinismo
        
    Returns:
        Tensor torch (1, dim) normalizado
    """
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    # Normaliza
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return torch.from_numpy(vec).unsqueeze(0)


def make_dummy_bbox(
    image_size: Tuple[int, int] = (1024, 1024),
    seed: int = 42
) -> Tuple[int, int, int, int]:
    """
    Cria um bounding box dummy válido.
    
    Args:
        image_size: (width, height) da imagem
        seed: Seed para determinismo
        
    Returns:
        Tupla (x1, y1, x2, y2)
    """
    rng = np.random.RandomState(seed)
    width, height = image_size
    
    # Garante bbox válido (pelo menos 100x100)
    w = rng.randint(100, min(400, width // 2))
    h = rng.randint(100, min(400, height // 2))
    x1 = rng.randint(0, width - w)
    y1 = rng.randint(0, height - h)
    x2 = x1 + w
    y2 = y1 + h
    
    return (x1, y1, x2, y2)


def img_hash(pil_image: Image.Image) -> str:
    """
    Calcula SHA256 de uma imagem PIL.
    
    Args:
        pil_image: Imagem PIL
        
    Returns:
        String hexadecimal do hash SHA256
    """
    # Salva em buffer PNG (formato determinístico)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    return hashlib.sha256(buffer.read()).hexdigest()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Calcula similaridade cosseno entre dois tensores.
    
    Args:
        a: Tensor 1D ou 2D
        b: Tensor 1D ou 2D
        
    Returns:
        Similaridade cosseno (float entre -1 e 1)
    """
    a_flat = a.flatten()
    b_flat = b.flatten()
    
    dot = torch.dot(a_flat, b_flat).item()
    norm_a = torch.norm(a_flat).item()
    norm_b = torch.norm(b_flat).item()
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot / (norm_a * norm_b)


def calculate_prominence(
    bbox: Tuple[int, int, int, int],
    image_size: Tuple[int, int]
) -> float:
    """
    Calcula score de prominence para um bbox.
    
    Delega para a implementação central em config.settings para evitar duplicação.
    prominence = area * centrality
    
    Args:
        bbox: (x1, y1, x2, y2)
        image_size: (width, height)
        
    Returns:
        Score de prominence
    """
    # Import local para evitar circularidade
    from config.settings import calculate_prominence as _calc_prom
    return _calc_prom(bbox, image_size)


def create_test_character_detections(
    n_characters: int = 5,
    image_size: Tuple[int, int] = (1024, 1024),
    seed: int = 42
) -> list:
    """
    Cria uma lista de detecções de personagens fake.
    
    Args:
        n_characters: Número de personagens
        image_size: Tamanho da imagem
        seed: Seed para determinismo
        
    Returns:
        Lista de dicts com 'bbox', 'char_id', 'prominence_score'
    """
    rng = np.random.RandomState(seed)
    detections = []
    
    for i in range(n_characters):
        bbox = make_dummy_bbox(image_size, seed=seed + i * 100)
        prominence = calculate_prominence(bbox, image_size)
        
        detections.append({
            'char_id': f'char_{i:03d}',
            'bbox': bbox,
            'prominence_score': prominence,
            'confidence': rng.uniform(0.7, 0.99)
        })
    
    return detections


def select_top_k(
    detections: list,
    k: int = 2
) -> list:
    """
    Seleciona os Top-K personagens por prominence.
    
    Args:
        detections: Lista de detecções
        k: Número máximo a selecionar
        
    Returns:
        Lista dos Top-K
    """
    sorted_dets = sorted(
        detections,
        key=lambda x: x.get('prominence_score', 0.0),
        reverse=True
    )
    return sorted_dets[:k]


def create_gaussian_mask(
    shape: Tuple[int, int],
    center: Optional[Tuple[float, float]] = None,
    sigma: Optional[float] = None
) -> np.ndarray:
    """
    Cria uma máscara gaussiana 2D.
    
    Args:
        shape: (height, width)
        center: (cx, cy) - padrão: centro da imagem
        sigma: Desvio padrão - padrão: baseado no tamanho
        
    Returns:
        Array numpy [0, 1] float32
    """
    h, w = shape
    
    if center is None:
        center = (w / 2, h / 2)
    if sigma is None:
        sigma = min(w, h) / 4
    
    y, x = np.ogrid[:h, :w]
    cx, cy = center
    
    dist_sq = ((x - cx) / sigma)**2 + ((y - cy) / sigma)**2
    mask = np.exp(-dist_sq / 2)
    
    return mask.astype(np.float32)


def get_ip_adapter_scale_at_step(
    step: int,
    total_steps: int,
    end_frac: float = 0.6,
    max_scale: float = 0.6
) -> float:
    """
    Calcula a escala do IP-Adapter para um step específico (Temporal Decay).
    
    Args:
        step: Step atual (0-based)
        total_steps: Número total de steps
        end_frac: Fração onde IP-Adapter desliga (padrão 0.6)
        max_scale: Escala máxima
        
    Returns:
        Escala do IP-Adapter para este step
    """
    progress = step / total_steps if total_steps > 0 else 0
    
    if progress >= end_frac:
        return 0.0
    
    # Decaimento linear
    decay_factor = 1.0 - (progress / end_frac)
    return max_scale * decay_factor

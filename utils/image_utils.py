"""
MangaAutoColor Pro - Utilitários de Processamento de Imagem

Funções auxiliares para:
- Carregamento e pré-processamento de imagens
- Conversão entre espaços de cor
- Manipulação de tiles
- Pré-processamento específico para mangá
"""

import torch
import numpy as np
from PIL import Image, ImageOps
from typing import Tuple, Optional, Union, List
import cv2
from pathlib import Path

from config.settings import MAX_RESOLUTION, TILE_SIZE, VERBOSE


def load_image(
    path: Union[str, Path],
    max_resolution: int = MAX_RESOLUTION,
    convert_rgb: bool = True
) -> Image.Image:
    """
    Carrega imagem com redimensionamento se necessário.
    
    GARANTE RGB: Converte qualquer modo (L, P, RGBA, etc) para RGB.
    
    Args:
        path: Caminho da imagem
        max_resolution: Resolução máxima (maior dimensão)
        convert_rgb: Converter para RGB (sempre True para mangá)
        
    Returns:
        Imagem PIL em modo RGB
    """
    image = Image.open(path)
    
    if convert_rgb:
        # Força conversão para RGB independente do modo original
        # Isso garante que mangás grayscale (L) sejam convertidos
        if image.mode != 'RGB':
            if image.mode == 'RGBA':
                # Cria fundo branco para transparente
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            elif image.mode == 'L':
                # Grayscale - converte direto para RGB
                image = image.convert('RGB')
            elif image.mode == 'P':
                # Palette - converte via RGBA para preservar transparência
                image = image.convert('RGBA')
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            elif image.mode == 'LA':
                # Grayscale com alpha
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[1])
                image = background
            else:
                # Qualquer outro modo, converte direto
                image = image.convert('RGB')
            
            if VERBOSE:
                print(f"[ImageUtils] Convertido {path}: modo original -> RGB")
    
    # Redimensiona se necessário
    w, h = image.size
    max_dim = max(w, h)
    
    if max_dim > max_resolution:
        scale = max_resolution / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        
        if VERBOSE:
            print(f"[ImageUtils] Redimensionado: {w}x{h} -> {new_w}x{new_h}")
    
    return image


def save_image(
    image: Union[Image.Image, np.ndarray],
    path: Union[str, Path],
    quality: int = 95
):
    """
    Salva imagem em disco.
    
    Args:
        image: Imagem PIL ou numpy array
        path: Caminho de destino
        quality: Qualidade JPEG (se aplicável)
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.suffix.lower() in ['.jpg', '.jpeg']:
        image.save(path, quality=quality)
    else:
        image.save(path)


def resize_keep_aspect(
    image: Union[Image.Image, np.ndarray],
    target_size: Tuple[int, int],
    padding_color: Tuple[int, int, int] = (255, 255, 255)
) -> Tuple[Union[Image.Image, np.ndarray], Tuple[int, int, int, int]]:
    """
    Redimensiona mantendo aspect ratio com padding se necessário.
    
    Args:
        image: Imagem
        target_size: (largura, altura) desejada
        padding_color: Cor do padding
        
    Returns:
        Tupla de (imagem redimensionada, bbox da imagem original)
    """
    is_numpy = isinstance(image, np.ndarray)
    
    if is_numpy:
        image = Image.fromarray(image)
    
    target_w, target_h = target_size
    img_w, img_h = image.size
    
    # Calcula escala mantendo aspecto
    scale = min(target_w / img_w, target_h / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    
    # Redimensiona
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    
    # Cria imagem com padding
    result = Image.new('RGB', target_size, padding_color)
    
    # Centraliza
    x1 = (target_w - new_w) // 2
    y1 = (target_h - new_h) // 2
    result.paste(resized, (x1, y1))
    
    bbox = (x1, y1, x1 + new_w, y1 + new_h)
    
    if is_numpy:
        result = np.array(result)
    
    return result, bbox


def crop_with_padding(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: int = 0,
    padding_color: Tuple[int, int, int] = (255, 255, 255)
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop com padding automático se bbox ultrapassar limites.
    
    Args:
        image: Imagem numpy (H, W, C)
        bbox: (x1, y1, x2, y2)
        padding: Padding adicional
        padding_color: Cor do padding
        
    Returns:
        Tupla de (crop, bbox_efetivo)
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Aplica padding ao bbox
    x1 -= padding
    y1 -= padding
    x2 += padding
    y2 += padding
    
    # Calcula padding necessário
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)
    
    # Clampa bbox
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    # Crop
    crop = image[y1:y2, x1:x2].copy()
    
    # Aplica padding se necessário
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        if len(crop.shape) == 3:
            crop = cv2.copyMakeBorder(
                crop, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=padding_color
            )
        else:
            crop = cv2.copyMakeBorder(
                crop, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=0
            )
    
    return crop, (x1, y1, x2, y2)


def normalize_image(
    image: np.ndarray,
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None
) -> np.ndarray:
    """
    Normaliza imagem para [-1, 1] ou com mean/std específicos.
    
    Args:
        image: Imagem em [0, 255]
        mean: Média para normalização
        std: Desvio padrão
        
    Returns:
        Imagem normalizada
    """
    image = image.astype(np.float32) / 255.0
    
    if mean is not None and std is not None:
        image = (image - np.array(mean)) / np.array(std)
    else:
        image = image * 2 - 1  # [0, 1] -> [-1, 1]
    
    return image


def denormalize_image(
    image: np.ndarray,
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None
) -> np.ndarray:
    """
    Denormaliza imagem de [-1, 1] para [0, 255].
    
    Args:
        image: Imagem normalizada
        mean: Média usada na normalização
        std: Desvio padrão usado
        
    Returns:
        Imagem em [0, 255]
    """
    if mean is not None and std is not None:
        image = image * np.array(std) + np.array(mean)
    else:
        image = (image + 1) / 2  # [-1, 1] -> [0, 1]
    
    return np.clip(image * 255, 0, 255).astype(np.uint8)


def pil_to_tensor(
    image: Image.Image,
    normalize: bool = True
) -> torch.Tensor:
    """
    Converte PIL Image para tensor torch.
    
    Args:
        image: Imagem PIL
        normalize: Normalizar para [-1, 1]
        
    Returns:
        Tensor (1, C, H, W)
    """
    image_np = np.array(image).astype(np.float32) / 255.0
    
    if normalize:
        image_np = image_np * 2 - 1
    
    # HWC -> CHW
    image_np = np.transpose(image_np, (2, 0, 1))
    
    return torch.from_numpy(image_np).unsqueeze(0)


def tensor_to_pil(
    tensor: torch.Tensor,
    denormalize: bool = True
) -> Image.Image:
    """
    Converte tensor torch para PIL Image.
    
    Args:
        tensor: Tensor (B, C, H, W) ou (C, H, W)
        denormalize: Denormalizar de [-1, 1]
        
    Returns:
        Imagem PIL
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # CHW -> HWC
    image_np = tensor.cpu().numpy().transpose(1, 2, 0)
    
    if denormalize:
        image_np = (image_np + 1) / 2
    
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(image_np)


def create_tile_grid(
    image_size: Tuple[int, int],
    tile_size: int = TILE_SIZE,
    overlap: int = 0
) -> List[Tuple[int, int, int, int]]:
    """
    Cria grid de tiles para uma imagem.
    
    Args:
        image_size: (largura, altura) da imagem
        tile_size: Tamanho do tile
        overlap: Sobreposição entre tiles
        
    Returns:
        Lista de bboxes (x1, y1, x2, y2)
    """
    width, height = image_size
    stride = tile_size - overlap
    
    tiles = []
    
    # Calcula posições iniciais
    if width <= tile_size:
        x_starts = [0]
    else:
        x_starts = list(range(0, width - tile_size + 1, stride))
        # Garante que o último tile cubra a borda
        if x_starts[-1] + tile_size < width:
            x_starts.append(width - tile_size)
    
    if height <= tile_size:
        y_starts = [0]
    else:
        y_starts = list(range(0, height - tile_size + 1, stride))
        if y_starts[-1] + tile_size < height:
            y_starts.append(height - tile_size)
    
    for y in y_starts:
        for x in x_starts:
            x2 = min(x + tile_size, width)
            y2 = min(y + tile_size, height)
            tiles.append((x, y, x2, y2))
    
    return tiles


def extract_tiles(
    image: np.ndarray,
    tile_size: int = TILE_SIZE,
    overlap: int = 0
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Extrai tiles de uma imagem.
    
    Args:
        image: Imagem numpy
        tile_size: Tamanho do tile
        overlap: Sobreposição
        
    Returns:
        Lista de (tile, bbox)
    """
    h, w = image.shape[:2]
    tiles_bboxes = create_tile_grid((w, h), tile_size, overlap)
    
    tiles = []
    for x1, y1, x2, y2 in tiles_bboxes:
        tile = image[y1:y2, x1:x2].copy()
        tiles.append((tile, (x1, y1, x2, y2)))
    
    return tiles


def merge_tiles(
    tiles: List[Tuple[np.ndarray, Tuple[int, int, int, int]]],
    image_size: Tuple[int, int],
    overlap: int = 0
) -> np.ndarray:
    """
    Mescla tiles em uma imagem completa.
    
    Args:
        tiles: Lista de (tile, bbox)
        image_size: (largura, altura) da imagem final
        overlap: Sobreposição usada na extração
        
    Returns:
        Imagem mesclada
    """
    width, height = image_size
    
    # Acumuladores
    result = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    
    for tile, (x1, y1, x2, y2) in tiles:
        h, w = tile.shape[:2]
        
        # Cria pesos para blending (mais alto no centro)
        weight = np.ones((h, w), dtype=np.float32)
        
        if overlap > 0:
            # Fade nas bordas que têm overlap
            fade = min(overlap // 2, w // 4, h // 4)
            
            # Bordas esquerda/direita
            for i in range(fade):
                weight[:, i] *= (i / fade)
                weight[:, w-1-i] *= (i / fade)
            
            # Bordas superior/inferior
            for i in range(fade):
                weight[i, :] *= (i / fade)
                weight[h-1-i, :] *= (i / fade)
        
        # Acumula
        result[y1:y1+h, x1:x1+w] += tile.astype(np.float32) * weight[:, :, np.newaxis]
        weights[y1:y1+h, x1:x1+w] += weight
    
    # Normaliza
    weights = np.maximum(weights, 1e-8)
    result = result / weights[:, :, np.newaxis]
    
    return np.clip(result, 0, 255).astype(np.uint8)


def enhance_contrast(
    image: np.ndarray,
    method: str = "clahe"
) -> np.ndarray:
    """
    Aumenta contraste da imagem.
    
    Args:
        image: Imagem numpy
        method: "clahe", "histogram", "auto"
        
    Returns:
        Imagem com contraste aumentado
    """
    if method == "clahe":
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    elif method == "histogram":
        # Equalização de histograma
        img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    
    else:  # auto
        # Ajusta contraste baseado na variância
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        variance = np.var(gray)
        
        if variance < 1000:
            return enhance_contrast(image, "clahe")
        return image


def remove_noise(
    image: np.ndarray,
    strength: int = 10
) -> np.ndarray:
    """
    Remove ruído da imagem preservando bordas.
    
    Args:
        image: Imagem numpy
        strength: Força da remoção
        
    Returns:
        Imagem filtrada
    """
    return cv2.fastNlMeansDenoisingColored(
        image, None, strength, strength, 7, 21
    )


def detect_and_crop_page(
    image: np.ndarray,
    padding: int = 10
) -> np.ndarray:
    """
    Detecta e corta região da página removendo bordas vazias.
    
    Args:
        image: Imagem numpy
        padding: Padding ao redor da página
        
    Returns:
        Imagem cortada
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Threshold para encontrar conteúdo
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Encontra contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    # Maior contorno (página)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    # Aplica padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)
    
    return image[y:y+h, x:x+w]


def compute_image_hash(
    image: Union[Image.Image, np.ndarray],
    hash_size: int = 16
) -> str:
    """
    Computa hash perceptual da imagem.
    
    Args:
        image: Imagem
        hash_size: Tamanho do hash
        
    Returns:
        String hex do hash
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Reduz e converte para grayscale
    small = cv2.resize(image, (hash_size, hash_size))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    
    # Calcula hash (diferença entre pixels adjacentes)
    diff = gray[:, 1:] > gray[:, :-1]
    
    # Converte para hex
    return ''.join(str(int(b)) for b in diff.flatten())


def calculate_psnr(
    img1: np.ndarray,
    img2: np.ndarray
) -> float:
    """
    Calcula PSNR entre duas imagens.
    
    Args:
        img1: Primeira imagem
        img2: Segunda imagem
        
    Returns:
        PSNR em dB
    """
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr


def calculate_ssim(
    img1: np.ndarray,
    img2: np.ndarray
) -> float:
    """
    Calcula SSIM entre duas imagens.
    
    Args:
        img1: Primeira imagem
        img2: Segunda imagem
        
    Returns:
        SSIM (0 a 1)
    """
    try:
        from skimage.metrics import structural_similarity as ssim
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        return ssim(gray1, gray2)
    except ImportError:
        # Fallback simples
        return 1.0 - np.mean(np.abs(img1.astype(float) - img2.astype(float))) / 255.0

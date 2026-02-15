"""
MangaAutoColor Pro - Configurações do Sistema
Configurações centralizadas para todo o pipeline Two-Pass com Differential Diffusion

Baseado em:
- Differential Diffusion: Change Map para controle pixel-a-pixel
- Blended Latent Diffusion: Máscaras suaves e blending latente
- SDXL-Lightning: Geração rápida em 4 steps
"""

import torch
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple


# ============================================================================
# HARDWARE E PERFORMANCE (Otimizado para RTX 3060 12GB)
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Otimizações de memória RTX 3060
ENABLE_CPU_OFFLOAD = True          # Descarrega para CPU quando não em uso
ENABLE_VAE_SLICING = True          # Slicing do VAE para economia de VRAM
ENABLE_VAE_TILING = False           # Tiling para imagens grandes (Desativado: causa artefatos de borda)
PCTC_ENABLED = False               # Semantic Correspondence & Temporal Consistency (v2.7)
PCTC_POINT_ENABLED = False         # Point matching specific flag
ENABLE_XFORMERS = DEVICE == "cuda"  # Otimização de atenção

# Face Analysis (InsightFace)
INSIGHTFACE_MODEL = "buffalo_l"
FACE_DETECTION_SIZE = (640, 640)

# YOLO / Detecção
YOLO_MODEL_ID = "deepghs/manga109_yolo"
YOLO_CONFIDENCE = 0.3              # Confiança mínima para detecção
DETECTION_IOU_THRESHOLD = 0.45     # IOU threshold para NMS
CONTEXT_INFLATION_FACTOR = 1.5     # Fator de expansão para contexto (IP-Adapter)

# Limites de resolução SDXL
MAX_RESOLUTION = 2048              # Máximo para SDXL
TILE_SIZE = int(os.environ.get('TILE_SIZE', 1024))                   # Tamanho do tile (SDXL nativo)
TILE_OVERLAP = int(os.environ.get('TILE_OVERLAP', 256))              # Overlap entre tiles

# VRAM Management
MAX_VRAM_USAGE_GB = 10.0           # Limite de segurança para RTX 3060 (12GB)
OFFLOAD_TO_CPU = True              # Sempre descarregar quando possível


# ============================================================================
# ENGINE V3 (SD 1.5 + LINEART + IP-ADAPTER)
# ============================================================================
V3_ENGINE_ENABLED = True
SD15_MODEL_ID = "runwayml/stable-diffusion-v1-5"
CONTROLNET_LINEART_ID = "lllyasviel/control_v11p_sd15s2_lineart_anime"
IP_ADAPTER_V3_REPO = "h94/IP-Adapter"
IP_ADAPTER_V3_FILE = "ip-adapter-plus-face_sd15.bin"
V3_STEPS = 50                  # Aumentado para 50 para máxima nitidez (Euler A)
V3_STRENGTH = 0.75
V3_GUIDANCE_SCALE = 9.0         # Aumentado para 9.0 para melhor aderência ao prompt
V3_IP_SCALE = 0.7
V3_CONTROL_SCALE = 0.8

# Presets de Qualidade V3
QUALITY_PRESETS = {
    "fast": {
        "steps": 15,
    },
    "balanced": {
        "steps": 20,
    },
    "high": {
        "steps": 30,
    }
}

# Text Compositing: Preservação de texto/balões
# ENABLE_TEXT_COMPOSITING: Ativa/desativa a preservação de texto
# False = Deixa a IA colorizar tudo (texto pode ficar colorido)
# True = Preserva texto da original (evita texto colorido, mas pode ter artefatos)
ENABLE_TEXT_COMPOSITING = False     # DESATIVADO por padrão - evita "efeito carta de resgate"

# Filtros do Smart Text Compositing (quando ENABLE_TEXT_COMPOSITING=True)
TEXT_COMPOSITING_MIN_AREA = 100     # Área mínima em pixels
TEXT_COMPOSITING_MAX_AREA = 50000   # Área máxima (evita caixas enormes)
TEXT_COMPOSITING_CONFIDENCE = 0.5   # Confiança mínima do detector

# VAE otimizado para FP16
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"

# ============================================================================
# RESOLUÇÃO DE SAÍDA (Qualidade vs Tamanho)
# ============================================================================

# SKIP_FINAL_DOWNSCALE: Mantém a imagem na resolução de geração (maior qualidade)
# True  = Máxima qualidade: mantém resolução SDXL (ex: 1024x1408)
#         - Imagens ~2.4x maiores
#         - Sem perda de detalhes no downscale
#         - Requer adaptação no text compositing
# False = Compatibilidade: volta para resolução original (ex: 650x918)
#         - Menor tamanho de arquivo
#         - Possível perda de detalhes finos
SKIP_FINAL_DOWNSCALE = True  # Recomendado: True para máxima qualidade

# ============================================================================
# TILE-AWARE PROCESSING (Princípio da Localidade)
# ============================================================================

# Configurações de Tile
MAX_REF_PER_TILE = 2               # Top-K: máximo de personagens por tile
TILE_STRIDE = TILE_SIZE - TILE_OVERLAP  # Stride com overlap

# Seleção Top-K por relevância
def calculate_prominence(bbox: Tuple[int, int, int, int], 
                         image_size: Tuple[int, int]) -> float:
    """
    Calcula score de prominence = Área * Centralidade.
    Personagens no centro têm maior peso.
    
    Args:
        bbox: (x1, y1, x2, y2)
        image_size: (width, height)
    
    Returns:
        Score de prominence (maior = mais importante)
    """
    x1, y1, x2, y2 = bbox
    img_w, img_h = image_size
    
    # Área
    area = (x2 - x1) * (y2 - y1)
    
    # Centralidade (distância do centro normalizada)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    img_cx, img_cy = img_w / 2, img_h / 2
    
    dist_from_center = ((cx - img_cx) ** 2 + (cy - img_cy) ** 2) ** 0.5
    max_dist = ((img_w / 2) ** 2 + (img_h / 2) ** 2) ** 0.5
    centrality = 1.0 - (dist_from_center / max_dist)  # 1.0 = centro, 0.0 = canto
    
    return area * centrality


# ============================================================================
# DIFFERENTIAL DIFFUSION - MÁSCARAS PROGRESSIVAS
# ============================================================================

# Gaussian Blur para máscaras suaves (Differential Diffusion)
GAUSSIAN_KERNEL_SIZE = 51          # Tamanho do kernel para blur
GAUSSIAN_SIGMA = 10.0              # Sigma para decaimento suave

# Máscaras de Força Variável (Change Map)
MASK_STRENGTH_CENTER = 1.0         # Força no centro do personagem
MASK_STRENGTH_EDGE = 0.0           # Força nas bordas
MASK_DECAY_TYPE = "gaussian"       # Tipo de decaimento: "gaussian", "linear"

# Background Isolation
BACKGROUND_IP_SCALE = 0.0          # IP-Adapter scale = 0 para background

# IP-Adapter Control (Global)
IP_ADAPTER_SCALE_DEFAULT = 0.7     # Escala padrão (pode ser sobrescrita por V3_IP_SCALE)
IP_ADAPTER_END_STEP = 0.6          # Step em que o IP-Adapter para de afetar (0.0-1.0)


# ============================================================================
# PROMPTS E ESTILOS
# ============================================================================

DEFAULT_NEGATIVE_PROMPT = (
    "nsfw, lowres, bad hands, text, watermark, blurry, "
    "neon, psychedelic, abstract, horror, zombie, monochrome, "
    "grayscale, screen tones, bad anatomy, worst quality, "
    "3d render, plastic texture, oversaturated"
)

PROMPT_SUFFIX = ", masterpiece, best quality, anime style, vibrant colors, detailed shading"

# Presets de Estilo para Narrativa
STYLE_PRESETS = {
    "default": {
        "prompt_addition": "",
        "negative_addition": "",
        "color_temperature": "neutral"
    },
    "vibrant": {
        "prompt_addition": "vibrant colors, saturated, high contrast, vivid tones",
        "negative_addition": "muted colors, desaturated, flat",
        "color_temperature": "warm"
    },
    "muted": {
        "prompt_addition": "soft colors, muted tones, pastel, gentle shading",
        "negative_addition": "vibrant, saturated, neon, harsh contrast",
        "color_temperature": "cool"
    },
    "sepia": {
        "prompt_addition": "sepia tone, vintage, warm brown tones, nostalgic",
        "negative_addition": "colorful, vibrant, modern, blue tones",
        "color_temperature": "warm"
    },
    "flashback": {
        "prompt_addition": "desaturated, faded colors, nostalgic, soft focus, memories",
        "negative_addition": "vibrant, saturated, sharp, high contrast",
        "color_temperature": "cool"
    },
    "dream": {
        "prompt_addition": "ethereal, glowing, surreal colors, fantasy, soft light",
        "negative_addition": "realistic, mundane, flat, harsh lighting",
        "color_temperature": "warm"
    },
    "nightmare": {
        "prompt_addition": "dark, ominous, red and black tones, distorted",
        "negative_addition": "bright, cheerful, pastel, soft",
        "color_temperature": "cold"
    }
}


# ============================================================================
# PALETA E CORES
# ============================================================================

# Regiões para extração de paleta
PALETTE_REGIONS = ["hair", "skin", "eyes", "clothes_primary", "clothes_secondary", "accessories"]

# Espaço de cor para processamento
COLOR_SPACE = "CIELAB"             # Delta E perceptualmente uniforme

# Thresholds de consistência
PALETTE_DRIFT_THRESHOLD = 5.0      # Delta E em CIELAB para considerar mudança
TEMPORAL_SMOOTHING = 0.3           # Fator de suavização temporal entre páginas

# Número de cores por região
PALETTE_COLORS_PER_REGION = 5


# ============================================================================
# DATABASE E CACHE (Two-Pass Architecture)
# ============================================================================

# Tipo de database
DATABASE_TYPE = "hybrid"           # FAISS + Parquet
FAISS_INDEX_TYPE = "IndexFlatIP"   # Inner Product para similaridade cosseno
EMBEDDING_DIM = 768                # Dimensão do embedding CLIP

# Diretórios de dados
DATA_DIR = Path("./data")
CACHE_DIR = DATA_DIR / "cache"
CHAPTER_CACHE_DIR = Path(os.getenv("MANGA_CHAPTER_CACHE", "./chapter_cache"))  # Injetável via ambiente
CHARACTER_DB_DIR = DATA_DIR / "character_db"
REFERENCE_GALLERY_DIR = DATA_DIR / "reference_gallery"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"  # Cache de tensores .pt
MASKS_DIR = DATA_DIR / "masks"             # Máscaras gaussianas pré-calculadas

# Criar diretórios se não existirem
for dir_path in [CACHE_DIR, CHARACTER_DB_DIR, REFERENCE_GALLERY_DIR, 
                 EMBEDDINGS_DIR, MASKS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Cache de Tensores (Imutabilidade no Pass 1)
EMBEDDING_CACHE_FORMAT = "pt"      # PyTorch tensor format
MAX_CACHED_EMBEDDINGS = 1000       # Limite de embeddings em memória


# ============================================================================
# BLENDING E PÓS-PROCESSAMENTO
# ============================================================================

# Latent Blending
FEATHER_WIDTH = 64                 # Pixels para feathering da máscara
LATENT_BLEND_ALPHA = 0.5           # Alpha para blending no espaço latente

# Poisson Blending (harmonização final)
POISSON_BLEND_ENABLED = True
POISSON_BLEND_ITERATIONS = 100

# Preservação de linhas
LINE_PRESERVATION_ENABLED = True
LINE_THRESHOLD = 0.1
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150


# ============================================================================
# ADR 004: SAM 2.1 SEGMENTAÇÃO SEMÂNTICA E Z-BUFFER
# ============================================================================

# SAM 2.1 Configurações
SAM2_ENABLED = True
SAM2_MODEL_SIZE = "tiny"  # tiny (35MB), small (80MB), base (180MB), large (400MB)
SAM2_DEVICE = "cpu"  # 'cpu' para Pass 1 (preserva VRAM), 'cuda' se disponível
SAM2_USE_ONNX = False  # Usar versão ONNX se disponível (mais estável em Windows)

# Z-Buffer Hierárquico Configurações
ZBUFFER_ENABLED = True
ZBUFFER_WEIGHT_Y = 0.5        # Posição vertical — soma=1.0 sem MiDaS (0.5+0.3+0.2)
ZBUFFER_WEIGHT_AREA = 0.3     # Área relativa (0.0-1.0)
ZBUFFER_WEIGHT_TYPE = 0.2     # Prioridade semântica (0.0-1.0)
ZBUFFER_WEIGHT_DEPTH = 0.1    # Profundidade MiDaS (0.0-1.0)
ZBUFFER_USE_MIDAS = False     # MiDaS Small opcional para profundidade

# Mask Processing (ADR 004)
MASK_MORPH_CLOSE_KERNEL = 3   # Kernel para morphological close
MASK_EROSION_PIXELS = 2       # Pixels para erosão em contatos
MASK_EDGE_BLUR_SIGMA = 0.5    # Sigma do blur gaussiano nas bordas
MASK_OVERLAP_DILATION = 1     # Dilatação para garantir overlap mínimo

# Fallback
SAM2_FALLBACK_TO_BBOX = True  # Se SAM falhar, usar BBox automático

# ============================================================================
# NARRATIVA E CONTEXTO
# ============================================================================

TIPOS_DE_CENA = ["present", "flashback", "dream", "nightmare", "hell", "memory"]
DEFAULT_SCENE_TYPE = "present"

# Herança de contexto entre páginas
CONTEXT_INHERITANCE_DECAY = 0.1    # Decaimento por página
CONTEXT_WINDOW_SIZE = 3            # Janela de contexto para consistência


# ============================================================================
# FUNÇÕES UTILITÁRIAS
# ============================================================================

def get_model_cache_dir() -> Path:
    """Retorna diretório de cache de modelos HuggingFace"""
    cache_dir = os.getenv("HF_HOME", "./models")
    return Path(cache_dir)


def get_device_properties() -> Dict[str, Any]:
    """Retorna propriedades do dispositivo CUDA se disponível"""
    if DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        return {
            "name": props.name,
            "total_memory_gb": props.total_memory / 1e9,
            "major": props.major,
            "minor": props.minor,
            "multi_processor_count": props.multi_processor_count
        }
    return {"name": "CPU", "total_memory_gb": 0, "major": 0, "minor": 0}


def is_high_vram() -> bool:
    """Verifica se há VRAM suficiente para modo high-quality"""
    if DEVICE == "cuda":
        props = get_device_properties()
        return props.get("total_memory_gb", 0) > 16
    return False


def get_optimal_batch_size() -> int:
    """Retorna batch size ótimo baseado na VRAM disponível"""
    if DEVICE == "cuda":
        props = get_device_properties()
        vram_gb = props.get("total_memory_gb", 0)
        if vram_gb > 20:
            return 4
        elif vram_gb > 12:
            return 2
        else:
            return 1
    return 1


def get_optimal_tile_size(image_size: Tuple[int, int]) -> int:
    """
    Determina tamanho de tile ótimo baseado na resolução da imagem.
    
    Args:
        image_size: (width, height) da imagem
        
    Returns:
        Tamanho do tile em pixels
    """
    width, height = image_size
    max_dim = max(width, height)
    
    if max_dim <= 1024:
        return 1024
    elif max_dim <= 1536:
        return 1024  # Ainda usa 1024 com mais overlap
    else:
        return 1024  # Sempre 1024 para SDXL, independente da resolução


def calculate_tile_grid(image_size: Tuple[int, int], 
                        tile_size: int = TILE_SIZE,
                        overlap: int = TILE_OVERLAP) -> Tuple[int, int, list]:
    """
    Calcula grid de tiles para uma imagem.
    
    Args:
        image_size: (width, height)
        tile_size: Tamanho de cada tile
        overlap: Overlap entre tiles
        
    Returns:
        (num_tiles_x, num_tiles_y, list_of_bboxes)
        Cada bbox é (x1, y1, x2, y2)
    """
    width, height = image_size
    stride = tile_size - overlap
    
    tiles = []
    
    # Calcula número de tiles em cada dimensão
    if width <= tile_size:
        x_starts = [0]
    else:
        x_starts = list(range(0, width - tile_size + 1, stride))
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
    
    return len(x_starts), len(y_starts), tiles


def get_ip_adapter_scale_for_step(current_step: int, total_steps: int) -> float:
    """
    Calcula escala do IP-Adapter para um step específico (Temporal Decay).
    
    Args:
        current_step: Step atual (0-based)
        total_steps: Número total de steps
        
    Returns:
        Escala do IP-Adapter (0.0 a IP_ADAPTER_SCALE_DEFAULT)
    """
    progress = current_step / total_steps
    
    if progress > IP_ADAPTER_END_STEP:
        return 0.0  # Desliga após 60%
    
    # Decaimento suave até o ponto de corte
    decay_factor = 1.0 - (progress / IP_ADAPTER_END_STEP)
    return IP_ADAPTER_SCALE_DEFAULT * decay_factor


# ============================================================================
# CONSTANTES DE ERRO E LOGGING
# ============================================================================

VERBOSE = os.getenv("MANGA_COLOR_VERBOSE", "true").lower() == "true"
LOG_LEVEL = os.getenv("MANGA_COLOR_LOG_LEVEL", "INFO")


# ============================================================================
# VALIDAÇÃO DE CONFIGURAÇÃO
# ============================================================================

def validate_config() -> bool:
    """
    Valida se a configuração é consistente.
    
    Returns:
        True se configuração é válida
        
    Raises:
        ValueError se houver inconsistências
    """
    # Verifica consistência de Tile-Aware
    if TILE_OVERLAP >= TILE_SIZE:
        raise ValueError("TILE_OVERLAP deve ser menor que TILE_SIZE")
    
    # Verifica limites de IP-Adapter
    if not 0 <= IP_ADAPTER_END_STEP <= 1:
        raise ValueError("IP_ADAPTER_END_STEP deve estar entre 0 e 1")
    
    # Verifica diretórios
    for dir_path in [CACHE_DIR, CHARACTER_DB_DIR, EMBEDDINGS_DIR]:
        if not dir_path.exists():
            raise ValueError(f"Diretório não existe: {dir_path}")
    
    return True


# Executa validação no import
if __name__ != "__main__":
    try:
        validate_config()
    except ValueError as e:
        print(f"[Config Warning] {e}")

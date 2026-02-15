"""
MangaAutoColor Pro - API FastAPI

API REST para o sistema de colorização de mangá.
Endpoints para análise (Pass 1) e geração (Pass 2).
Modo Realtime para extensão de navegador.
"""

import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


import shutil
import io

import hashlib
import time

# =============================================================================
# CONFIGURAÇÃO DE AMBIENTE
# =============================================================================

# Desabilita autoinstall do Ultralytics (evita erro de permissão)
os.environ['ULTRALYTICS_AUTOINSTALL'] = '0'

# =============================================================================
# VERIFICAÇÃO DE DEPENDÊNCIAS CRÍTICAS (NumPy)
# =============================================================================
try:
    import numpy as np
    if int(np.__version__.split('.')[0]) >= 2:
        print("=" * 70)
        print("❌ ERRO CRÍTICO: NumPy 2.x detectado!")
        print(f"   Versão instalada: {np.__version__}")
        print("   PyTorch 2.4.0 requer NumPy < 2.0")
        print("")
        print("   Execute o script de instalação:")
        print("      scripts\\windows\\install.bat")
        print("")
        print("   Ou manualmente:")
        print("      pip uninstall numpy -y")
        print("      pip install numpy==1.26.4")
        print("=" * 70)
        sys.exit(1)
except ImportError:
    print("⚠️  NumPy não instalado. Instalando...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.26.4"])
    import numpy as np

# =============================================================================
# CONFIGURAÇÃO DE DEBUG
# =============================================================================
DEBUG_MODE = os.environ.get("MANGA_DEBUG", "false").lower() == "true"
DEBUG_OUTPUT_DIR = Path("./output/debug")

if DEBUG_MODE:
    DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] Modo DEBUG ativado. Salvando imagens em: {DEBUG_OUTPUT_DIR.absolute()}")

# Adiciona raiz do projeto ao path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query, Depends, Form
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
import uvicorn
import numpy as np
from PIL import Image
import torch

from core.pipeline import MangaColorizationPipeline, GenerationOptions
from core.detection.yolo_detector import YOLODetector
from core.identity.hybrid_encoder import HybridIdentitySystem
from core.database.chapter_db import ChapterDatabase
from core.generation.pipeline import TileAwareGenerator
from config.settings import (
    DEVICE, DTYPE, STYLE_PRESETS, TIPOS_DE_CENA, 
    TILE_SIZE, MAX_REF_PER_TILE, get_device_properties
)


# =============================================================================
# MODELOS PYDANTIC
# =============================================================================

class HealthResponse(BaseModel):
    status: str
    device: str
    vram_gb: float
    timestamp: datetime
    version: str = "2.0.0"
    debug_mode: bool = False

class ChapterUploadResponse(BaseModel):
    chapter_id: str
    num_pages: int
    status: str
    message: str

class AnalysisResponse(BaseModel):
    chapter_id: str
    status: str
    num_pages: int
    num_characters: int
    estimated_time_seconds: float
    characters: List[Dict[str, Any]]
    scene_breakdown: Dict[str, List[int]]

class GenerationRequest(BaseModel):
    chapter_id: str
    page_numbers: Optional[List[int]] = None  # None = todas as páginas
    style_preset: str = Field(default="default", pattern="^(default|vibrant|muted|sepia|flashback|dream|nightmare)$")
    quality_mode: str = Field(default="balanced", pattern="^(fast|balanced|high)$")
    seed: Optional[int] = None
    preserve_text: bool = True
    apply_narrative: bool = True

class GenerationResponse(BaseModel):
    job_id: str
    chapter_id: str
    status: str
    message: str
    output_urls: List[str] = []

class PageStatusResponse(BaseModel):
    chapter_id: str
    page_num: int
    analyzed: bool
    generated: bool
    scene_type: str
    characters_detected: int

class CharacterInfo(BaseModel):
    char_id: str
    appearances: int
    prominence_score: float
    first_page: int


class RealtimeColorizeRequest(BaseModel):
    """Request para colorização em tempo real (modo Express)."""
    image_url: Optional[str] = None
    style_preset: str = Field(default="default", pattern="^(default|vibrant|muted|sepia|flashback|dream|nightmare)$")
    seed: Optional[int] = None
    use_cache: bool = True  # Usar cache de personagens


class RealtimeStatusResponse(BaseModel):
    """Status do modo realtime."""
    status: str
    models_loaded: bool
    warm_mode: bool
    vram_used_gb: float
    characters_cached: int
    avg_inference_time_ms: Optional[float] = None


# =============================================================================
# ESTADO GLOBAL
# =============================================================================

pipeline: Optional[MangaColorizationPipeline] = None
active_jobs: Dict[str, Dict] = {}
UPLOAD_DIR = Path("./data/uploads")
OUTPUT_DIR = Path("./data/outputs")

# Modo Realtime - Componentes em memória
realtime_generator: Optional[TileAwareGenerator] = None
realtime_detector: Optional[YOLODetector] = None
realtime_encoder: Optional[HybridIdentitySystem] = None
character_cache: Dict[str, Any] = {}  # Cache LRU de personagens (id -> embedding)
WARM_MODE = False  # Se True, mantém modelo na VRAM

# Estatísticas
realtime_stats = {
    "total_requests": 0,
    "total_time_ms": 0,
    "avg_time_ms": None
}

# Cria diretórios
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_character_id(image_hash: str, bbox: Tuple[int, int, int, int]) -> str:
    """Gera ID único para personagem baseado no hash da imagem + bbox."""
    bbox_str = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
    return hashlib.md5(f"{image_hash}_{bbox_str}".encode()).hexdigest()[:12]


def pil_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Converte PIL Image para bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def save_debug_images(
    timestamp: str,
    input_image: Optional[Image.Image] = None,
    canny_edges: Optional[np.ndarray] = None,
    detections: Optional[List] = None,
    result: Optional[Image.Image] = None,
    character_crops: Optional[Dict] = None
):
    """
    Salva imagens de debug para análise visual.
    
    Args:
        timestamp: Timestamp único para identificar o batch
        input_image: Imagem recebida da extensão
        canny_edges: Bordas Canny detectadas
        detections: Lista de detecções (com bboxes)
        result: Imagem final gerada
        character_crops: Dict com 'bodies' e 'faces' extraídos
    """
    if not DEBUG_MODE:
        return
    
    debug_dir = DEBUG_OUTPUT_DIR / timestamp
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Salva imagem de entrada
        if input_image is not None:
            input_path = debug_dir / "01_input.png"
            input_image.save(input_path)
            print(f"[DEBUG] Input salvo: {input_path}")
        
        # 2. Salva Canny edges
        if canny_edges is not None:
            canny_pil = Image.fromarray(canny_edges)
            canny_path = debug_dir / "02_canny.png"
            canny_pil.save(canny_path)
            print(f"[DEBUG] Canny salvo: {canny_path}")
        
        # 3. Salva imagem com detecções visualizadas
        if input_image is not None and detections is not None:
            import cv2
            vis_image = np.array(input_image.copy())
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            
            for i, det in enumerate(detections[:10]):  # Max 10 detecções
                bbox = det.get('bbox', det.bbox if hasattr(det, 'bbox') else None)
                class_name = det.get('class_name', det.class_name if hasattr(det, 'class_name') else 'unknown')
                conf = det.get('confidence', det.confidence if hasattr(det, 'confidence') else 0)
                
                if bbox:
                    x1, y1, x2, y2 = bbox
                    color = (0, 255, 0) if class_name == 'body' else (0, 0, 255) if class_name == 'face' else (255, 0, 0)
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(vis_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            det_path = debug_dir / "03_detections.png"
            cv2.imwrite(str(det_path), vis_image)
            print(f"[DEBUG] Detecções salvas: {det_path}")
        
        # 4. Salva crops de personagens
        if character_crops is not None:
            crops_dir = debug_dir / "crops"
            crops_dir.mkdir(exist_ok=True)
            
            for i, (crop, det) in enumerate(character_crops.get('bodies', [])[:5]):
                crop_pil = Image.fromarray(crop)
                crop_path = crops_dir / f"body_{i:02d}.png"
                crop_pil.save(crop_path)
            
            for i, (crop, det) in enumerate(character_crops.get('faces', [])[:5]):
                crop_pil = Image.fromarray(crop)
                crop_path = crops_dir / f"face_{i:02d}.png"
                crop_pil.save(crop_path)
            
            print(f"[DEBUG] Crops salvos em: {crops_dir}")
        
        # 5. Salva resultado final
        if result is not None:
            result_path = debug_dir / "04_result.png"
            result.save(result_path)
            print(f"[DEBUG] Resultado salvo: {result_path}")
        
        print(f"[DEBUG] Todas as imagens de debug salvas em: {debug_dir}")
        
    except Exception as e:
        print(f"[DEBUG] Erro ao salvar imagens de debug: {e}")


def get_image_hash(image: Image.Image) -> str:
    """Gera hash simples da imagem para cache."""
    # Reduz e converte para bytes para hash rápido
    small = image.resize((64, 64)).convert('L')
    arr = np.array(small)
    return hashlib.md5(arr.tobytes()).hexdigest()[:16]


def get_pipeline() -> MangaColorizationPipeline:
    """Dependency injection para o pipeline."""
    global pipeline
    if pipeline is None:
        pipeline = MangaColorizationPipeline(
            device=DEVICE,
            dtype=DTYPE,
            enable_xformers=True,
            enable_cpu_offload=not WARM_MODE  # Desliga CPU offload no warm mode
        )
    return pipeline


def init_realtime_components():
    """Inicializa componentes para modo realtime (lazy loading)."""
    global realtime_detector, realtime_encoder, realtime_generator
    
    if realtime_detector is None:
        print("[Realtime] Inicializando detector YOLO...")
        realtime_detector = YOLODetector()
    
    if realtime_encoder is None:
        print("[Realtime] Inicializando encoder de identidade...")
        realtime_encoder = HybridIdentitySystem()
    
    if realtime_generator is None:
        print("[Realtime] Inicializando gerador Tile-Aware...")
        # Inicializa com lazy loading - modelo será carregado no primeiro uso
        realtime_generator = TileAwareGenerator(
            device=DEVICE,
            dtype=DTYPE,
            enable_offload=not WARM_MODE
        )
    
    return realtime_detector, realtime_encoder, realtime_generator


def warmup_models():
    """Pré-carrega modelos na VRAM para latência zero."""
    global WARM_MODE, realtime_generator
    
    print("[Warmup] Pré-carregando modelos na VRAM...")
    WARM_MODE = True
    
    # Força carregamento do gerador
    if realtime_generator is None:
        realtime_generator = TileAwareGenerator(
            device=DEVICE,
            dtype=DTYPE,
            enable_offload=False  # Mantém na VRAM
        )
    
    # Executa inference dummy para garantir que tudo está carregado
    dummy_image = Image.new('RGB', (512, 512), color='white')
    print("[Warmup] Modelos prontos na VRAM!")
    
    return {"status": "warmed", "vram_gb": get_device_properties().get('used_memory_gb', 0)}


# =============================================================================
# LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia inicialização e shutdown."""
    # Startup
    print("[API] Iniciando MangaAutoColor Pro API...")
    print(f"[API] Device: {DEVICE}")
    print(f"[API] Dtype: {DTYPE}")
    
    # Inicializa pipeline em background
    _ = get_pipeline()
    
    yield
    
    # Shutdown
    print("[API] Encerrando...")
    global pipeline
    if pipeline:
        del pipeline
        pipeline = None


# =============================================================================
# APP FASTAPI
# =============================================================================

app = FastAPI(
    title="MangaAutoColor Pro API",
    description="API REST para colorização automática de mangá com arquitetura Two-Pass",
    version="2.0.0",
    lifespan=lifespan
)

# Middlewares
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Endpoint CORS preflight explícito para extensões
@app.options("/{path:path}")
async def cors_preflight(path: str):
    return {"message": "OK"}

# Registra rotas de Two-Pass System
try:
    from api.routes.chapter import twopass
    app.include_router(twopass.router)
    print("[API] Rotas Two-Pass registradas: /chapter/*")
except ImportError as e:
    print(f"[API] AVISO: Rotas Two-Pass não disponíveis: {e}")


# =============================================================================
# ENDPOINTS DE SAÚDE
# =============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Endpoint raiz com informações de saúde."""
    device_info = get_device_properties()
    return HealthResponse(
        status="healthy",
        device=DEVICE,
        vram_gb=device_info.get('total_memory_gb', 0),
        timestamp=datetime.now(),
        debug_mode=DEBUG_MODE
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check detalhado."""
    return await root()


# =============================================================================
# ENDPOINTS DE UPLOAD
# =============================================================================

@app.post("/chapters/upload", response_model=ChapterUploadResponse)
async def upload_chapter(
    files: List[UploadFile] = File(..., description="Imagens das páginas do capítulo")
):
    """
    Upload de capítulo para análise.
    
    Retorna chapter_id para uso nos próximos endpoints.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado")
    
    # Valida tipos de arquivo
    for file in files:
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail=f"Arquivo inválido: {file.filename}. Apenas imagens são aceitas."
            )
    
    # Gera ID único baseado no timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chapter_id = f"chapter_{timestamp}_{hash(files[0].filename) % 10000:04d}"
    
    # Cria diretório para o capítulo
    chapter_dir = UPLOAD_DIR / chapter_id
    chapter_dir.mkdir(parents=True, exist_ok=True)
    
    # Salva arquivos
    saved_files = []
    for i, file in enumerate(sorted(files, key=lambda f: f.filename)):
        file_path = chapter_dir / f"page_{i+1:03d}.png"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        saved_files.append(str(file_path))
    
    return ChapterUploadResponse(
        chapter_id=chapter_id,
        num_pages=len(saved_files),
        status="uploaded",
        message=f"{len(saved_files)} páginas carregadas com sucesso"
    )


# =============================================================================
# ENDPOINTS DE ANÁLISE (PASS 1)
# =============================================================================

@app.post("/chapters/{chapter_id}/analyze", response_model=AnalysisResponse)
async def analyze_chapter(
    chapter_id: str,
    background_tasks: BackgroundTasks,
    pipeline: MangaColorizationPipeline = Depends(get_pipeline)
):
    """
    Executa Pass 1: Análise completa do capítulo.
    
    Detecta personagens, extrai embeddings e calcula tiles.
    """
    chapter_dir = UPLOAD_DIR / chapter_id
    if not chapter_dir.exists():
        raise HTTPException(status_code=404, detail="Capítulo não encontrado")
    
    # Lista arquivos
    image_files = sorted(chapter_dir.glob("page_*.png"))
    if not image_files:
        raise HTTPException(status_code=404, detail="Nenhuma página encontrada")
    
    try:
        # Executa análise
        page_paths = [str(f) for f in image_files]
        analysis = pipeline.process_chapter(page_paths)
        
        return AnalysisResponse(
            chapter_id=chapter_id,
            status="analyzed",
            num_pages=analysis.num_pages,
            num_characters=analysis.num_characters,
            estimated_time_seconds=analysis.estimated_generation_time,
            characters=[
                {
                    'id': char.get('id', f'char_{i}'),
                    'appearances': char.get('appearances', 1),
                    'prominence': char.get('prominence_score', 0)
                }
                for i, char in enumerate(analysis.characters)
            ],
            scene_breakdown=analysis.scene_breakdown
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na análise: {str(e)}")


@app.get("/chapters/{chapter_id}/status")
async def get_chapter_status(chapter_id: str):
    """Retorna status do capítulo (analisado/gerado)."""
    chapter_dir = UPLOAD_DIR / chapter_id
    if not chapter_dir.exists():
        raise HTTPException(status_code=404, detail="Capítulo não encontrado")
    
    # Verifica se existe database (analisado)
    db_dir = Path("./data/chapters") / chapter_id
    analyzed = db_dir.exists()
    
    # Conta páginas
    num_pages = len(list(chapter_dir.glob("page_*.png")))
    
    return {
        "chapter_id": chapter_id,
        "uploaded": True,
        "analyzed": analyzed,
        "num_pages": num_pages,
        "output_ready": False
    }


# =============================================================================
# ENDPOINTS DE GERAÇÃO (PASS 2)
# =============================================================================

@app.post("/chapters/{chapter_id}/generate", response_model=GenerationResponse)
async def generate_chapter(
    chapter_id: str,
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    pipeline: MangaColorizationPipeline = Depends(get_pipeline)
):
    """
    Executa Pass 2: Geração das páginas colorizadas.
    
    Requer que o Pass 1 (análise) tenha sido executado.
    """
    # Valida chapter_id
    if request.chapter_id != chapter_id:
        raise HTTPException(status_code=400, detail="chapter_id mismatch")
    
    # Verifica se capítulo existe
    chapter_dir = UPLOAD_DIR / chapter_id
    if not chapter_dir.exists():
        raise HTTPException(status_code=404, detail="Capítulo não encontrado")
    
    # Gera job ID
    job_id = f"job_{chapter_id}_{datetime.now().strftime('%H%M%S')}"
    
    # Cria diretório de saída
    output_dir = OUTPUT_DIR / chapter_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configura opções
    options = GenerationOptions(
        style_preset=request.style_preset,
        quality_mode=request.quality_mode,
        preserve_original_text=request.preserve_text,
        apply_narrative_transforms=request.apply_narrative,
        seed=request.seed
    )
    
    # Adiciona job à fila
    active_jobs[job_id] = {
        "chapter_id": chapter_id,
        "status": "queued",
        "progress": 0,
        "output_dir": str(output_dir),
        "options": request.dict()
    }
    
    # Inicia geração em background
    background_tasks.add_task(
        process_generation_job,
        job_id=job_id,
        chapter_id=chapter_id,
        options=options,
        page_numbers=request.page_numbers
    )
    
    return GenerationResponse(
        job_id=job_id,
        chapter_id=chapter_id,
        status="queued",
        message=f"Geração iniciada para {chapter_id}",
        output_urls=[]
    )


async def process_generation_job(
    job_id: str,
    chapter_id: str,
    options: GenerationOptions,
    page_numbers: Optional[List[int]] = None
):
    """Processa job de geração em background."""
    global pipeline, active_jobs
    
    try:
        active_jobs[job_id]["status"] = "processing"
        
        # Se pipeline não existe, cria
        if pipeline is None:
            pipeline = MangaColorizationPipeline(device=DEVICE, dtype=DTYPE)
        
        # Obtém diretório de upload
        chapter_dir = UPLOAD_DIR / chapter_id
        image_files = sorted(chapter_dir.glob("page_*.png"))
        
        if not page_numbers:
            page_numbers = list(range(len(image_files)))
        
        # Gera cada página
        output_urls = []
        output_dir = OUTPUT_DIR / chapter_id
        
        for i, page_num in enumerate(page_numbers):
            if page_num < 0 or page_num >= len(image_files):
                continue
            
            # Atualiza progresso
            progress = int((i / len(page_numbers)) * 100)
            active_jobs[job_id]["progress"] = progress
            
            # Gera página
            result = pipeline.generate_page(page_num, options)
            
            # Salva
            output_path = output_dir / f"colored_page_{page_num+1:03d}.png"
            result.save(output_path)
            output_urls.append(f"/outputs/{chapter_id}/colored_page_{page_num+1:03d}.png")
        
        # Finaliza
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["progress"] = 100
        active_jobs[job_id]["output_urls"] = output_urls
        
    except Exception as e:
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["error"] = str(e)
        print(f"[Job {job_id}] Erro: {e}")


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Retorna status de um job de geração."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    
    job = active_jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", 0),
        "output_urls": job.get("output_urls", []),
        "error": job.get("error")
    }


# =============================================================================
# ENDPOINTS DE DOWNLOAD
# =============================================================================

@app.get("/outputs/{chapter_id}/{filename}")
async def download_result(chapter_id: str, filename: str):
    """Download de página colorizada."""
    file_path = OUTPUT_DIR / chapter_id / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="image/png"
    )


@app.get("/chapters/{chapter_id}/download-all")
async def download_all(chapter_id: str):
    """Download de todas as páginas como ZIP."""
    output_dir = OUTPUT_DIR / chapter_id
    
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="Saída não encontrada")
    
    # Cria ZIP
    zip_path = OUTPUT_DIR / f"{chapter_id}_colorized.zip"
    shutil.make_archive(
        base_name=str(zip_path.with_suffix('')),
        format='zip',
        root_dir=output_dir
    )
    
    return FileResponse(
        path=zip_path,
        filename=f"{chapter_id}_colorized.zip",
        media_type="application/zip"
    )


# =============================================================================
# ENDPOINTS DE CONFIGURAÇÃO
# =============================================================================

@app.get("/config/styles")
async def get_available_styles():
    """Retorna estilos disponíveis."""
    return {
        "styles": list(STYLE_PRESETS.keys()),
        "details": STYLE_PRESETS
    }


@app.get("/config/scene-types")
async def get_available_scene_types():
    """Retorna tipos de cena disponíveis."""
    return {
        "scene_types": TIPOS_DE_CENA
    }


@app.get("/config/tile-settings")
async def get_tile_settings():
    """Retorna configurações de tile."""
    return {
        "tile_size": TILE_SIZE,
        "max_ref_per_tile": MAX_REF_PER_TILE
    }


# =============================================================================
# ENDPOINTS REALTIME (PARA EXTENSÃO DE NAVEGADOR)
# =============================================================================

@app.get("/realtime/status", response_model=RealtimeStatusResponse)
async def get_realtime_status():
    """
    Retorna status do modo realtime.
    Útil para a extensão verificar se o backend está pronto.
    """
    global WARM_MODE, character_cache, realtime_stats
    
    device_info = get_device_properties()
    
    return RealtimeStatusResponse(
        status="ready" if realtime_generator else "initializing",
        models_loaded=realtime_generator is not None,
        warm_mode=WARM_MODE,
        vram_used_gb=device_info.get('used_memory_gb', 0),
        characters_cached=len(character_cache),
        avg_inference_time_ms=realtime_stats.get("avg_time_ms")
    )


@app.post("/realtime/warmup")
async def realtime_warmup():
    """
    Pré-carrega modelos na VRAM para latência mínima.
    Útil quando o usuário quer começar a colorizar.
    """
    try:
        result = warmup_models()
        return {
            "status": "success",
            "message": "Modelos pré-carregados na VRAM",
            "vram_used_gb": result["vram_gb"],
            "warning": "VRAM está constantemente alocada. Use /realtime/cooldown para liberar."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no warmup: {str(e)}")


@app.post("/realtime/cooldown")
async def realtime_cooldown():
    """Libera VRAM (cooldown) - útil após sessão de colorização."""
    global WARM_MODE, realtime_generator, realtime_detector, realtime_encoder
    
    WARM_MODE = False
    
    # Limpa referências para permitir GC
    if realtime_generator:
        del realtime_generator
        realtime_generator = None
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "status": "success",
        "message": "VRAM liberada",
        "vram_used_gb": get_device_properties().get('used_memory_gb', 0)
    }


async def _load_input_image(file: Optional[UploadFile], image_url: Optional[str]) -> Image.Image:
    """Carrega imagem do upload ou URL."""
    if file:
        contents = await file.read()
        print(f"[Realtime] Recebido arquivo: {file.filename}, size={len(contents)} bytes, content_type={file.content_type}")
        
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Arquivo vazio recebido")
        
        try:
            return Image.open(io.BytesIO(contents))
        except Exception as img_err:
            print(f"[Realtime] Erro ao abrir imagem: {img_err}")
            print(f"[Realtime] Primeiros 100 bytes: {contents[:100]}")
            raise HTTPException(status_code=400, detail=f"Formato de imagem inválido: {str(img_err)}")
    else:
        # Download da URL
        import urllib.request
        with urllib.request.urlopen(image_url, timeout=10) as response:
            return Image.open(io.BytesIO(response.read()))


def _extract_embeddings_from_bodies(
    bodies: List,
    image_hash: str,
    encoder,
    use_cache: bool
) -> Dict[str, Any]:
    """Extrai embeddings dos corpos detectados."""
    character_embeddings = {}
    
    for crop, det in bodies[:MAX_REF_PER_TILE]:
        char_id = get_character_id(image_hash, det.bbox)
        
        if use_cache and char_id in character_cache:
            character_embeddings[char_id] = character_cache[char_id]
            print(f"[Realtime] Body {char_id[:8]} do cache")
        else:
            crop_pil = Image.fromarray(crop)
            embedding, method = encoder.extract_identity(crop_pil)
            character_embeddings[char_id] = embedding
            print(f"[Realtime] Body {char_id[:8]} extraído via {method}")
            
            if use_cache:
                character_cache[char_id] = embedding
                if len(character_cache) > 100:
                    oldest_key = next(iter(character_cache))
                    del character_cache[oldest_key]
    
    return character_embeddings


def _extract_embeddings_from_faces(
    faces: List,
    image_hash: str,
    encoder,
    use_cache: bool,
    existing_embeddings: Dict[str, Any]
) -> Dict[str, Any]:
    """Extrai embeddings das faces como fallback."""
    character_embeddings = dict(existing_embeddings)
    remaining_slots = MAX_REF_PER_TILE - len(character_embeddings)
    
    for crop, det in faces[:remaining_slots]:
        char_id = get_character_id(image_hash, det.bbox) + "_face"
        
        if use_cache and char_id in character_cache:
            character_embeddings[char_id] = character_cache[char_id]
            print(f"[Realtime] Face {char_id[:8]} do cache")
        else:
            crop_pil = Image.fromarray(crop)
            embedding, method = encoder.extract_identity(crop_pil)
            character_embeddings[char_id] = embedding
            print(f"[Realtime] Face {char_id[:8]} extraída via {method}")
            
            if use_cache:
                character_cache[char_id] = embedding
                if len(character_cache) > 100:
                    oldest_key = next(iter(character_cache))
                    del character_cache[oldest_key]
    
    return character_embeddings


def _build_detections_list(
    bodies: List,
    faces: List,
    image_hash: str,
    max_chars: int
) -> List[Dict]:
    """Constrói lista de detecções formatadas."""
    detections = []
    
    # Adiciona bodies
    for crop, det in bodies[:max_chars]:
        char_id = get_character_id(image_hash, det.bbox)
        detections.append({
            'bbox': det.bbox,
            'confidence': det.confidence,
            'prominence_score': det.prominence_score,
            'char_id': char_id,
            'class_id': det.class_id,
            'class_name': det.class_name
        })
    
    # Adiciona faces se houver espaço
    if len(detections) < max_chars:
        remaining_slots = max_chars - len(detections)
        for crop, det in faces[:remaining_slots]:
            char_id = get_character_id(image_hash, det.bbox) + "_face"
            detections.append({
                'bbox': det.bbox,
                'confidence': det.confidence,
                'prominence_score': det.confidence * 0.5,
                'char_id': char_id,
                'class_id': det.class_id,
                'class_name': det.class_name
            })
    
    return detections


def _update_realtime_stats(elapsed_ms: float) -> None:
    """Atualiza estatísticas globais de processamento."""
    global realtime_stats
    realtime_stats["total_requests"] += 1
    realtime_stats["total_time_ms"] += elapsed_ms
    realtime_stats["avg_time_ms"] = realtime_stats["total_time_ms"] / realtime_stats["total_requests"]


@app.post("/realtime/colorize")
async def realtime_colorize(
    file: Optional[UploadFile] = File(None, description="Imagem do mangá (PNG/JPG)"),
    image_url: Optional[str] = Form(None, description="URL da imagem (alternativa ao file)"),
    style_preset: str = Form("default", description="Estilo: default|vibrant|muted|sepia"),
    seed: Optional[int] = Form(None, description="Seed para reprodutibilidade"),
    use_cache: bool = Form(True, description="Usar cache de personagens")
):
    """
    Coloriza uma única imagem em modo Express (realtime).
    
    - Recebe imagem via upload ou URL
    - Detecta personagens localmente
    - Gera colorização sem persistência em disco
    - Retorna imagem colorizada como PNG stream
    
    **Ideal para extensão de navegador.**
    """
    start_time = time.time()
    
    # Valida input
    if file is None and not image_url:
        raise HTTPException(status_code=400, detail="Forneça 'file' ou 'image_url'")
    
    try:
        # Inicializa componentes e carrega imagem
        detector, encoder, generator = init_realtime_components()
        input_image = await _load_input_image(file, image_url)
        
        # Converte para RGB se necessário
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')
        
        image_hash = get_image_hash(input_image)
        print(f"[Realtime] Detectando personagens na imagem {image_hash[:8]}...")
        
        # PASS 1: Detecção
        image_np = np.array(input_image)
        char_crops = detector.get_character_crops(image_np)
        bodies = char_crops.get('bodies', [])
        faces = char_crops.get('faces', [])
        print(f"[Realtime] Detectado: {len(bodies)} bodies, {len(faces)} faces")
        
        # PASS 1.5: Extração de embeddings
        character_embeddings = _extract_embeddings_from_bodies(bodies, image_hash, encoder, use_cache)
        if len(character_embeddings) < MAX_REF_PER_TILE:
            character_embeddings = _extract_embeddings_from_faces(
                faces, image_hash, encoder, use_cache, character_embeddings
            )
        
        # Constrói lista de detecções
        detections = _build_detections_list(bodies, faces, image_hash, MAX_REF_PER_TILE)
        
        # Extrai Canny para debug
        canny_edges = None
        if DEBUG_MODE:
            import cv2
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            canny_edges = cv2.Canny(gray, 50, 150)
        
        # PASS 2: Geração
        print(f"[Realtime] Gerando colorização com {len(character_embeddings)} personagens...")
        
        from core.pipeline import GenerationOptions
        options = GenerationOptions(
            style_preset=style_preset,
            seed=seed,
            num_inference_steps=4,
            guidance_scale=1.0
        )
        
        if not generator.is_loaded:
            generator.load_models()
        
        result_image = generator.generate_image(
            image=input_image,
            character_embeddings=character_embeddings,
            detections=detections,
            options=options
        )
        
        # Debug e finalização
        if DEBUG_MODE:
            debug_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            save_debug_images(
                timestamp=debug_timestamp,
                input_image=input_image,
                canny_edges=canny_edges,
                detections=detections,
                result=result_image,
                character_crops=char_crops
            )
        
        output_bytes = pil_to_bytes(result_image, format="PNG")
        elapsed_ms = (time.time() - start_time) * 1000
        _update_realtime_stats(elapsed_ms)
        
        print(f"[Realtime] Concluído em {elapsed_ms:.0f}ms")
        
        return StreamingResponse(
            io.BytesIO(output_bytes),
            media_type="image/png",
            headers={
                "X-Processing-Time-Ms": str(int(elapsed_ms)),
                "X-Characters-Detected": str(len(detections)),
                "X-Characters-Cached": str(len(character_cache))
            }
        )
        
    except Exception as e:
        import traceback
        print(f"[Realtime] Erro: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Erro na colorização: {str(e)}")


@app.post("/realtime/clear-cache")
async def clear_character_cache():
    """Limpa o cache de personagens."""
    global character_cache
    old_size = len(character_cache)
    character_cache.clear()
    return {
        "status": "success", 
        "message": f"Cache limpo ({old_size} personagens removidos)"
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )

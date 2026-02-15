"""
MangaAutoColor Pro - API Endpoints para Two-Pass System

Endpoints:
- POST /chapter/analyze     : Pass 1 - Análise completa
- POST /chapter/generate    : Pass 2 - Geração
- GET  /chapter/{id}/status : Status do processamento
- GET  /chapter/{id}/pages  : Lista páginas disponíveis
"""

import os
import uuid
import shutil
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from config.settings import DEVICE
from core.database.chapter_db import ChapterDatabase


router = APIRouter(prefix="/chapter", tags=["chapter"])

# Diretório para uploads temporários
UPLOAD_DIR = Path("./uploads/chapters")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Diretório para resultados
OUTPUT_DIR = Path("./output/chapters")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Executor para tarefas em background
executor = ThreadPoolExecutor(max_workers=2)

# Cache de status de processamento
chapter_status = {}


# ============================================================================
# Schemas
# ============================================================================

class ChapterStatusResponse(BaseModel):
    chapter_id: str
    status: str  # "pending", "analyzing", "analyzed", "generating", "completed", "error"
    progress: int
    total_pages: int
    message: str


class ChapterGenerateRequest(BaseModel):
    chapter_id: str
    page_numbers: Optional[List[int]] = None  # None = todas
    options: Optional[dict] = {}
    text_compositing: Optional[bool] = False  # Preservar texto original (pode causar artefatos)
    max_quality: Optional[bool] = True  # True = SKIP_FINAL_DOWNSCALE (máxima qualidade)
    style_preset: Optional[str] = "default"  # default|vibrant|muted|sepia|flashback|dream|nightmare


class ChapterResultResponse(BaseModel):
    chapter_id: str
    status: str
    output_dir: str
    generated_pages: List[str]


# ============================================================================
# Helper Functions
# ============================================================================

def save_uploaded_files(files: List[UploadFile], chapter_id: str) -> List[str]:
    """Salva arquivos uploadados e retorna lista de paths"""
    chapter_dir = UPLOAD_DIR / chapter_id
    chapter_dir.mkdir(exist_ok=True)
    
    paths = []
    for i, file in enumerate(sorted(files, key=lambda x: x.filename)):
        ext = Path(file.filename).suffix or ".png"
        file_path = chapter_dir / f"page_{i:03d}{ext}"
        
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        paths.append(str(file_path))
    
    return sorted(paths)


def save_color_references(files: List[UploadFile], chapter_id: str) -> List[str]:
    """Salva imagens de referência coloridas"""
    ref_dir = UPLOAD_DIR / chapter_id / "color_references"
    ref_dir.mkdir(parents=True, exist_ok=True)
    
    paths = []
    for i, file in enumerate(files):
        ext = Path(file.filename).suffix or ".png"
        file_path = ref_dir / f"ref_{i:03d}{ext}"
        
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        paths.append(str(file_path))
    
    return paths


from core.pipeline import MangaColorizationPipeline, ChapterAnalysis, GenerationOptions
import json

def save_chapter_status_to_disk(chapter_id: str, status_data: dict):
    """Persiste status do capítulo em disco para sobreviver a restarts"""
    try:
        status_file = UPLOAD_DIR / chapter_id / "status.json"
        if status_file.parent.exists():
            with open(status_file, "w") as f:
                json.dump(status_data, f, indent=2)
    except Exception as e:
        print(f"[API] Erro ao salvar status para {chapter_id}: {e}")

def load_chapter_status_from_disk(chapter_id: str) -> Optional[dict]:
    """Carrega status do disco se existir"""
    try:
        status_file = UPLOAD_DIR / chapter_id / "status.json"
        if status_file.exists():
            with open(status_file, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def run_pass1_analysis(chapter_id: str, page_paths: List[str], color_ref_paths: List[str] = None):
    """Executa Pass 1 em thread separada via Pipeline"""
    try:
        chapter_status[chapter_id] = {
            "status": "analyzing",
            "progress": 0,
            "total_pages": len(page_paths),
            "has_color_references": bool(color_ref_paths),
            "message": "Analisando páginas..."
        }
        
        def progress_callback(page_num, stage, progress):
            current = page_num + (1 if stage == "complete" else 0)
            total = len(page_paths)
            chapter_status[chapter_id]["progress"] = current
            chapter_status[chapter_id]["message"] = f"Analisando página {current}/{total} ({stage})..."
            # Otimização: Salvar em disco apenas a cada X updates ou mudanças de fase? 
            # Para simplificar, salva sempre (baixo overhead para poucos usuários)
            save_chapter_status_to_disk(chapter_id, chapter_status[chapter_id])
            
        save_chapter_status_to_disk(chapter_id, chapter_status[chapter_id])

        
        # Inicializa pipeline
        pipeline = MangaColorizationPipeline(device=DEVICE)
        pipeline.set_progress_callback(progress_callback)
        
        # Executa análise
        analysis = pipeline.process_chapter(
            page_paths=page_paths,
            color_reference_paths=color_ref_paths,
            chapter_id=chapter_id  # Passa ID externo para consistência com Pass 2
        )
        
        chapter_status[chapter_id].update({
            "status": "analyzed",
            "progress": len(page_paths),
            "message": f"Análise concluída. {analysis.num_characters} personagens detectados.",
            "summary": analysis.to_dict()
        })
        save_chapter_status_to_disk(chapter_id, chapter_status[chapter_id])

        
    except Exception as e:
        import traceback
        traceback.print_exc()
        chapter_status[chapter_id].update({
            "status": "error",
            "message": str(e)
        })
        save_chapter_status_to_disk(chapter_id, chapter_status[chapter_id])



def run_pass2_generation(chapter_id: str, page_numbers: Optional[List[int]], options: dict, text_compositing: bool = False, max_quality: bool = True, style_preset: str = "default"):
    """Executa Pass 2 em thread separada via Pipeline"""
    try:
        chapter_status[chapter_id].update({
            "status": "generating",
            "progress": 0,
            "message": "Iniciando geração..."
        })
        save_chapter_status_to_disk(chapter_id, chapter_status[chapter_id])

        
        output_dir = OUTPUT_DIR / chapter_id
        output_dir.mkdir(exist_ok=True)
        
        # Inicializa pipeline
        pipeline = MangaColorizationPipeline(device=DEVICE)
        
        # Prepara estado do pipeline manualmente (já que não estamos usando process_chapter aqui)
        # O pipeline precisa saber o chapter_id e page_paths
        # Hack: recarregar do banco ou apenas setar o ID se o pipeline suportar
        # O pipeline atual carrega do banco sob demanda se tivermos os paths configurados
        
        # Carrega DB para obter paths
        db = ChapterDatabase(chapter_id)
        if not db.exists():
            raise ValueError(f"Capítulo {chapter_id} não encontrado")
            
        summary = db.get_summary()
        # Precisamos configurar o pipeline com os paths corretos para ele funcionar
        # Mas o pipeline espera process_chapter ser chamado para configurar _page_paths
        # Vamos usar uma abordagem mais direta: iterar e chamar generate_page
        
        # Configura manualmente o pipeline para este capítulo
        pipeline._current_chapter_id = chapter_id
        pipeline._chapter_loaded = True # Força flag de carregado
        
        # Recria a lista de paths a partir do banco (se possível) ou do diretório de upload
        # O pipeline precisa de _page_paths para validação de range no generate_page
        upload_dir = UPLOAD_DIR / chapter_id
        
        # Debug: mostra o conteúdo do diretório
        print(f"[DEBUG] Upload dir: {upload_dir}")
        print(f"[DEBUG] Upload dir exists: {upload_dir.exists()}")
        
        if upload_dir.exists():
            all_files = list(upload_dir.glob("*"))
            print(f"[DEBUG] Arquivos no diretório: {[f.name for f in all_files[:10]]}")
        
        page_files = sorted(upload_dir.glob("page_*.png"))
        print(f"[DEBUG] Página files encontrados: {len(page_files)}")
        
        # Se não encontrar page_*.png, tenta outros padrões
        if not page_files:
            page_files = sorted(upload_dir.glob("*.png"))
            print(f"[DEBUG] Fallback *.png: {len(page_files)}")
        
        if not page_files:
            page_files = sorted(upload_dir.glob("*.jpg")) + sorted(upload_dir.glob("*.jpeg"))
            print(f"[DEBUG] Fallback *.jpg: {len(page_files)}")
        
        pipeline._page_paths = [str(p) for p in page_files]
        
        if not page_numbers:
            page_numbers = list(range(len(page_files)))
        
        print(f"[DEBUG] page_numbers: {page_numbers}")
        print(f"[DEBUG] total pages to generate: {len(page_numbers)}")
            
        generated_pages = []
        
        # Opções de geração
        gen_options = GenerationOptions(
            style_preset=style_preset,
            preserve_original_text=text_compositing,
            quality_mode="high" if max_quality else "balanced",
            # Outros mapeamentos...
        )
        if options:
            # Merge de opções extras se necessário
            pass

        total_pages = len(page_numbers)
        
        for i, page_num in enumerate(page_numbers):
            try:
                chapter_status[chapter_id]["message"] = f"Gerando página {i+1}/{total_pages}..."
                
                result = pipeline.generate_page(page_num, gen_options)
                
                # Salva
                output_path = output_dir / f"page_{page_num:03d}_colored.png"
                result.save(output_path)
                generated_pages.append(str(output_path))
                
                # Atualiza progresso
                chapter_status[chapter_id]["progress"] = int(((i + 1) / total_pages) * 100)
                save_chapter_status_to_disk(chapter_id, chapter_status[chapter_id])

                
            except Exception as e:
                print(f"Erro gerando página {page_num}: {e}")
                # Continua para próximas páginas
        
        chapter_status[chapter_id].update({
            "status": "completed",
            "progress": 100,
            "message": f"Geração concluída. {len(generated_pages)} páginas geradas.",
            "output_dir": str(output_dir),
            "generated_pages": generated_pages
        })
        save_chapter_status_to_disk(chapter_id, chapter_status[chapter_id])

        
    except Exception as e:
        import traceback
        traceback.print_exc()
        chapter_status[chapter_id].update({
            "status": "error",
            "message": str(e)
        })
        save_chapter_status_to_disk(chapter_id, chapter_status[chapter_id])



# ============================================================================
# Endpoints
# ============================================================================

@router.post("/analyze", response_model=ChapterStatusResponse)
async def analyze_chapter(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    chapter_id: Optional[str] = Form(None),
    color_references: Optional[List[UploadFile]] = File(None)
):
    """
    Pass 1: Análise completa do capítulo.
    
    Upload múltiplas imagens de páginas de mangá.
    Opcionalmente, forneça imagens de referência coloridas para extrair paletas.
    
    O sistema irá:
    1. Detectar todos os personagens
    2. Extrair embeddings de identidade
    3. Extrair paletas das imagens de referência (se fornecidas)
    4. Criar cache imutável
    
    Retorna chapter_id para usar no Pass 2.
    """
    if len(files) == 0:
        raise HTTPException(400, "Nenhum arquivo enviado")
    
    # Gera ID único se não fornecido
    if chapter_id is None:
        chapter_id = f"ch_{uuid.uuid4().hex[:12]}"
    
    # Verifica se já existe
    if chapter_id in chapter_status:
        raise HTTPException(400, f"Capítulo {chapter_id} já existe")
    
    # Salva arquivos das páginas
    try:
        page_paths = save_uploaded_files(files, chapter_id)
    except Exception as e:
        raise HTTPException(500, f"Erro ao salvar arquivos: {e}")
    
    # Salva imagens de referência coloridas (se houver)
    color_ref_paths = []
    if color_references:
        try:
            color_ref_paths = save_color_references(color_references, chapter_id)
            print(f"[API] {len(color_ref_paths)} imagens de referência salvas")
        except Exception as e:
            print(f"[API] Erro ao salvar referências: {e}")
    
    # Inicia análise em background
    chapter_status[chapter_id] = {
        "chapter_id": chapter_id,
        "status": "pending",
        "progress": 0,
        "total_pages": len(page_paths),
        "has_color_references": len(color_ref_paths) > 0,
        "message": "Aguardando início..."
    }
    save_chapter_status_to_disk(chapter_id, chapter_status[chapter_id])

    
    # Executa em thread separada (passa referências coloridas)
    executor.submit(run_pass1_analysis, chapter_id, page_paths, color_ref_paths)
    
    return ChapterStatusResponse(
        chapter_id=chapter_id,
        status="pending",
        progress=0,
        total_pages=len(page_paths),
        message="Análise iniciada em background" + (f" ({len(color_ref_paths)} refs)" if color_ref_paths else "")
    )


@router.post("/generate", response_model=ChapterStatusResponse)
async def generate_chapter(
    request: ChapterGenerateRequest,
    background_tasks: BackgroundTasks
):
    """
    Pass 2: Geração das páginas colorizadas.
    
    Requer que o Pass 1 tenha sido concluído.
    """
    chapter_id = request.chapter_id
    
    if chapter_id not in chapter_status:
        # Verifica se existe no disco
        db = ChapterDatabase(chapter_id)
        if not db.exists():
            raise HTTPException(404, f"Capítulo {chapter_id} não encontrado. Execute Pass 1 primeiro.")
    else:
        current_status = chapter_status[chapter_id].get("status")
        if current_status not in ["analyzed", "completed"]:
            raise HTTPException(400, f"Pass 1 não concluído. Status atual: {current_status}")
    
    # Inicia geração em background
    executor.submit(
        run_pass2_generation,
        chapter_id,
        request.page_numbers,
        request.options or {},
        request.text_compositing if request.text_compositing is not None else False,
        request.max_quality if request.max_quality is not None else True,  # Padrão: máxima qualidade
        request.style_preset if request.style_preset is not None else "default"  # Estilo de colorização
    )
    
    return ChapterStatusResponse(
        chapter_id=chapter_id,
        status="generating",
        progress=0,
        total_pages=chapter_status.get(chapter_id, {}).get("total_pages", 0),
        message="Geração iniciada em background"
    )


@router.get("/{chapter_id}/status", response_model=ChapterStatusResponse)
async def get_chapter_status(chapter_id: str):
    """Retorna status atual do processamento do capítulo"""
    
    status = {}
    if chapter_id in chapter_status:
        status = chapter_status[chapter_id]
    else:
        # Tenta carregar do disco (recuperação de falha/restart)
        disk_status = load_chapter_status_from_disk(chapter_id)
        if disk_status:
            chapter_status[chapter_id] = disk_status # Re-hidrata memória
            status = disk_status
        else:
             # Verifica se existe no disco (comportamento antigo)
            db = ChapterDatabase(chapter_id)
            if db.exists():
                summary = db.get_summary()
                return ChapterStatusResponse(
                    chapter_id=chapter_id,
                    status="analyzed",
                    progress=summary["total_pages"],
                    total_pages=summary["total_pages"],
                    message=f"Análise concluída. {summary['total_characters']} personagens."
                )
            raise HTTPException(404, f"Capítulo {chapter_id} não encontrado")

    return ChapterStatusResponse(
        chapter_id=chapter_id,
        status=status.get("status", "unknown"),
        progress=status.get("progress", 0),
        total_pages=status.get("total_pages", 0),
        message=status.get("message", "")
    )



@router.get("/{chapter_id}/pages")
async def list_chapter_pages(chapter_id: str):
    """Lista páginas disponíveis no capítulo"""
    db = ChapterDatabase(chapter_id)
    if not db.exists():
        raise HTTPException(404, f"Capítulo {chapter_id} não encontrado")
    
    db.load_all()
    summary = db.get_summary()
    
    return {
        "chapter_id": chapter_id,
        "total_pages": summary["total_pages"],
        "total_characters": summary["total_characters"],
        "pages": list(range(summary["total_pages"]))
    }


@router.get("/{chapter_id}/result/{page_num}")
async def get_page_result(chapter_id: str, page_num: int):
    """Retorna imagem gerada de uma página específica"""
    output_dir = OUTPUT_DIR / chapter_id
    
    # Procura arquivo da página
    patterns = [
        f"page_{page_num:03d}_colored.png",
        f"page_{page_num}_colored.png",
        f"{page_num:03d}_colored.png"
    ]
    
    for pattern in patterns:
        file_path = output_dir / pattern
        if file_path.exists():
            return FileResponse(str(file_path), media_type="image/png")
    
    raise HTTPException(404, f"Página {page_num} não encontrada para capítulo {chapter_id}")


@router.get("/{chapter_id}/download")
async def download_chapter(chapter_id: str):
    """Download de todas as páginas como ZIP"""
    output_dir = OUTPUT_DIR / chapter_id
    if not output_dir.exists():
        raise HTTPException(404, f"Resultados não encontrados para capítulo {chapter_id}")
    
    # Cria ZIP
    zip_path = output_dir / f"{chapter_id}_colored.zip"
    
    if not zip_path.exists():
        import zipfile
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for img_file in sorted(output_dir.glob("*_colored.png")):
                zf.write(img_file, img_file.name)
    
    return FileResponse(
        str(zip_path),
        media_type="application/zip",
        filename=f"{chapter_id}_colored.zip"
    )


@router.get("/{chapter_id}/colored-images")
async def get_colored_images(chapter_id: str):
    """
    Retorna imagens colorizadas como base64 para exibição no navegador.
    """
    output_dir = OUTPUT_DIR / chapter_id
    
    if not output_dir.exists():
        raise HTTPException(404, f"Capítulo {chapter_id} não encontrado")
    
    colored_images = sorted(output_dir.glob("*_colored.png"))
    
    if not colored_images:
        return {"images": [], "mappings": []}
    
    import base64
    
    images_data = []
    mappings = []
    
    for i, img_path in enumerate(colored_images):
        with open(img_path, 'rb') as f:
            img_bytes = f.read()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        images_data.append({
            "index": i,
            "filename": img_path.name,
            "data": img_base64
        })
        
        # Mapeamento: page_XXX_colored.png -> pagina X
        page_num = i  # Assume ordem sequencial
        mappings.append({
            "page": page_num,
            "filename": img_path.name
        })
    
    return {
        "images": images_data,
        "mappings": mappings,
        "total": len(images_data)
    }


@router.delete("/{chapter_id}")
async def delete_chapter(chapter_id: str):
    """Remove capítulo e todos os dados associados"""
    # Remove do status
    if chapter_id in chapter_status:
        del chapter_status[chapter_id]
    
    # Remove database
    db = ChapterDatabase(chapter_id)
    db.clear()
    
    # Remove arquivos
    for dir_path in [UPLOAD_DIR / chapter_id, OUTPUT_DIR / chapter_id]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
    
    return {"message": f"Capítulo {chapter_id} removido"}

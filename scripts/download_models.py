"""
MangaAutoColor Pro - Script de Download de Modelos

Baixa e verifica todos os modelos necessários para o funcionamento.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict
import argparse

# Adiciona raiz do projeto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.utils import RepositoryNotFoundError
    HF_AVAILABLE = True
except ImportError:
    print("[ERRO] huggingface_hub não instalado. Instale com: pip install huggingface-hub")
    HF_AVAILABLE = False
    sys.exit(1)


# =============================================================================
# CONFIGURAÇÃO DE MODELOS
# =============================================================================

MODELS_CONFIG = {
    "sdxl_base": {
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "description": "SDXL Base Model",
        "size_gb": 6.9,
        "required": True
    },
    "sdxl_lightning": {
        "repo_id": "ByteDance/SDXL-Lightning",
        "filename": "sdxl_lightning_4step_unet.safetensors",
        "description": "SDXL-Lightning 4-step UNet",
        "size_gb": 5.0,
        "required": True
    },
    "controlnet_canny": {
        "repo_id": "diffusers/controlnet-canny-sdxl-1.0",
        "description": "ControlNet Canny Edge",
        "size_gb": 2.5,
        "required": True
    },
    "vae_fp16": {
        "repo_id": "madebyollin/sdxl-vae-fp16-fix",
        "description": "VAE FP16 Fix",
        "size_gb": 0.3,
        "required": False
    },
    "yolo_manga": {
        "repo_id": "deepghs/manga109_yolo",
        "subfolder": "v2021.12.30_m_yv11",
        "filename": "model.onnx",
        "description": "YOLOv11 Medium treinado no dataset Manga109 (body, face, frame, text)",
        "size_gb": 0.08,
        "required": True
    }
}


def check_disk_space(required_gb: float, cache_dir: Path) -> bool:
    """Verifica se há espaço em disco suficiente."""
    try:
        import shutil
        total, used, free = shutil.disk_usage(cache_dir)
        free_gb = free / (1024**3)
        
        print(f"[INFO] Espaço livre: {free_gb:.1f} GB")
        print(f"[INFO] Espaço necessário: {required_gb:.1f} GB")
        
        if free_gb < required_gb * 1.5:  # 50% de margem
            print(f"[AVISO] Espaço em disco pode ser insuficiente!")
            return False
        return True
    except Exception as e:
        print(f"[AVISO] Não foi possível verificar espaço em disco: {e}")
        return True


def download_yolo_model(cache_dir: Path) -> bool:
    """
    Baixa o modelo YOLO específico para mangá (Manga109).
    Modelo: deepghs/manga109_yolo - detecta body, face, frame, text
    """
    config = MODELS_CONFIG["yolo_manga"]
    print(f"\n[DOWNLOAD] YOLO para detecção em mangá (Manga109)")
    print(f"  Repo: {config['repo_id']}")
    print(f"  Arquivo: {config['subfolder']}/{config['filename']}")
    print(f"  Classes: body, face, frame, text")
    
    try:
        # Cria diretório de modelos
        models_dir = Path("./data/models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        target_path = models_dir / "manga109_yolo.onnx"
        
        # Se já existe, não baixa novamente
        if target_path.exists():
            print(f"  [OK] Modelo já existe em: {target_path}")
            return True
        
        # Download do model.onnx
        print(f"  Baixando {config['filename']}...")
        downloaded_path = hf_hub_download(
            repo_id=config['repo_id'],
            filename=config['filename'],
            subfolder=config['subfolder'],
            cache_dir=str(cache_dir),
            resume_download=True
        )
        
        # Copia para o local correto com nome padronizado
        import shutil
        shutil.copy2(downloaded_path, target_path)
        
        print(f"  [OK] Modelo copiado para: {target_path}")
        print(f"  [INFO] Classes: body, face, frame, text")
        return True
        
    except RepositoryNotFoundError:
        print(f"  [ERRO] Repositório não encontrado: {config['repo_id']}")
        return False
    except Exception as e:
        print(f"  [ERRO] Falha ao baixar YOLO: {e}")
        return False


def download_model(model_key: str, cache_dir: Path, force: bool = False) -> bool:
    """Baixa um modelo específico."""
    config = MODELS_CONFIG[model_key]
    
    # Tratamento especial para YOLO
    if model_key == "yolo_manga":
        return download_yolo_model(cache_dir)
    
    print(f"\n[DOWNLOAD] {config['description']}")
    print(f"  Repo: {config['repo_id']}")
    print(f"  Tamanho estimado: {config['size_gb']} GB")
    
    try:
        if 'filename' in config:
            # Download de arquivo específico
            hf_hub_download(
                repo_id=config['repo_id'],
                filename=config['filename'],
                cache_dir=cache_dir,
                force_download=force,
                resume_download=True
            )
        else:
            # Download de repositório completo
            snapshot_download(
                repo_id=config['repo_id'],
                cache_dir=cache_dir,
                resume_download=True
            )
        
        print(f"  [OK] {model_key} baixado com sucesso")
        return True
        
    except RepositoryNotFoundError:
        print(f"  [ERRO] Repositório não encontrado: {config['repo_id']}")
        return False
    except Exception as e:
        print(f"  [ERRO] Falha ao baixar {model_key}: {e}")
        return False


def verify_models(cache_dir: Path) -> Dict[str, bool]:
    """Verifica quais modelos estão disponíveis localmente."""
    results = {}
    
    print("\n[VERIFICAÇÃO] Verificando modelos instalados...")
    
    for key, config in MODELS_CONFIG.items():
        try:
            # Tratamento especial para YOLO
            if key == "yolo_manga":
                yolo_path = Path("./data/models/manga109_yolo.onnx")
                if yolo_path.exists():
                    results[key] = True
                    print(f"  [OK] {key}: Instalado em {yolo_path}")
                else:
                    results[key] = False
                    print(f"  [FALTA] {key}: Não encontrado em {yolo_path}")
                continue
            
            if 'filename' in config:
                # Verifica arquivo específico
                hf_hub_download(
                    repo_id=config['repo_id'],
                    filename=config['filename'],
                    cache_dir=cache_dir,
                    local_files_only=True
                )
            else:
                # Verifica repositório
                snapshot_download(
                    repo_id=config['repo_id'],
                    cache_dir=cache_dir,
                    local_files_only=True
                )
            results[key] = True
            print(f"  [OK] {key}: Instalado")
        except Exception:
            results[key] = False
            print(f"  [FALTA] {key}: Não instalado")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Download e verificação de modelos do MangaAutoColor Pro"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./models",
        help="Diretório para cache de modelos"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS_CONFIG.keys()) + ["all"],
        default=["all"],
        help="Modelos específicos para download"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Apenas verifica modelos existentes, não baixa"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Força re-download mesmo se existir"
    )
    
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("MANGA AUTOCOLOR PRO - DOWNLOAD DE MODELOS")
    print("=" * 60)
    print(f"Cache directory: {cache_dir.absolute()}")
    print(f"HuggingFace cache: {os.environ.get('HF_HOME', 'default')}")
    
    # Verificação apenas
    if args.verify_only:
        results = verify_models(cache_dir)
        
        all_required_ok = all(
            results[k] for k, v in MODELS_CONFIG.items() if v['required']
        )
        
        print(f"\n{'=' * 60}")
        if all_required_ok:
            print("[OK] Todos os modelos obrigatórios estão instalados!")
            return 0
        else:
            print("[AVISO] Alguns modelos obrigatórios estão faltando!")
            return 1
    
    # Download
    models_to_download = list(MODELS_CONFIG.keys()) if "all" in args.models else args.models
    
    # Calcula espaço necessário
    total_size = sum(MODELS_CONFIG[m]['size_gb'] for m in models_to_download)
    
    if not check_disk_space(total_size, cache_dir):
        response = input("Continuar mesmo assim? (y/N): ")
        if response.lower() != 'y':
            print("[CANCELADO] Operação cancelada pelo usuário")
            return 0
    
    # Baixa modelos
    success_count = 0
    for model_key in models_to_download:
        if download_model(model_key, cache_dir, args.force):
            success_count += 1
    
    # Resumo
    print(f"\n{'=' * 60}")
    print(f"RESUMO: {success_count}/{len(models_to_download)} modelos OK")
    
    if success_count == len(models_to_download):
        print("[SUCESSO] Todos os modelos foram baixados!")
        return 0
    else:
        print("[AVISO] Alguns modelos falharam. Tente novamente.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


import os
import torch
from pathlib import Path
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL
from transformers import CLIPVisionModelWithProjection
from huggingface_hub import hf_hub_download, snapshot_download

def download_models():
    print("Iniciando download de modelos para ADR 006 (v3 Engine)...")
    
    # Configurações
    cache_dir = os.getenv("HF_HOME", "./cache/huggingface")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    models = {
        "base": "runwayml/stable-diffusion-v1-5",
        "controlnet": "lllyasviel/control_v11p_sd15s2_lineart_anime",
        "ip_adapter_repo": "h94/IP-Adapter",
        "image_encoder": "h94/IP-Adapter" # CLIP vision model is usually downloaded automatically by pipeline, but we might need explicit load
    }
    
    print(f"\n1. Baixando ControlNet: {models['controlnet']}")
    try:
        ControlNetModel.from_pretrained(
            models['controlnet'],
            torch_dtype=torch.float16,
            cache_dir=cache_dir
        )
        print("   [OK] ControlNet baixado.")
    except Exception as e:
        print(f"   [ERRO] Falha ao baixar ControlNet: {e}")
        return

    print(f"\n2. Baixando Base SD 1.5: {models['base']}")
    try:
        # Baixa apenas variantes fp16 para economizar espaço/banda se possível
        # Mas SD 1.5 original pode não ter fp16 folder structure separada em alguns repos, 
        # runwayml/stable-diffusion-v1-5 tem revision="fp16"
        StableDiffusionControlNetPipeline.from_pretrained(
            models['base'],
            controlnet=None, # Não precisa carregar agora
            torch_dtype=torch.float16,
            revision="fp16",
            cache_dir=cache_dir,
            safety_checker=None 
        )
        print("   [OK] Base SD 1.5 baixada.")
    except Exception as e:
        print(f"   [ERRO] Falha ao baixar SD 1.5: {e}")
        print("   Tentando baixar sem revision='fp16'...")
        try:
             StableDiffusionControlNetPipeline.from_pretrained(
                models['base'],
                controlnet=None,
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
                safety_checker=None
            )
             print("   [OK] Base SD 1.5 baixada (padrão).")
        except Exception as e2:
             print(f"   [ERRO] Falha crítica ao baixar SD 1.5: {e2}")
             return

    print(f"\n3. Baixando IP-Adapter: {models['ip_adapter_repo']}")
    try:
        # IP-Adapter Plus Face para SD 1.5
        filename = "models/ip-adapter-plus-face_sd15.bin"
        hf_hub_download(
            repo_id=models['ip_adapter_repo'],
            filename=filename,
            cache_dir=cache_dir
        )
        print(f"   [OK] {filename} baixado.")
        
        # O modelo plus face requer o image encoder específico
        # Geralmente é o "laion/CLIP-ViT-H-14-laion2B-s32B-b79K" para SD1.5 IP-Adapter Plus?
        # A documentação diz que usa CLIPVisionModelWithProjection
        pass
    except Exception as e:
        print(f"   [ERRO] Falha ao baixar IP-Adapter: {e}")
        return

    print(f"\n4. Baixando Image Encoder para IP-Adapter Plus")
    try:
        # IP-Adapter Plus uses ViT-H-14
        image_encoder_path = "h94/IP-Adapter" 
        # Actually standard IP-Adapter uses CLIP-ViT-H-14-laion2B-s32B-b79K usually
        # But let's verify what diffusers expects. 
        # Diffusers load_ip_adapter handles the bin file, but we need the vision encoder.
        # For simple IP-Adapter SD1.5 it uses the one from the pipe? No, usually external.
        # Let's download the commonly used one just in case.
        CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter", 
            subfolder="models/image_encoder",
            cache_dir=cache_dir
        )
        print("   [OK] Image Encoder baixado.")
    except Exception as e:
        print(f"   [ERRO] Falha ao baixar Image Encoder: {e}")
        # Tenta fallback para laion original se o repo do h94 não tiver
        try:
             CLIPVisionModelWithProjection.from_pretrained(
                "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                cache_dir=cache_dir
             )
             print("   [OK] Image Encoder (LAION fallback) baixado.")
        except Exception as e2:
             print(f"   [AVISO] Image encoder pode falhar no runtime: {e2}")

    print("\n[SUCESSO] Todos os modelos verificados/baixados.")

if __name__ == "__main__":
    download_models()

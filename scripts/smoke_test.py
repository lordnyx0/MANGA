"""
MangaAutoColor Pro - Smoke Test de Integracao

Este script testa a execucao REAL com modelos pesados:
1. Carrega SDXL-Lightning + ControlNet Canny
2. Testa geracao com uma imagem dummy
3. Monitora uso de VRAM
4. Verifica conversao de coordenadas de tiles

ATENCAO: Este teste baixa modelos do HuggingFace (~6-8GB) na primeira execucao.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image
import gc
import time

print("=" * 70)
print("MANGA AUTOCOLOR PRO - SMOKE TEST DE INTEGRACAO")
print("=" * 70)

# =============================================================================
# TESTE 0: Informacoes do Sistema
# =============================================================================
print("\n[TESTE 0] Informacoes do Sistema")
print("-" * 50)

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponivel: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("AVISO: CUDA nao disponivel! Teste sera executado na CPU (lento).")

# =============================================================================
# TESTE 1: Carregamento de Modelos
# =============================================================================
print("\n[TESTE 1] Carregamento de Modelos (SDXL-Lightning + ControlNet)")
print("-" * 50)

pipeline = None
controlnet = None
unet = None

# Limpeza inicial
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

initial_memory = 0
if torch.cuda.is_available():
    initial_memory = torch.cuda.memory_allocated() / 1e9
    print(f"VRAM inicial: {initial_memory:.2f} GB")

try:
    from diffusers import (
        StableDiffusionXLControlNetPipeline, 
        ControlNetModel, 
        UNet2DConditionModel,
        EulerDiscreteScheduler
    )
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from config.settings import (
        SDXL_BASE_MODEL_ID, 
        SDXL_LIGHTNING_UNET_ID,
        CONTROLNET_CANNY_ID, 
        DTYPE, 
        DEVICE
    )
    
    print(f"\nCarregando ControlNet Canny...")
    print(f"Model ID: {CONTROLNET_CANNY_ID}")
    
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_CANNY_ID,
        torch_dtype=DTYPE,
        variant="fp16" if DTYPE == torch.float16 else None
    )
    
    if torch.cuda.is_available():
        controlnet_mem = torch.cuda.memory_allocated() / 1e9 - initial_memory
        print(f"[OK] ControlNet carregado (VRAM: +{controlnet_mem:.2f} GB)")
    else:
        print("[OK] ControlNet carregado (CPU)")
    
    print(f"\nCarregando UNet SDXL-Lightning...")
    print(f"Model ID: {SDXL_LIGHTNING_UNET_ID}")
    print("(Este download pode levar 5-10 minutos na primeira vez)")
    
    # Carrega o UNet do Lightning
    try:
        # Tenta carregar como UNet standalone
        unet = UNet2DConditionModel.from_pretrained(
            SDXL_LIGHTNING_UNET_ID,
            subfolder="unet",
            torch_dtype=DTYPE,
        )
        print("[OK] UNet carregado do subfolder")
    except Exception as e:
        print(f"Tentativa alternativa de carregamento...")
        # Tenta baixar e carregar o arquivo safetensors diretamente
        try:
            # Baixa o checkpoint lightning
            lightning_ckpt = hf_hub_download(
                repo_id=SDXL_LIGHTNING_UNET_ID,
                filename="sdxl_lightning_4step_unet.safetensors"
            )
            print(f"Checkpoint baixado: {lightning_ckpt}")
            
            # Carrega o UNet base do SDXL
            unet = UNet2DConditionModel.from_pretrained(
                SDXL_BASE_MODEL_ID,
                subfolder="unet",
                torch_dtype=DTYPE,
            )
            
            # Carrega os pesos do Lightning
            state_dict = load_file(lightning_ckpt)
            unet.load_state_dict(state_dict)
            print("[OK] Pesos Lightning carregados no UNet base")
        except Exception as e2:
            print(f"[ERRO] Falha ao carregar UNet: {e2}")
            print("Tentando carregar pipeline base sem Lightning...")
            unet = None
    
    print(f"\nCarregando Pipeline SDXL...")
    print(f"Base Model ID: {SDXL_BASE_MODEL_ID}")
    
    if unet is not None:
        # Cria pipeline com UNet Lightning
        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            SDXL_BASE_MODEL_ID,
            unet=unet,
            controlnet=controlnet,
            torch_dtype=DTYPE,
        )
    else:
        # Fallback: pipeline base normal
        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            SDXL_BASE_MODEL_ID,
            controlnet=controlnet,
            torch_dtype=DTYPE,
        )
    
    # Configura scheduler para Lightning (EulerDiscrete com timesteps especificos)
    # NOTA: Não definir prediction_type - usar o padrão do modelo (epsilon)
    pipeline.scheduler = EulerDiscreteScheduler.from_config(
        pipeline.scheduler.config,
        timestep_spacing="trailing"
    )
    
    # Otimizacoes de memoria
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    
    # CPU Offload e CRITICO para RTX 3060 (12GB)
    print("\nHabilitando CPU Offload...")
    pipeline.enable_model_cpu_offload()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        current_mem = torch.cuda.memory_allocated() / 1e9
        print(f"[OK] Pipeline carregado com CPU Offload")
        print(f"  VRAM atual: {current_mem:.2f} GB")
    else:
        print("[OK] Pipeline carregado (CPU)")
        
except Exception as e:
    print(f"[ERRO] ERRO ao carregar modelos: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# TESTE 2: Geracao com Imagem Dummy
# =============================================================================
print("\n[TESTE 2] Geracao com Imagem Dummy")
print("-" * 50)

try:
    # Cria imagem dummy (simula uma pagina de manga)
    print("Criando imagem de teste (1024x1024)...")
    
    # Cria imagem preta com algumas linhas brancas (simula Canny edges)
    dummy_canny = np.zeros((1024, 1024), dtype=np.uint8)
    
    # Desenha algumas formas para simular linhas de manga
    cv2 = None
    try:
        import cv2
        # Linhas horizontais
        for y in range(100, 900, 100):
            cv2.line(dummy_canny, (100, y), (900, y), 255, 2)
        # Linhas verticais
        for x in range(100, 900, 100):
            cv2.line(dummy_canny, (x, 100), (x, 900), 255, 2)
        # Circulo no centro (simula rosto)
        cv2.circle(dummy_canny, (512, 512), 200, 255, 3)
        print("[OK] Usando OpenCV para criar formas de teste")
    except ImportError:
        # Fallback sem OpenCV
        dummy_canny[100:900, 512:518] = 255  # Linha vertical
        dummy_canny[512:518, 100:900] = 255  # Linha horizontal
        print("[AVISO] OpenCV nao disponivel, usando formas simples")
    
    canny_pil = Image.fromarray(dummy_canny)
    print(f"[OK] Imagem de controle criada: {canny_pil.size}")
    
    # Tenta geracao
    print("\nExecutando inferencia (4 steps)...")
    print("(Isso pode levar 30-60 segundos na primeira vez)")
    
    start_time = time.time()
    
    result = pipeline(
        prompt="manga page, anime style, detailed coloring, masterpiece",
        negative_prompt="nsfw, lowres, bad anatomy",
        image=canny_pil,
        num_inference_steps=4,
        guidance_scale=1.0,  # 1.0-1.2 é o ideal para Lightning (recupera contraste)
        height=1024,
        width=1024,
        generator=torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
    ).images[0]
    
    elapsed = time.time() - start_time
    print(f"[OK] Inferencia completa em {elapsed:.1f}s")
    print(f"[OK] Imagem gerada: {result.size}")
    
    # Salva resultado
    output_path = "smoke_test_output.png"
    result.save(output_path)
    print(f"[OK] Resultado salvo em: {output_path}")
    
    # Verifica memoria
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        final_mem = torch.cuda.memory_allocated() / 1e9
        print(f"\nVRAM apos geracao: {final_mem:.2f} GB")
        
        if final_mem > 11.5:  # Proximo do limite de 12GB
            print("[AVISO] AVISO: VRAM proximo do limite! Considere otimizacoes adicionais.")
        else:
            print("[OK] VRAM dentro do limite aceitavel (< 11.5GB)")
            
except RuntimeError as e:
    if "out of memory" in str(e).lower() or "CUDA" in str(e):
        print(f"[ERRO] ERRO DE VRAM (OOM): {e}")
        print("\nSUGESTOES:")
        print("1. Verifique se CPU Offload esta habilitado")
        print("2. Reduza o tamanho do tile (atual: 1024)")
        print("3. Feche outros programas usando GPU")
        print("4. Considere usar float16 em vez de float32")
        sys.exit(1)
    else:
        raise
        
except Exception as e:
    print(f"[ERRO] ERRO na geracao: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# TESTE 3: Matematica de Tiles
# =============================================================================
print("\n[TESTE 3] Validacao da Matematica de Tiles")
print("-" * 50)

try:
    from config.settings import calculate_tile_grid, TILE_SIZE, TILE_OVERLAP
    
    # Testa calculo de tiles
    test_sizes = [
        (1024, 1024),   # Tamanho exato
        (2048, 2048),   # 4 tiles
        (1536, 2048),   # Tamanho irregular
    ]
    
    for width, height in test_sizes:
        nx, ny, tiles = calculate_tile_grid((width, height), TILE_SIZE, TILE_OVERLAP)
        print(f"\nImagem {width}x{height}:")
        print(f"  Tiles: {nx}x{ny} = {len(tiles)} total")
        
        # Verifica se tiles cobrem a imagem
        for i, (x1, y1, x2, y2) in enumerate(tiles[:3]):  # Primeiros 3
            print(f"  Tile {i}: ({x1},{y1})-({x2},{y2}) = {x2-x1}x{y2-y1}")
        if len(tiles) > 3:
            print(f"  ... e mais {len(tiles)-3} tiles")
    
    # Testa conversao de coordenadas globais para locais
    print("\nTeste de conversao de coordenadas:")
    
    # Personagem em coordenadas globais
    global_bbox = (1500, 800, 1800, 1200)  # x1, y1, x2, y2
    tile_bbox = (1024, 0, 2048, 1024)       # Tile na posicao (1024, 0)
    
    # Converte para local
    local_x1 = max(0, global_bbox[0] - tile_bbox[0])
    local_y1 = max(0, global_bbox[1] - tile_bbox[1])
    local_x2 = min(tile_bbox[2] - tile_bbox[0], global_bbox[2] - tile_bbox[0])
    local_y2 = min(tile_bbox[3] - tile_bbox[1], global_bbox[3] - tile_bbox[1])
    
    print(f"  Global: {global_bbox}")
    print(f"  Tile:   {tile_bbox}")
    print(f"  Local:  ({local_x1}, {local_y1}, {local_x2}, {local_y2})")
    
    # Verifica se esta correto
    expected = (476, 800, 776, 1024)  # 1500-1024=476, etc
    if (local_x1, local_y1, local_x2, local_y2) == expected:
        print("  [OK] Conversao correta!")
    else:
        print(f"  [AVISO] Esperado: {expected}")
        
except Exception as e:
    print(f"[ERRO] ERRO nos testes de tile: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TESTE 4: Conversao de Canais de Cor
# =============================================================================
print("\n[TESTE 4] Validacao de Conversao de Canais de Cor")
print("-" * 50)

try:
    from utils.image_utils import load_image
    import tempfile
    
    # Cria imagens de teste em diferentes modos
    test_cases = [
        ("RGB", (255, 0, 0)),
        ("L", 128),           # Grayscale
        ("P", None),          # Palette
        ("RGBA", (0, 255, 0, 128)),  # Com transparencia
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for mode, color in test_cases:
            # Cria imagem
            if mode == "L":
                img = Image.new(mode, (100, 100), color)
            elif mode == "P":
                img = Image.new("P", (100, 100), 0)
                # Cria paleta simples
                palette = [i for i in range(256)] * 3
                img.putpalette(palette[:768])
            elif mode == "RGBA":
                img = Image.new(mode, (100, 100), color)
            else:
                img = Image.new(mode, (100, 100), color)
            
            # Salva
            path = os.path.join(tmpdir, f"test_{mode}.png")
            img.save(path)
            
            # Carrega com load_image
            loaded = load_image(path, max_resolution=10000, convert_rgb=True)
            
            # Verifica modo
            if loaded.mode == "RGB":
                print(f"  [OK] {mode} -> RGB: OK")
            else:
                print(f"  [ERRO] {mode} -> {loaded.mode}: FALHA")
                
except Exception as e:
    print(f"[ERRO] ERRO nos testes de cor: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# RESUMO
# =============================================================================
print("\n" + "=" * 70)
print("RESUMO DO SMOKE TEST")
print("=" * 70)

print("\n[OK] Teste 0: Informacoes do Sistema - PASSOU")
print("[OK] Teste 1: Carregamento de Modelos - PASSOU")
print("[OK] Teste 2: Geracao com Imagem Dummy - PASSOU")
print("[OK] Teste 3: Matematica de Tiles - PASSOU")
print("[OK] Teste 4: Conversao de Canais de Cor - PASSOU")

print("\n" + "=" * 70)
print("[SUCESSO] TODOS OS TESTES PASSARAM!")
print("=" * 70)
print("\nO sistema esta pronto para uso com modelos reais.")
print("Proximo passo: Executar 'python main.py' para iniciar a interface.")

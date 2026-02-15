# 🛠️ Guia de Instalação - MangaAutoColor Pro v3.0

## Requisitos de Sistema

### Hardware (Validado)

| Componente | Mínimo | Recomendado | Validado |
|------------|--------|-------------|----------|
| GPU | NVIDIA GTX 1070 8GB | NVIDIA RTX 3060 12GB | RTX 3060 ✅ |
| VRAM | 6 GB | 8+ GB | 8.0 GB (Peak) |
| RAM | 16 GB | 32 GB | - |
| CPU | 6 cores | 8+ cores | - |
| Armazenamento | 10 GB SSD | 50 GB NVMe | - |
| Internet | 10 Mbps | 50+ Mbps | - |

### Software (Validado)

- **OS**: Windows 10/11, Linux (Ubuntu 20.04+)
- **Python**: 3.10, 3.11 (recomendado)
- **CUDA**: 11.8, 12.1 (recomendado)
- **PyTorch**: 2.2.x, 2.3.x (compatível com CUDA 11.8/12.1)

> ⚠️ **Nota**: Use Python 3.10 ou 3.11 para melhor compatibilidade com xformers e insightface.

---

## 🚀 Instalação Rápida

### 1. Clone o Repositório

```bash
git clone https://github.com/seu-usuario/manga-autocolor-pro.git
cd manga-autocolor-pro
```

### 2. Instalação Automática (Windows)

Basta executar o script de instalação, que cria o ambiente virtual, instala dependências e baixa os modelos:

```batch
scripts\windows\install.bat
```

### 3. Iniciar Servidor

```batch
scripts\windows\run.bat
```

---

## ✅ Verificação Completa

### Smoke Test (Recomendado)

Execute o teste de integração que carrega modelos reais e valida o pipeline:

```bash
python scripts/smoke_test.py
```

**Saída esperada (RTX 3060):**
```
✅ CUDA disponível
✅ Modelos baixados
✅ SD 1.5 carregado
✅ ControlNet carregado
✅ IP-Adapter carregado
✅ VAE configurado
✅ Pipeline compilado
✅ Teste de matemática de tiles
✅ Teste de conversão de cores
✅ Pipeline de geração (1024x1024)

🎉 Todos os testes passaram!

⚡ Métricas:
   - Tempo de inferência: ~25s (20 steps)
   - VRAM após geração: 0.1GB
   - Status: OK
```

### Teste de Unidade

```bash
# Todos os testes
pytest tests/unit -v
```

---

## 📦 Download de Modelos

### Download Automático

O `scripts\windows\install.bat` já executa o script de download. Se precisar rodar manualmente:

```bash
python scripts/download_models_v3.py
```

**Modelos V3 (~5 GB):**
| Modelo | Tamanho | Uso |
|--------|---------|-----|
| runwayml/stable-diffusion-v1-5 | ~2.5 GB | Base Model |
| lllyasviel/control_v11p_sd15s2_lineart_anime | ~1.4 GB | Lineart Control |
| h94/IP-Adapter (Plus Face SD15) | ~0.5 GB | Identidade |
| keremberke/yolov8m-manga-10k | ~50 MB | Detecção |
| openai/clip-vit-large-patch14 | ~1.5 GB | Encoder |

---

## 🔧 Configuração de Hardware

### RTX 3060 12GB (Configuração Padrão)

O sistema já vem otimizado para RTX 3060:

```python
# config/settings.py (valores padrão)
DTYPE = torch.float16              # FP16 obrigatório
ENABLE_CPU_OFFLOAD = True          # Economia de VRAM
TILE_SIZE = 1024                   # Tamanho do tile (SD 1.5 nativo é 512, mas suporta 1024 com tiling)
MAX_REF_PER_TILE = 2               # Limite de personagens
```

**Resultado medido:**
- VRAM Pico durante geração: **~5.5 GB** (CPU Offload)
- Tempo 1024×1408: **~25s**

### Outras GPUs

Para GPUs com mais VRAM (16GB+), desative CPU offload para mais velocidade:

```python
# RTX 4090 24GB
ENABLE_CPU_OFFLOAD = False
```

---

## 🐛 Solução de Problemas

### Problema: `CUDA out of memory`

**Causa**: VRAM insuficiente ou modelos muito grandes.

**Solução:**
```python
# Em config/settings.py
ENABLE_CPU_OFFLOAD = True    # Já ativado por padrão
TILE_SIZE = 768              # Reduzir tile
MAX_REF_PER_TILE = 1         # Limitar personagens
```

### Problema: `torch.cuda.OutOfMemoryError` durante download

**Causa**: Tentativa de carregar modelo em VRAM cheia.

**Solução:**
```bash
# Limpe VRAM
python -c "import torch; torch.cuda.empty_cache()"

# Ou reinicie o terminal
```

### Problema: `ModuleNotFoundError: No module named 'diffusers'`

**Solução:**
```bash
# Reinstale dependências
pip install -r requirements.txt --force-reinstall
```

### Problema: Modelos não baixam (timeout)

**Solução:**
```bash
# Use mirror alternativo
export HF_ENDPOINT=https://hf-mirror.com

# Ou configure proxy
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080

# Download manual
python scripts/download_models.py --retry 5
```

### Problema: `insightface` não instala no Windows

**Solução:**
```bash
# Insightface é opcional (para ArcFace)
# Se falhar, o sistema usa apenas CLIP
pip install insightface --pre

# Ou ignore o erro - CLIP é suficiente
```

### Problema: `RuntimeError: CUDA error: invalid device ordinal`

**Causa**: GPU não detectada.

**Solução:**
```bash
# Verifique CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Se False, reinstale PyTorch com CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## ⚙️ Configuração Avançada

### Variáveis de Ambiente

```bash
# Windows (PowerShell)
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512"
$env:HF_HOME = "C:\Models\HuggingFace"
$env:HF_HUB_DISABLE_SYMLINKS = "1"

# Linux/Mac
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export HF_HOME="/path/to/models"
```

### Configuração por Arquivo

Crie `config/local.yaml` (opcional):

```yaml
performance:
  device: "cuda"
  dtype: "float16"
  enable_cpu_offload: true
  tile_size: 1024
  max_ref_per_tile: 2

models:
  sdxl_model: "ByteDance/SDXL-Lightning"
  sdxl_steps: 4
  controlnet: "diffusers/controlnet-canny-sdxl-1.0"
  yolo_model: "keremberke/yolov8m-manga-10k"

generation:
  ip_adapter_end_step: 0.6
  background_ip_scale: 0.0
  context_inflation: 1.5
```

---

## 🎯 Execução

### Modo CLI

```bash
# Pipeline completo
python cli.py full ./chapter_01 --output ./output --style vibrant

# Apenas análise
python cli.py analyze ./chapter_01

# Apenas geração
python cli.py generate --chapter-id <id> --pages 1,2,3
```

### Modo API

```bash
# Iniciar servidor
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Testar
curl http://localhost:8000/health
```

### Modo Python

```python
from core.pipeline import MangaColorizationPipeline

pipeline = MangaColorizationPipeline()
chapter_id, summary = pipeline.process_chapter("./chapter_01")

for result in pipeline.generate_chapter(chapter_id):
    result.image.save(f"output/page_{result.page_number:03d}.png")
```

---

## 📊 Benchmark

```bash
python scripts/benchmark.py
```

**Resultados esperados:**

| Hardware | Análise | Geração 1024² | VRAM Pico |
|----------|---------|---------------|-----------|
| RTX 3060 12GB | ~2s/página | ~30s | ~11.5GB |
| RTX 4090 24GB | ~0.8s | ~8s | ~18GB |
| CPU (8 cores) | ~10s | ~300s | ~8GB RAM |

---

## 🆘 Suporte

### Informações para Debug

Se precisar de ajuda, execute:

```bash
# Coleta informações do sistema
python -c "
import torch
import sys
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB' if torch.cuda.is_available() else 'N/A')
"
```

### Logs de Erro

Habilite logs detalhados:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 🎉 Próximos Passos

1. ✅ **Execute o smoke test**: `python scripts/smoke_test.py`
2. 📚 **Leia a [API Reference](API.md)**
3. 🏗️ **Explore a [Arquitetura](ARCHITECTURE.md)**
4. 🧪 **Execute testes**: `pytest tests/high/ -v`

**Bem-vindo ao MangaAutoColor Pro v2.0!** 🎨

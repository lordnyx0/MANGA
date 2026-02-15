# 📚 API Reference - MangaAutoColor Pro v2.0

## Interface Principal

### MangaColorizationPipeline

Classe principal que orquestra o sistema Two-Pass com Tile-Aware Generation.

```python
from core.pipeline import MangaColorizationPipeline

pipeline = MangaColorizationPipeline(
    device="cuda",           # "cuda" ou "cpu"
    dtype=torch.float16,     # torch.float16 ou torch.float32
    cache_dir="./cache",     # Diretório de cache
    enable_cpu_offload=True  # Otimização para RTX 3060
)
```

#### Métodos

##### `process_chapter(chapter_path: Path) -> Tuple[str, ChapterSummary]`

Executa o **Passo 1: Análise Completa** em todas as páginas do capítulo.

```python
from pathlib import Path

# Análise de capítulo
chapter_path = Path("chapter_01/")
chapter_id, summary = pipeline.process_chapter(chapter_path)

print(f"Chapter ID: {chapter_id}")
print(f"Páginas: {summary.num_pages}")
print(f"Personagens: {summary.num_characters}")

# Informações dos personagens
for char in summary.characters:
    print(f"  {char['name']}: {char['appearances']} aparições")
```

**Retorno:**
- `chapter_id`: ID único do capítulo para geração
- `ChapterSummary`: Resumo com metadados do capítulo

**Exceções:**
- `AnalysisError`: Erro durante análise
- `ValueError`: Diretório inválido ou vazio

---

##### `generate_page(chapter_id: str, page_number: int, style_preset: str = "default") -> GeneratedPage`

Executa o **Passo 2: Geração Tile-Aware** para uma página específica.

```python
# Gera página 3
result = pipeline.generate_page(chapter_id, page_number=3)

# Acesso aos resultados
result.image.save("output/page_003.png")
print(f"Tempo de inferência: {result.inference_time:.1f}s")
print(f"VRAM usada: {result.vram_used_gb:.2f}GB")

# Com preset de estilo
result = pipeline.generate_page(
    chapter_id, 
    page_number=5,
    style_preset="vibrant"  # "default", "vibrant", "pastel", "cyberpunk"
)
```

**Parâmetros:**
- `chapter_id`: ID retornado por `process_chapter()`
- `page_number`: Número da página (1-based)
- `style_preset`: Estilo visual (ver `StylePreset`)

**Retorno:**
- `GeneratedPage`: Imagem + metadados da geração

---

##### `generate_chapter(chapter_id: str, page_numbers: List[int] = None, style_preset: str = "default") -> Generator[GeneratedPage, None, None]`

Gera múltiplas páginas com streaming de progresso.

```python
# Gera todas as páginas
for result in pipeline.generate_chapter(chapter_id):
    result.image.save(f"output/page_{result.page_number:03d}.png")
    print(f"Progresso: {result.page_number}/{result.total_pages}")

# Gera páginas específicas (não-linear)
pages_to_generate = [1, 3, 5, 10, 8]  # Qualquer ordem
for result in pipeline.generate_chapter(chapter_id, pages=pages_to_generate):
    print(f"Gerada página {result.page_number}")
```

---

##### `get_available_presets() -> Dict[str, StylePresetInfo]`

Retorna presets de estilo disponíveis.

```python
presets = pipeline.get_available_presets()
for name, info in presets.items():
    print(f"{name}: {info.description}")
    print(f"  - Sampler: {info.sampler}")
    print(f"  - Steps: {info.steps}")
```

---

## CLI (Command Line Interface)

### Comandos Principais

```bash
# Pipeline completo (analyze + generate)
python cli.py full ./chapter_01 --output ./output --style vibrant

# Apenas análise
python cli.py analyze ./chapter_01 --db ./data

# Apenas geração
python cli.py generate --chapter-id <id> --pages 1,2,3 --output ./output

# Listar presets disponíveis
python cli.py list-styles

# Informações do sistema
python cli.py info
```

### Opções Comuns

```bash
# Progresso detalhado
python cli.py full ./chapter --output ./out --verbose

# Geração em batch
python cli.py full ./chapter --batch-size 4

# Limite de VRAM
python cli.py full ./chapter --vram-limit 10

# Dry run (sem salvar)
python cli.py full ./chapter --dry-run
```

---

## REST API (FastAPI)

### Endpoints

#### `GET /health`

Verifica status do serviço.

```bash
curl http://localhost:8000/health
```

**Resposta:**
```json
{
  "status": "healthy",
  "device": "cuda",
  "vram_total_gb": 12.0,
  "vram_free_gb": 10.5,
  "models_loaded": true
}
```

---

#### `POST /chapters/upload`

Upload de capítulo para análise.

```bash
curl -X POST http://localhost:8000/chapters/upload \
  -F "file=@chapter_01.zip" \
  -F "chapter_id=chapter_01"
```

**Resposta:**
```json
{
  "chapter_id": "chapter_01",
  "num_pages": 20,
  "status": "analyzing"
}
```

---

#### `POST /chapters/{chapter_id}/analyze`

Inicia análise do capítulo. Suporta upload de imagens de referência de cor.

```bash
# Análise simples
curl -X POST http://localhost:8000/chapters/chapter_01/analyze

# Com referências de cor
curl -X POST http://localhost:8000/chapters/chapter_01/analyze \
  -F "color_references=@char1_reference.png" \
  -F "color_references=@char2_reference.jpg"
```

**Parâmetros (multipart/form-data):**
- `color_references` (opcional): Arquivos de imagem com cores oficiais dos personagens. Cada imagem deve conter o personagem bem destacado. O sistema extrai a paleta real das imagens e ignora os STYLE_PRESETS durante a geração.

**Resposta:**
```json
{
  "job_id": "job_abc123",
  "status": "queued"
}
```

---

#### `POST /chapters/{chapter_id}/generate`

Inicia geração de páginas.

```bash
curl -X POST http://localhost:8000/chapters/chapter_01/generate \
  -H "Content-Type: application/json" \
  -d '{
    "page_numbers": [1, 2, 3],
    "style_preset": "vibrant",
    "quality": "high"
  }'
```

**Parâmetros:**
- `page_numbers`: Lista de números de páginas para gerar (1-based)
- `style_preset`: Estilo visual quando não há referências de cor (veja tabela abaixo)
- `quality`: Qualidade de geração ("fast", "balanced", "high")

**STYLE_PRESETS Disponíveis:**

| Preset | Descrição | Efeito |
|--------|-----------|--------|
| `default` | Padrão equilibrado | Cores naturais, realismo moderado |
| `vibrant` | Cores vibrantes | Saturação aumentada, mais dramático |
| `muted` | Tons suaves | Paleta desaturada, atmosférico |
| `sepia` | Tom sépia | Efeito vintage, marrom-acastanhado |
| `flashback` | Flashback | Preto e branco com toques de cor |
| `dream` | Sonho | Cores pastel etéreas, suave |
| `nightmare` | Pesadelo | Saturação reduzida, sombras profundas |

> **Nota:** Quando referências de cor são fornecidas na análise, os STYLE_PRESETS são ignorados e as cores reais das referências são usadas.

**Resposta:**
```json
{
  "job_id": "job_def456",
  "status": "queued",
  "estimated_time_seconds": 90
}
```

---

#### `GET /jobs/{job_id}`

Consulta status de um job.

```bash
curl http://localhost:8000/jobs/job_abc123
```

**Resposta:**
```json
{
  "job_id": "job_abc123",
  "status": "running",
  "progress": 45,
  "message": "Processando página 9/20",
  "result": null
}
```

---

#### `GET /outputs/{chapter_id}/{filename}`

Download de resultado gerado.

```bash
curl http://localhost:8000/outputs/chapter_01/page_001.png \
  --output page_001.png
```

---

#### `GET /chapters/{chapter_id}/logs`

Download dos logs de geração em formato ZIP.

```bash
curl http://localhost:8000/chapters/chapter_01/logs \
  --output chapter_01_logs.zip
```

**Conteúdo do ZIP:**
- `generation_log.json`: Log completo em JSON com timeline, prompts, erros
- `prompts_used.txt`: Lista formatada de todos os prompts
- `timeline.txt`: Timeline formatada de execução

**Exemplo de `generation_log.json`:**
```json
{
  "chapter_id": "ch_d8d4c0757039",
  "start_time": "2025-02-07T01:15:23.123456",
  "end_time": "2025-02-07T01:18:45.654321",
  "steps": [
    {
      "step_name": "page_001_tile_0",
      "start_time": "2025-02-07T01:15:23.123456",
      "end_time": "2025-02-07T01:16:45.123456",
      "status": "success",
      "prompt": "colorful manga illustration...",
      "negative_prompt": "black and white, grayscale...",
      "metadata": {
        "tile_bbox": [0, 0, 1024, 1024],
        "regional_ip_active": true,
        "num_characters": 2
      }
    }
  ],
  "errors": []
}
```

---

### Iniciar Servidor API

```bash
# Desenvolvimento
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Produção
uvicorn api.main:app --workers 1 --host 0.0.0.0 --port 8000
```

---

## Classes de Dados

### ChapterSummary

```python
@dataclass
class ChapterSummary:
    chapter_id: str
    num_pages: int
    num_characters: int
    characters: List[CharacterInfo]
    total_appearances: int
    has_consolidated_data: bool
```

### CharacterInfo

```python
@dataclass
class CharacterInfo:
    char_id: str
    name: str
    appearances: int
    pages: List[int]
    prominence_score: float  # 0.0 - 1.0
    reference_embedding_path: Optional[str]
```

### GeneratedPage

```python
@dataclass
class GeneratedPage:
    image: Image.Image
    page_number: int
    total_pages: int
    inference_time: float  # segundos
    vram_used_gb: float
    style_preset: str
    character_ids: List[str]
```

### StylePreset

```python
@dataclass
class StylePreset:
    name: str
    description: str
    sampler: str
    steps: int
    cfg_scale: float
    prompt_prefix: str
    negative_prompt: str
```

---

## Configurações

### Settings (config/settings.py)

```python
from config.settings import Settings

settings = Settings()

# Hardware
settings.DEVICE              # "cuda" ou "cpu"
settings.DTYPE               # torch.float16
settings.ENABLE_CPU_OFFLOAD  # True (recomendado para RTX 3060)

# Tiling
settings.TILE_SIZE           # 1024
settings.TILE_OVERLAP        # 256
settings.MAX_REF_PER_TILE    # 2 (Top-K limite)

# Geração
settings.IP_ADAPTER_END_STEP      # 0.6 (temporal decay)
settings.BACKGROUND_IP_SCALE      # 0.0 (isolação)
settings.CONTEXT_INFLATION_FACTOR # 1.5 (bbox inflation)

# Modelos
settings.SDXL_MODEL_ID       # "ByteDance/SDXL-Lightning"
settings.SDXL_STEPS          # 4
settings.CONTROLNET_MODEL    # "diffusers/controlnet-canny-sdxl-1.0"
```

---

## Exemplos Completos

### Exemplo 1: Colorização Básica

```python
from core.pipeline import MangaColorizationPipeline
from pathlib import Path

# Inicializa
pipeline = MangaColorizationPipeline(
    device="cuda",
    enable_cpu_offload=True  # Otimizado para RTX 3060
)

# Passo 1: Análise
chapter_path = Path("./manga/chapter_01")
chapter_id, summary = pipeline.process_chapter(chapter_path)

print(f"Capítulo: {chapter_id}")
print(f"Páginas: {summary.num_pages}")
print(f"Personagens: {summary.num_characters}")

# Passo 2: Geração
output_dir = Path("./output/chapter_01")
output_dir.mkdir(exist_ok=True)

for result in pipeline.generate_chapter(chapter_id):
    output_path = output_dir / f"page_{result.page_number:03d}.png"
    result.image.save(output_path)
    print(f"✓ Página {result.page_number}/{result.total_pages} "
          f"({result.inference_time:.1f}s)")
```

### Exemplo 2: Geração Não-Linear

```python
pipeline = MangaColorizationPipeline()

# Análise completa primeiro
chapter_id, summary = pipeline.process_chapter(Path("./chapter_02"))

# Gera apenas páginas de interesse
interesting_pages = [5, 12, 18]  # Qualquer ordem

for result in pipeline.generate_chapter(chapter_id, pages=interesting_pages):
    result.image.save(f"./output/page_{result.page_number:03d}.png")
```

### Exemplo 3: Múltiplos Estilos

```python
pipeline = MangaColorizationPipeline()
chapter_id, _ = pipeline.process_chapter(Path("./chapter_03"))

# Gera versões com diferentes estilos
styles = ["default", "vibrant", "pastel"]

for style in styles:
    style_dir = Path(f"./output/{style}")
    style_dir.mkdir(exist_ok=True)
    
    for result in pipeline.generate_chapter(
        chapter_id, 
        page_numbers=[1, 2, 3],
        style_preset=style
    ):
        output_path = style_dir / f"page_{result.page_number:03d}.png"
        result.image.save(output_path)
```

### Exemplo 4: API REST com Python

```python
import requests

base_url = "http://localhost:8000"

# Upload
data = {"chapter_id": "chapter_test"}
files = {"file": open("chapter_test.zip", "rb")}
resp = requests.post(f"{base_url}/chapters/upload", data=data, files=files)
chapter_id = resp.json()["chapter_id"]

# Analisar
resp = requests.post(f"{base_url}/chapters/{chapter_id}/analyze")
job_id = resp.json()["job_id"]

# Aguardar conclusão
import time
while True:
    status = requests.get(f"{base_url}/jobs/{job_id}").json()
    if status["status"] == "completed":
        break
    time.sleep(1)

# Gerar
payload = {"page_numbers": [1, 2, 3], "style_preset": "vibrant"}
resp = requests.post(f"{base_url}/chapters/{chapter_id}/generate", json=payload)
gen_job_id = resp.json()["job_id"]

# Download resultados
import urllib.request
for page in [1, 2, 3]:
    url = f"{base_url}/outputs/{chapter_id}/page_{page:03d}.png"
    urllib.request.urlretrieve(url, f"page_{page:03d}.png")
```

---

## Tratamento de Erros

```python
from core.pipeline import MangaColorizationPipeline
from core.exceptions import (
    AnalysisError,
    GenerationError,
    ModelLoadError,
    OutOfMemoryError,
    InvalidChapterError
)

pipeline = MangaColorizationPipeline()

try:
    chapter_id, summary = pipeline.process_chapter(chapter_path)
    result = pipeline.generate_page(chapter_id, 1)
    
except AnalysisError as e:
    print(f"Erro na análise: {e}")
    print(f"Arquivo problemático: {e.file_path}")
    
except GenerationError as e:
    print(f"Erro na geração: {e}")
    print(f"Página: {e.page_number}")
    
except ModelLoadError as e:
    print(f"Falha ao carregar modelo: {e.model_name}")
    print("Verifique a conexão com internet ou cache local")
    
except OutOfMemoryError as e:
    print(f"VRAM insuficiente: {e.requested_gb:.1f}GB requerido")
    print("Sugestões:")
    print("1. Reduzir TILE_SIZE para 512")
    print("2. Habilitar ENABLE_CPU_OFFLOAD")
    print("3. Reduzir MAX_REF_PER_TILE para 1")
    
except InvalidChapterError as e:
    print(f"Capítulo inválido: {e}")
    print("Certifique-se de que o diretório contém imagens .png/.jpg/.webp")
```

---

## Constantes e Enums

```python
from core.types import StylePreset, QualityMode

# Style Presets
StylePreset.DEFAULT     # "default"
StylePreset.VIBRANT     # "vibrant"
StylePreset.PASTEL      # "pastel"
StylePreset.CYBERPUNK   # "cyberpunk"
StylePreset.SEPIA       # "sepia"
StylePreset.MUTED       # "muted"

# Quality Modes
QualityMode.FAST        # ~20s/página
QualityMode.BALANCED    # ~30s/página
QualityMode.HIGH        # ~60s/página
```

---

## Limitações e Notas

1.  **Resolução Máxima**: 4096px (usa tiling automaticamente)
2.  **Formatos Suportados**: PNG, JPG, JPEG, WEBP
3.  **VRAM Mínima**: 6GB (com CPU Offload), 8GB recomendado
4.  **Personagens por Tile**: Máximo 2 (MAX_REF_PER_TILE)
5.  **Consistência**: Garantida via cache imutável de embeddings
6.  **Engine**: SD 1.5 + ControlNet Lineart Anime + IP-Adapter Plus Face

---

## Referências

- [Arquitetura](ARCHITECTURE.md) - Detalhes da arquitetura Two-Pass
- [SETUP.md](SETUP.md) - Guia de instalação
- [CLI](CLI.md) - Documentação completa do CLI

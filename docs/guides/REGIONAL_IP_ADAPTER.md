# 🎭 Regional IP-Adapter System

## Visão Geral

O sistema de Regional IP-Adapter permite injetar identidades de personagens em regiões específicas da imagem durante a geração. Isso é especialmente útil em páginas de mangá com múltiplos personagens, onde cada um precisa manter sua aparência consistente.

## Arquitetura

### Fluxo de Dados

```
Pass 1 (Análise):
  ┌─────────────────────────────────────────────────────┐
  │  YOLO Detection → Crop Personagem → Extração       │
  │  • Crop da região detectada (+50% contexto)         │
  │  • Extração CLIP embedding (ViT-H)                 │
  │  • Extração Face embedding (ArcFace)               │
  │  → Salvo em: chapter_db (embeddings_dir)           │
  └─────────────────────────────────────────────────────┘
                           ↓
Pass 2 (Geração):
  ┌─────────────────────────────────────────────────────┐
  │  Load Embeddings → Regional IP-Adapter → Geração   │
  │  • Carrega embeddings do banco de dados            │
  │  • Cria máscaras binárias para cada personagem     │
  │  • Injeta embeddings apenas nas regiões ativas     │
  └─────────────────────────────────────────────────────┘
```

## Componentes Principais

### 1. IdentityEncoder

```python
from core.models.identity_encoder import IdentityEncoder

encoder = IdentityEncoder(
    device="cuda",
    clip_model_path="./models/image_encoder",
    face_model_path="./models/arcface_model"
)

# Extração de identidade
clip_emb, face_emb = encoder.extract_identity(character_crop)
# clip_emb: [1, 1024] - Embedding CLIP ViT-H
# face_emb: [1, 512] - Embedding ArcFace (se face detectada)
```

### 2. Pipeline Integration

```python
# Em core/generation/pipeline.py

def _generate_single_tile(self, image, condition, options, page_data):
    # ... preparação ...
    
    # Configurar Regional IP-Adapter se crops disponíveis
    crops = options.get('character_crops', [])
    if crops:
        self._setup_regional_ip_adapter(pipe, crops)
    
    # Geração
    result = pipe(...)

def _setup_regional_ip_adapter(self, pipe, crops):
    """Configura IP-Adapter para injeção regional"""
    adapters = []
    masks = []
    scales = []
    
    for char_id, crop in crops.items():
        adapter = IPAdapterPlus(
            pipe,
            image_encoder_path="...",
            ip_ckpt="...",
            num_tokens=4
        )
        adapters.append(adapter)
        masks.append(crop['mask'])
        scales.append(crop['scale'])
```

### 3. Estratégia de Injeção (Early-Heavy)

A estratégia Early-Heavy maximiza o impacto da identidade nos primeiros passos de denoising:

```python
def regional_ip_adapter_callback(pipe, step_index, timestep, callback_kwargs):
    """
    Callback para injeção regional de IP-Adapter.
    
    Estratégia: Early-Heavy Injection
    - Passo 0: Personagem A (scale=1.0), Personagem B (scale=0.0)
    - Passo 1: Personagem A (scale=0.0), Personagem B (scale=0.6)
    - Passos 2-3: Ambos off (deixar SDXL consolidar)
    """
    latent = callback_kwargs["latents"]
    
    if step_index == 0:
        # Apenas Personagem A ativo
        inject_with_mask(latent, adapter_A, mask_A, scale=1.0)
    elif step_index == 1:
        # Apenas Personagem B ativo
        inject_with_mask(latent, adapter_B, mask_B, scale=0.6)
    # Passos 2-3: sem injeção
    
    return callback_kwargs
```

## Uso da API

### Com CLI

```bash
# Geração com Regional IP (automático quando há múltiplos personagens)
python cli.py generate --chapter-id <id> --pages 1,2,3

# Desativar Regional IP
python cli.py generate --chapter-id <id> --pages 1 --no-regional-ip
```

### Com API REST

```bash
curl -X POST http://localhost:8000/chapters/chapter_01/generate \
  -H "Content-Type: application/json" \
  -d '{
    "page_numbers": [5],
    "regional_ip": true,
    "ip_strength": 0.8
  }'
```

### Python Direct

```python
from core.generation.pipeline import TileAwareGenerator

generator = TileAwareGenerator(device="cuda", dtype=torch.float16)

# Opções com Regional IP
options = {
    'character_crops': {
        'char_001': {
            'image': crop_image,
            'mask': binary_mask,
            'scale': 0.8
        }
    },
    'character_embeddings': {
        'char_001': {
            'clip': clip_embedding,
            'face': face_embedding
        }
    }
}

result = generator.generate_image(
    image=input_image,
    prompt="colorful manga illustration...",
    options=options
)
```

## Máscaras

### Formato

As máscaras são tensores binários no formato `[1, 1, H, W]`:

```python
# Exemplo de criação de máscara
mask = torch.zeros(1, 1, 1024, 1024)
mask[:, :, y1:y2, x1:x2] = 1.0  # Região do personagem
```

### Overlap Handling

Quando personagens se sobrepõem:

```python
def resolve_mask_overlap(masks: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Resolve sobreposições priorizando personagens maiores.
    """
    # Calcular área de cada máscara
    areas = [m.sum() for m in masks]
    
    # Ordenar por área (maior primeiro)
    sorted_indices = sorted(range(len(areas)), key=lambda i: -areas[i])
    
    # Combinar máscaras
    combined = torch.zeros_like(masks[0])
    for idx in sorted_indices:
        masks[idx] = masks[idx] * (1 - combined)
        combined = combined | (masks[idx] > 0)
    
    return masks
```

## Configurações

### Ajuste de Força

A força do IP-Adapter pode ser ajustada por personagem:

```python
# Configurações recomendadas
CONFIGS = {
    'strong': {
        'scale': 1.0,
        'num_tokens': 16,
        'steps_active': [0, 1]  # Ativo nos primeiros 2 passos
    },
    'balanced': {
        'scale': 0.7,
        'num_tokens': 4,
        'steps_active': [0]
    },
    'subtle': {
        'scale': 0.4,
        'num_tokens': 4,
        'steps_active': [0]
    }
}
```

### Arquivos de Configuração

```yaml
# config/regional_ip.yaml
regional_ip_adapter:
  enabled: true
  strategy: "early_heavy"
  
  early_heavy:
    step_0_scale: 1.0
    step_1_scale: 0.6
    steps_off: [2, 3, 4]
  
  mask_blur: 8  # Blur nas bordas da máscara (pixels)
  overlap_mode: "larger_wins"  # ou "blend", "priority"
```

## Troubleshooting

### Personagem não reconhecido

```python
# Verificar embeddings salvos
db = ChapterDatabase(chapter_id)
palettes = db.load_all_palettes()

for char_id, palette in palettes.items():
    print(f"{char_id}: {palette.colors}")
    
# Verificar se há embeddings
import json
emb_file = db.embeddings_dir / f"{char_id}_embedding.json"
if emb_file.exists():
    with open(emb_file) as f:
        data = json.load(f)
        print(f"CLIP: {len(data['clip_embedding'])}")
        print(f"Face: {len(data['face_embedding'])}")
```

### Artefatos nas bordas

Se houver artefatos nas bordas das máscaras:

```python
# Aumentar blur nas máscaras
options['mask_blur'] = 16  # Padrão é 8

# Ou usar feathering
from PIL import ImageFilter
mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=8))
mask = np.array(mask_pil) / 255.0
```

### Identidade fraca

Se o personagem não está parecido o suficiente:

```python
# Aumentar scale
options['character_crops'][char_id]['scale'] = 1.2

# Aumentar num_tokens no IP-Adapter
adapter = IPAdapterPlus(pipe, ..., num_tokens=16)  # Padrão é 4

# Usar mais passos de injeção
# Modificar callback para injetar em steps 0, 1, 2
```

## Performance

### Otimizações

```python
# Desativar quando não necessário
if num_characters == 0:
    # Sem personagens detectados, skip Regional IP
    options['character_crops'] = []

# Cache de adapters
adapter_cache = {}
def get_adapter(char_id):
    if char_id not in adapter_cache:
        adapter_cache[char_id] = create_adapter(char_id)
    return adapter_cache[char_id]

# Limpar cache após geração
adapter_cache.clear()
torch.cuda.empty_cache()
```

### Métricas

| Cenário | VRAM Extra | Tempo Extra |
|---------|-----------|-------------|
| Sem Regional IP | 0 MB | 0s |
| 1 Personagem | ~400 MB | +2s |
| 2 Personagens | ~600 MB | +3s |
| 3+ Personagens | ~800 MB | +4s |

## Limitações

1. **Máximo de Personagens**: Recomendado máximo de 4 personagens simultâneos
2. **Resolução**: Máscaras são redimensionadas para a resolução do latent (1/8 da imagem)
3. **Overlap Complexo**: Cenários com múltiplas sobreposições podem precisar de ajuste manual
4. **Memória**: Cada adapter adicional consome ~200-300MB de VRAM

## Referências

- IP-Adapter Paper: https://arxiv.org/abs/2308.06721
- IP-Adapter Plus: https://github.com/tencent-ailab/IP-Adapter
- Regional Attention: https://arxiv.org/abs/2312.09613

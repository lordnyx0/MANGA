# Plano de Implementação ADR 004: SAM 2.1 Segmentation & Z-Buffer

## Visão Geral da Arquitetura

```
┌─────────────────────────────────────────────────────────────────────┐
│ PASS 1: ANÁLISE (CPU Bound)                                         │
│ ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│ │   YOLO       │───▶│  SAM 2.1     │───▶│  Z-Buffer    │           │
│ │  Detector    │    │  Segmenter   │    │  Calculator  │           │
│ └──────────────┘    └──────────────┘    └──────────────┘           │
│       │                    │                    │                   │
│       ▼                    ▼                    ▼                   │
│   Detecções          Máscaras              Ordenação               │
│   (Bbox)             Precisas             de Profundidade          │
│       │                    │                    │                   │
│       └────────────────────┴────────────────────┘                   │
│                          │                                          │
│                          ▼                                          │
│              ┌──────────────────────┐                              │
│              │   Chapter Database   │                              │
│              │   (Máscaras RLE +    │                              │
│              │    Metadados)        │                              │
│              └──────────────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PASS 2: GERAÇÃO (GPU Bound)                                         │
│ ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│ │  ChapterDB   │───▶│  Regional    │───▶│   SDXL       │           │
│ │   Loader     │    │  IP-Adapter  │    │  Pipeline    │           │
│ └──────────────┘    └──────────────┘    └──────────────┘           │
│       │                    │                                        │
│       ▼                    ▼                                        │
│   Máscaras SAM    ip_adapter_masks                                  │
│   (Binary +       (com blur suave)                                  │
│    Blur 0.5)                                                        │
└─────────────────────────────────────────────────────────────────────┘
```

## Componentes a Implementar

### 1. SAM2Segmenter (`core/analysis/segmentation.py`)
**Responsabilidade:** Gerar máscaras pixel-wise precisas a partir de bounding boxes YOLO

**Interface:**
```python
class SAM2Segmenter:
    def __init__(self, model_size: str = "tiny", device: str = "cpu")
    def segment(self, image: np.ndarray, detections: List[DetectionResult]) -> Dict[str, np.ndarray]
    def batch_segment(self, images: List[np.ndarray], detections_list: List[List[DetectionResult]]) -> List[Dict[str, np.ndarray]]
```

**Detalhes de Implementação:**
- Usar SAM 2.1 Tiny (35MB) via `facebookresearch/sam2`
- Entrada: Detecções YOLO (bbox + class_id)
- Saída: Máscaras binárias por personagem (char_id -> mask)
- Operação em CPU durante Pass 1 (preserva VRAM para Pass 2)
- Cache de encoder para múltiplas páginas sequenciais (consistência temporal)

**Fórmula de Segmentação:**
```
M_SAM = SAM(image, point_grid=bbox_center, box=bbox)
M_refined = morphological_close(M_SAM, kernel=3x3)
```

### 2. ZBufferCalculator (`core/analysis/z_buffer.py`)
**Responsabilidade:** Ordenar personagens por profundidade relativa

**Interface:**
```python
class ZBufferCalculator:
    def __init__(self, weights: ZBufferWeights)
    def calculate_depth(self, detection: DetectionResult, image_size: Tuple[int, int]) -> float
    def sort_by_depth(self, detections: List[DetectionResult], image_size: Tuple[int, int]) -> List[DetectionResult]
```

**Fórmula de Profundidade (ADR 004):**
```
D(p) = w₁·H(y_center) + w₂·(1 - A_max/area(p)) + w₃·τ(type_p) + w₄·δ(p)

Onde:
- H(y_center): Posição vertical normalizada (0-1, topo-fundo)
- area(p): Área da detecção em pixels
- A_max: Maior área entre todas detecções
- τ(type_p): Prioridade semântica (face=0.0, body=0.5, frame=1.0)
- δ(p): Profundidade estimada MiDaS Small (opcional, 0 se não disponível)
```

**Pesos Padrão (configuráveis em settings.py):**
```python
ZBUFFER_WEIGHT_Y = 0.4        # Posição vertical (mais importante em mangá)
ZBUFFER_WEIGHT_AREA = 0.3     # Área relativa
ZBUFFER_WEIGHT_TYPE = 0.2     # Prioridade semântica
ZBUFFER_WEIGHT_DEPTH = 0.1    # Profundidade MiDaS (opcional)
```

### 3. MaskProcessor (`core/analysis/mask_processor.py`)
**Responsabilidade:** Pós-processamento e cache de máscaras

**Interface:**
```python
class MaskProcessor:
    def apply_morphological_close(self, mask: np.ndarray, kernel_size: int = 3) -> np.ndarray
    def apply_erosion(self, mask: np.ndarray, pixels: int = 2) -> np.ndarray
    def compute_occlusion_masks(self, masks: Dict[str, np.ndarray], depth_order: List[str]) -> Dict[str, np.ndarray]
    def encode_rle(self, mask: np.ndarray) -> str
    def decode_rle(self, rle: str, shape: Tuple[int, int]) -> np.ndarray
```

**Fórmula de Isolamento (ADR 004):**
```
M_i_final = M_i_SAM ∩ (¬⋃_{j∈Front(i)} M_j_SAM) ∩ (¬⋃_{k∈SamePlane(i)} ε(M_k_SAM))

Onde:
- ε: Erosão morfológica de 2px
- Front(i): Personagens com menor profundidade (mais à frente)
- SamePlane(i): Personagens no mesmo plano de profundidade
```

### 4. Modificações no Pass1Analyzer
**Integração:**
1. Após detecção YOLO, chamar SAM2Segmenter
2. Calcular Z-Buffer e ordenar personagens
3. Aplicar fórmula de isolamento
4. Salvar máscaras otimizadas no ChapterDatabase

**Novos Atributos em Character Data:**
```python
character_data = {
    'bbox': (x1, y1, x2, y2),
    'sam_mask_path': 'path/to/mask.npy',  # NOVO
    'depth_score': 0.75,                   # NOVO
    'depth_rank': 2,                       # NOVO (1 = mais à frente)
    # ... outros campos existentes
}
```

### 5. Modificações no Pass2Generator
**Integração:**
1. Priorizar uso de máscaras SAM do ChapterDatabase
2. Fallback para máscaras BBox se SAM não disponível
3. Aplicar blur gaussiano sigma=0.5 nas bordas antes de passar para IP-Adapter

**Pipeline de Máscaras:**
```
1. Carrega máscara SAM binária (do DB ou gera on-the-fly)
2. Aplica subtração de oclusores (baseado em depth_rank)
3. Aplica erosão de 2px em regiões de contato
4. Aplica GaussianBlur sigma=0.5 (anti-hard-edge)
5. Passa para ip_adapter_masks
```

### 6. Modificações no ChapterDatabase
**Novas Colunas em `characters.parquet`:**
- `sam_mask_path`: Path para máscara .npy
- `depth_score`: Score de profundidade calculado
- `depth_rank`: Ordem de profundidade (1 = frente)

**Novas Colunas em `tiles.parquet`:**
- `sam_mask_paths`: JSON {char_id: mask_path}
- `depth_order`: JSON [char_id, ...] ordenado por profundidade

## Estrutura de Arquivos

```
core/
├── analysis/
│   ├── __init__.py
│   ├── segmentation.py       # SAM2Segmenter
│   ├── z_buffer.py           # ZBufferCalculator
│   └── mask_processor.py     # MaskProcessor (morphological ops + RLE)
├── detection/
│   └── ...
├── generation/
│   └── ...
└── database/
    └── ...
```

## Configurações (settings.py)

```python
# SAM 2.1 Configurações
SAM2_ENABLED = True
SAM2_MODEL_SIZE = "tiny"  # tiny, small, base, large
SAM2_DEVICE = "cpu"  # cpu para Pass 1, preserva VRAM
SAM2_USE_ONNX = True  # Fallback para ONNX se CUDA indisponível

# Z-Buffer Configurações
ZBUFFER_ENABLED = True
ZBUFFER_WEIGHT_Y = 0.4
ZBUFFER_WEIGHT_AREA = 0.3
ZBUFFER_WEIGHT_TYPE = 0.2
ZBUFFER_WEIGHT_DEPTH = 0.1
ZBUFFER_USE_MIDAS = False  # MiDaS Small opcional

# Mask Processing
MASK_MORPH_CLOSE_KERNEL = 3
MASK_EROSION_PIXELS = 2
MASK_EDGE_BLUR_SIGMA = 0.5  # Para IP-Adapter

# Fallback
SAM2_FALLBACK_TO_BBOX = True  # Se SAM falhar, usar BBox
```

## Fluxo de Dados Completo

### Pass 1 (Análise)
```
1. YOLO Detection → Lista de DetectionResult (bbox, class, conf)
2. SAM2Segmenter → Máscaras brutas por personagem
3. ZBufferCalculator → Ordenação por profundidade
4. MaskProcessor → Isolamento + Morfologia + RLE
5. ChapterDatabase → Persistência (máscaras .npy + metadados)
```

### Pass 2 (Geração)
```
1. ChapterDB Loader → Máscaras + depth_order
2. MaskProcessor.decode → Máscaras numpy
3. MaskProcessor.blur_edges → Suavização borda
4. RegionalIPAdapter → ip_adapter_masks
5. SDXL Pipeline → Geração final
```

## Métricas de Qualidade Esperadas

| Métrica | v2.6.4 (BBox) | v2.7 (SAM 2.1) | Melhoria |
|---------|---------------|----------------|----------|
| Color Bleeding em Overlap | ~72% | ~94% | +22% |
| Flickering Temporal | Alto | Baixo | Significativa |
| IoU Segmentação | N/A | 68-89% | Novo |

## Fallback Strategy

```python
if OOM or SAM2_FAILED:
    if SAM2_FALLBACK_TO_BBOX:
        logger.warning("SAM 2.1 falhou, usando BBox como fallback")
        masks = generate_bbox_masks(detections)  # Método atual v2.6.4
    else:
        raise RuntimeError("Segmentação falhou e fallback desativado")
```

## Testes Requeridos

### Unitários
- `test_sam2_segmenter.py`: Testar segmentação com mock SAM
- `test_z_buffer.py`: Verificar ordenação por profundidade
- `test_mask_processor.py`: Testar RLE encode/decode, morfologia

### Integração
- `test_pass1_with_sam.py`: Verificar integração completa Pass 1
- `test_pass2_with_sam_masks.py`: Verificar uso de máscaras SAM na geração

### Performance
- Benchmark tempo de segmentação vs BBox
- Benchmark VRAM com/sem SAM

## Checklist de Implementação

- [ ] SAM2Segmenter implementado e testado
- [ ] ZBufferCalculator implementado e testado
- [ ] MaskProcessor implementado e testado
- [ ] Pass1Analyzer integrado com novos componentes
- [ ] Pass2Generator usando máscaras SAM
- [ ] ChapterDatabase estendido para novos campos
- [ ] Configurações adicionadas em settings.py
- [ ] Fallback implementado e testado
- [ ] Documentação atualizada

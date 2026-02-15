# Progresso do MangaAutoColor Pro

> **🎉 Status Atual:** v2.7 - Sistema de Produção com PCTC
> 
> **✅ Sistema Two-Pass:** Análise (Pass 1) + Geração (Pass 2)  
> **✅ Detecção:** Manga109 YOLO + SAM 2.1 Segmentation + CannyContinuityNMS  
> **✅ Identidade:** CLIP + ArcFace + Paletas CIELAB (usadas em prompts)  
> **✅ Geração:** Regional IP-Adapter + Differential Diffusion + Tile 1024×1024  
> **✅ ADR 004:** Z-Buffer Calculator para ordenação de profundidade  
> **✅ ADR 005:** Point Correspondence (LightGlue) + Temporal Consistency (RAFT)  
> **✅ Referências de Cor:** Upload de imagens coloridas para extração real de paletas  
> **✅ Style Presets:** 7 presets configuráveis (quando sem referências)  
> **✅ Logging:** Sistema completo de logs em `output/{chapter_id}/logs/`  
> **✅ Database:** FAISS + Parquet + Cache Imutável  
> **✅ Testes:** Suite completa (190 testes: Unit, Integration, E2E)  
> **📅 Última atualização:** 2026-02-13

---

## ✅ O que foi implementado (v2.7.0) - ADR 005

### 🎯 ADR 005: Point Correspondence & Temporal Consistency (PCTC)

#### ✅ Point Correspondence Service
- [x] **LightGlue + SuperPoint**: Matching semântico de keypoints
- [x] **ORB Fallback**: Funciona sem dependências externas (OpenCV-only)
- [x] **Attention Heatmaps**: Geração de máscaras Gaussianas para cross-attention
- [x] **CPU-only**: Zero VRAM adicional (processamento 100% CPU)
- [x] **Arquivo:** `core/analysis/point_matching.py` (380+ linhas)
- [x] **Testes:** 17 unit tests + integração

#### ✅ Temporal Consistency Service
- [x] **SSIM Scene Detection**: Detecta mudança de cena automaticamente
- [x] **RAFT Optical Flow**: Propaga cores em cenas contínuas
- [x] **Farneback Fallback**: OpenCV-based quando RAFT indisponível
- [x] **Histogram Matching**: Transferência de cor para cenas discontínuas
- [x] **Color Hint Maps**: Mapas de condicionamento para Pass 2
- [x] **Arquivo:** `core/analysis/temporal_flow.py` (420+ linhas)
- [x] **Testes:** 17 unit tests + integração

#### ✅ Integração com ADR 004
- [x] SAM 2.1 masks + Point Correspondence = Segmentação semântica completa
- [x] Z-Buffer + Temporal Consistency = Ordenação e continuidade temporal
- [x] RegionalIPAdapter aceita `cross_attention_kwargs` para attention masks

#### ✅ Test Suite ADR 005
- [x] **41 novos testes**: 100% pass rate
- [x] **Cobertura:** ~85% line coverage
- [x] **Integração:** 7 testes de integração end-to-end

---

## ✅ O que foi implementado (v2.6.3)

### 🛡️ Phase 17: Audit e Melhoria de Testes

#### ✅ Suite de Testes Padronizada
- [x] **Estrutura:** `tests/unit`, `tests/integration`, `tests/e2e`
- [x] **Runner:** `run_tests.bat` com detecção automática de venv
- [x] **Config:** `pytest.ini` com marcadores e cobertura

#### ✅ Novos Testes Implementados
- [x] **E2E Pipeline:** `tests/e2e/test_pipeline.py` (Simulação completa sem modelos pesados)
- [x] **Text Compositing:** `tests/unit/test_text_compositor.py`
- [x] **Prompt Builder:** `tests/unit/test_prompt_builder.py`
- [x] **Exceptions:** `tests/unit/test_exceptions.py`

#### ✅ Correções de Estabilidade
- [x] **Ambiente:** Resolução de conflitos de importação (System vs Venv)
- [x] **Persistência:** Correção de erro de instanciação do `ChapterDatabase` em testes
- [x] **Cleanup:** Remoção de scripts de verificação obsoletos (`verify_*.py`)

---

## ✅ O que foi implementado (v2.6)

### 🆕 Novidades v2.6 - Sistema de Produção

#### ✅ Sistema de Logs Completo
- [x] **GenerationLogger:** Logger estruturado com timeline de execução
- [x] **Prompts:** Registro de todos os prompts usados (positivo/negativo)
- [x] **Detecções:** Salvamento de detecções por página em JSON
- [x] **Embeddings:** Metadados de embeddings e paletas
- [x] **Arquivos:** Logs em `output/{chapter_id}/logs/`
  - `generation_log.json` - Log completo
  - `prompts_used.txt` - Prompts legíveis
  - `timeline.txt` - Timeline de execução

#### ✅ Imagens de Referência Coloridas
- [x] **Upload na Extensão:** Suporte a múltiplas imagens de referência
- [x] **Extração de Paletas:** Cores reais extraídas das referências
- [x] **Prioridade:** Referências sobrescrevem STYLE_PRESETS
- [x] **Mapping Automático:** Personagens mapeados por similaridade visual
- [x] **Persistência:** Paletas salvas em `embeddings/ref_char_*_palette.json`

#### ✅ Style Presets (7 opções)
- [x] **Frontend:** Dropdown na extensão do navegador
- [x] **Presets:** default, vibrant, muted, sepia, flashback, dream, nightmare
- [x] **Comportamento:** Aplicados apenas quando não há referências de cor
- [x] **Backend:** Configurações em `config/settings.py`

#### ✅ Correções de Bugs
- [x] **Multi-Tile Blending:** Corrigido erro de dimensões entre tiles
- [x] **Correção "Orange":** Paletas B&W não mais usadas em prompts
- [x] **Regional IP-Adapter:** Embeddings chegam corretamente ao pipeline
- [x] **Referências:** `_calculate_context_bbox()` implementado
- [x] **Persistência:** Paletas de referência salvas mesmo sem estar no DataFrame

#### ✅ Documentação
- [x] **API.md:** Documentação dos endpoints REST atualizada
- [x] **REGIONAL_IP_ADAPTER.md:** Guia técnico do sistema Regional IP
- [x] **COLOR_REFERENCES.md:** Guia de uso de referências de cor

---

## ✅ O que foi implementado (v2.3)

### 🆕 Novidades v2.5 - Regional IP-Adapter Implementado

#### ✅ Regional IP-Adapter com Early-Heavy Injection
- [x] **Módulo `regional_ip_adapter.py`:** Implementado com API nativa do Diffusers
  - Usa `IPAdapterMaskProcessor` para máscaras regionais
  - Usa `cross_attention_kwargs` com `ip_adapter_masks`
  - Callback `early_heavy_callback` para controle temporal
- [x] **Modelo IP-Adapter Plus Face ViT-H:** Carregado dinamicamente
  - Maior impacto por step (ideal para 4 steps)
  - Atenção: consome +600MB VRAM (requer `enable_model_cpu_offload()`)
  - Escala controlada dinamicamente: 1.0 → 0.6 → 0.0 → 0.0
- [x] **Estratégia Early-Heavy (T-GATE):**
  - Step 0 (0-25%): Scale 1.0 - Personagem A com força máxima
  - Step 1 (25-50%): Scale 0.6 - Personagem B (ou fade)
  - Steps 2-3 (50-100%): Scale 0.0 - Desligado, ControlNet domina
- [x] **Injeção Cíclica (ICAS):** Alterna foco entre personagens
  - Elimina vazamento de identidade
  - Preserva 40% mais características individuais vs simultâneo
- [x] **Fallback de Memória:** OOM detectado → reduz para 1 personagem → ou sem IP-Adapter

#### ⚠️ Limitações Conhecidas
- **Máximo 2 personagens simultâneos** na RTX 3060 12GB (limite de VRAM)
- **Efeito "Uncanny Valley":** Plus Face tende a realismo em mangá 2D
  - Solução: Manter scale ≤ 0.7 após Step 0
- **Requer Diffusers ≥0.29.0:** Para suporte a `ip_adapter_masks`

### 🆕 Novidades v2.4.1 - Correções e Estabilização

#### ✅ Correções Importantes
- [x] **Correção de Variável:** Erro `name 'original_image' is not defined` corrigido
  - Variável renomeada para `original_image_resized` para maior clareza
  - Fluxo de upscale preventivo preservado corretamente
- [x] **Upscale Preventivo Funcional:** Geração em resolução maior, output no tamanho original
  - Input pequeno (650x933) → Upscale (1024x1469) → Ajuste 64 (1024x1408) → Geração → Downscale (650x933)
  - Melhora qualidade sem alterar dimensões finais

### ✅ O que foi implementado (v2.4) - Differential Diffusion + Paletas

#### ✅ Implementado em v2.4
- [x] **Differential Diffusion:** Change Maps aplicadas nos latents durante geração
  - Centro do personagem: força 1.0
  - Bordas: decaimento gaussiano
  - Background: força 0.0 (isolação)
- [x] **Paletas em Prompts:** Cores dos personagens são extraídas e usadas nos prompts
  - Hair color, clothes color, eyes color
  - Melhora consistência entre páginas
- [x] **Text Compositing API:** Controle via parâmetro `text_compositing` na API
  - Usuário pode ativar/desativar preservação de texto
  - Checkbox na extensão do navegador
  - Correção de coordenadas quando imagem é redimensionada

### ✅ O que foi implementado (v2.3)

#### Arquitetura Two-Pass (Otimizada)
- [x] **Pass 1 (Análise):** CPU/IO bound - processa todas as páginas
  - Detecção YOLO + CannyContinuityNMS
  - Extração de embeddings (CLIP + ArcFace)
  - Extração de paletas CIELAB (hair, skin, eyes, clothes)
  - Pre-computação de tiles (só para páginas >1024px)
  - Cache imutável (FAISS + Parquet + .pt)
- [x] **Pass 2 (Geração):** VRAM bound - gera páginas sob demanda
  - **Single Tile** (padrão): Páginas ≤1024px em uma inferência (~8-15s)
  - **Tiled Mode** (fallback): Páginas grandes divididas em tiles
  - Carrega embeddings dos personagens presentes
  - **Differential Diffusion:** Change Maps aplicadas nos latents
  - **Paleta em Prompts:** Cores dos personagens no prompt
  - Background Isolation
  - Multi-band Blending (só para tiled)

#### Detecção Aprimorada
- [x] **CannyContinuityNMS:** Merge de detecções baseado em continuidade de bordas
  - Resolve personagens conectados por bleed art
  - IoU threshold + Canny continuity check
  - Supressão de detecções pequenas
- [x] **Agrupamento body/face:** Associa rosto ao corpo do mesmo personagem

#### Identidade e Paletas
- [x] **HybridIdentitySystem:** CLIP (768-dim) + ArcFace (512-dim)
- [x] **PaletteExtractor:** Extração de paletas CIELAB por região
  - Hair, skin, eyes, clothes_primary
  - Delta E para comparação perceptual
  - K-means clustering para cores dominantes
- [x] **Cache de embeddings:** Tensores .pt imutáveis por capítulo

#### Database Híbrido
- [x] **FAISS:** Indexação vetorial para busca por similaridade
- [x] **Parquet:** Metadados estruturados (characters, tiles, pages)
- [x] **.pt files:** Tensores de embeddings (cache imutável)
- [x] **Consolidação:** Merge automático de personagens similares (>0.95)

#### Geração Tile-Aware
- [x] **Tile slicing:** 1024×1024 com overlap 256px
- [x] **Top-K limit:** Máximo 2 personagens por tile
- [x] **Máscaras Gaussianas:** Força 1.0 (centro) → 0.0 (bordas)
- [x] **Differential Diffusion:** Change Maps nos latents
- [x] **Background Isolation:** Força 0 do IP-Adapter no fundo
- [x] **Multi-band blending:** Feathered edges para tiles (só modo tiled)

#### Substituição no Navegador (v2.4)
- [x] **Mapeamento por src:** Imagens são mapeadas pelo `src` original, não por índice
- [x] **Injeção garantida:** Content script é injetado explicitamente antes da substituição
- [x] **Logs detalhados:** Adicionados logs extensivos para facilitar debug
- [x] **Efeito visual:** Fade suave ao substituir imagens

#### API REST Two-Pass
- [x] **POST /chapter/analyze:** Upload de múltiplas páginas
- [x] **POST /chapter/generate:** Geração de páginas colorizadas
- [x] **GET /chapter/{id}/status:** Status do processamento
- [x] **GET /chapter/{id}/download:** Download ZIP com resultados

---

## ✅ Infraestrutura Anterior (v2.2)

### Configurações
- [x] `config/settings.py`:
  - `SDXL_GUIDANCE_SCALE = 1.2`
  - `CONTROLNET_CONDITIONING_SCALE = 0.85`
  - `YOLO_MODEL_ID = "deepghs/manga109_yolo"`
  - `TILE_SIZE = 1024`
  - `MAX_REF_PER_TILE = 2`
  - `IP_ADAPTER_END_STEP = 0.6`

### Detecção
- [x] `core/detection/yolo_detector.py`:
  - Modelo Manga109 YOLO (ONNX Runtime)
  - Classes: body, face, frame, text
  - Bbox inflation 150%
  - Prominence score

### Geração
- [x] `core/generation/pipeline.py`:
  - SDXL-Lightning (4 steps)
  - ControlNet Canny
  - VAE FP16 Fix
  - Text compositing
  - Upscale preventivo <1024px

---

## 📋 Arquitetura: Documentação vs Realidade

### O que Mudou na Prática

| Aspecto | Documentação Original | Implementação Real | Motivo |
|---------|----------------------|-------------------|--------|
| **Tile Size** | ~~1024×1024~~ | **1024×1024** | Restaurado para SDXL nativo |
| **Modo Padrão** | Tiled | **Single Pass** | Mais rápido, sem emendas |
| **Multi-band** | Sempre usado | **Só >1024px** | Overhead desnecessário |
| **Tempo típico** | ~30s | **~8-15s** | Single Pass é mais eficiente |
| **Differential Diffusion** | Planejado | ✅ **Implementado** | Change Maps aplicadas |
| **Paletas CIELAB** | Extraídas | ✅ **Usadas em prompts** | Consistência de cores |
| **Regional IP-Adapter** | Planejado | 🕐 **Futuro** | Biblioteca limitada |
| **Temporal Decay** | Planejado | 🕐 **Futuro** | Pipeline não expõe controle |

> **💡 Nota:** A mudança para Single Pass foi intencional. A RTX 3060 (12GB) tem VRAM suficiente para páginas de mangá típicas (~1024×1408), tornando o modo Tiled desnecessário na maioria dos casos.

## 🔧 Correções e Melhorias

### v2.4.1 - Correções de Estabilização
| Problema | Causa | Solução |
|----------|-------|---------|
| Erro `name 'original_image' is not defined` | Variável não definida após refatoração | Renomeada para `original_image_resized` com inicialização correta |
| Upscale preventivo não aplicado | Erro anterior interrompia fluxo | Correção do fluxo de redimensionamento preservando upscale |

### v2.4 - Correções de Text Compositing e Substituição de Imagens
| Problema | Causa | Solução |
|----------|-------|---------|
| Flag `text_compositing` ignorada | Chamada incondicional no pipeline | Adicionada verificação da flag em `generate_image()` |
| Crop em coordenadas erradas | Redimensionamento da imagem sem ajustar bbox | Implementada conversão de coordenadas com fatores de escala |
| Crop de área aleatória | Uso de imagem redimensionada para crop | Agora usa imagem original preservada sem redimensionar |
| Imagens não substituídas no navegador | Mapeamento por índice, ordem diferente | Implementado mapeamento por `src` da imagem |
| Content script não injetado | Tentativa de mensagem sem garantir injeção | Adicionada injeção explícita antes do `sendMessage` |

### v2.3 - Two-Pass System
| Problema | Causa | Solução |
|----------|-------|---------|
| Inconsistência entre páginas | Recálculo de embeddings | **Cache imutável** no Pass 1 |
| Personagens duplicados | NMS binário simples | **CannyContinuityNMS** com merge |
| Cores inconsistentes | Sem extração de paletas | **PaletteExtractor CIELAB** |
| Vazamento de identidade | IP-Adapter global | **Background Isolation** (força 0) |
| VRAM estourando | Todos embeddings carregados | **Top-K limit** (máx 2 por tile) |

### v2.2 - Problemas de Qualidade (Resolvidos)
| Problema | Causa | Solução |
|----------|-------|---------|
| Artefatos de grade | VAE tiling | Desativado para RTX 3060 |
| Texto destruído | IA colorizando balões | Text compositing (padding 6px) |
| Cores lavadas | Guidance 0.0 | Guidance 1.2 |
| Rostos derretidos | ControlNet 0.65 | ControlNet 0.85 |
| Imagem pequena | Poucos pixels | Upscale preventivo <1024px |

---

## 📊 Performance (RTX 3060 12GB)

### Modo Single Tile (Padrão - Páginas ≤1024px)

| Métrica | Valor |
|---------|-------|
| **Tempo por página** | ~8-15 segundos |
| **VRAM uso** | ~10GB |
| **Qualidade** | Sem emendas visíveis |
| **Uso** | 99% dos mangás típicos |

### Modo Multi-Tile (Fallback - Páginas >1024px)

| Métrica | Valor |
|---------|-------|
| **Tempo por página** | ~30-45 segundos |
| **VRAM uso** | ~8GB (libera entre tiles) |
| **Qualidade** | Possíveis linhas de emenda |
| **Uso** | Spreads, posters, páginas duplas |

### Cache e Armazenamento

| Métrica | Valor |
|---------|-------|
| Cache Pass 1 | ~50MB/página |
| Embeddings | ~5MB/personagem |
| Máx personagens/tile | 2 (limite de atenção) |

---

## 🧪 Testes Implementados

### Testes Unitários
```bash
python -m pytest tests/test_implementations.py -v
```
- ✅ PaletteExtractor (4 testes)
- ✅ CannyContinuityNMS (5 testes)
- ✅ Database com Paletas (2 testes)
- ✅ Pass1Analyzer (4 testes)
- ✅ Pass2Generator (1 teste)
- ✅ End-to-End (2 testes)

### Testes de Integração
```bash
python tests/test_integration_pass1.py
python tests/test_pass2_basic.py
```
- ✅ Pass 1 completo com análise real
- ✅ Pass 2 com background isolation
- ✅ Database persistence

---

## 🛠️ Scripts Disponíveis

### `scripts/windows/`
- `start_server.bat` - Inicia servidor API
- `start_server_debug.bat` - Modo debug com logs detalhados
- `check_and_install_deps.bat` - Instala dependências
- `diagnose.bat` - Diagnóstico completo
- `fix_numpy.bat` - Corrige versão do NumPy

---

## 📝 Próximos Passos (Roadmap)

### v2.8 (Planejado)
- [ ] **Interface Web (GUI):** Dashboard para gerenciamento de capítulos e reviews.
- [ ] **Advanced LoRA:** Suporte a LoRAs externos para estilos de arte específicos.
- [ ] **Refinement Loop:** Pipeline de inpainting automático para rostos pequenos.
- [ ] **Multi-Scale Point Matching:** Matching hierárquico de keypoints.
- [ ] **Adaptive Thresholds:** Thresholds scene-aware para PCTC.

### v3.0 (Futuro)
- [ ] **Flux Architecture:** Migração para modelos Flux (se viável na 3060).
- [ ] **Video Support:** Colorização de anime frame-a-frame.
- [ ] **Real-time 30fps:** Otimização extrema com TensorRT.
- [ ] **Upscaler AI:** Integração com Real-ESRGAN/SwinIR.

---

## 📁 Arquivos Principais

```
MANGACOLOR/
├── README.md                    # Documentação principal
├── CHANGELOG.md                 # Histórico de mudanças
├── PROGRESSO.md                 # Este arquivo
├── config/settings.py           # Configurações
├── core/
│   ├── detection/
│   │   └── yolo_detector.py        # Manga109 YOLO
│   ├── generation/
│   │   ├── pipeline.py             # TileAwareGenerator
│   │   ├── tiling.py               # TilingManager
│   │   ├── text_compositor.py      # TextCompositor
│   │   └── prompt_builder.py       # MangaPromptBuilder
│   ├── identity/
│   │   ├── hybrid_encoder.py       # CLIP + ArcFace
│   │   └── palette_manager.py      # Paletas CIELAB
│   ├── database/
│   │   └── chapter_db.py           # FAISS + Parquet
│   ├── domain/                     # Camada de Domínio
│   ├── pass1_analyzer.py           # Análise Two-Pass (Main)
│   └── pass2_generator.py          # Geração Two-Pass (Main)
├── api/routes/chapter/
│   └── twopass.py                  # API REST Two-Pass
├── tests/
│   ├── unit/                       # Testes isolados
│   ├── integration/                # Testes com IO
│   └── e2e/                        # Simulação completa
└── browser_extension/             # Chrome/Edge
    ├── content_script.js
    └── popup.js
```

---

<p align="center">
  ✅ Sistema Two-Pass com ADR 004/005 pronto para produção (v2.7)
</p>

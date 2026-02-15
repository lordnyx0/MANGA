# 🏗️ Arquitetura MangaAutoColor Pro v3.0 (ADR 006)

## Visão Geral

O MangaAutoColor Pro utiliza uma arquitetura **Two-Pass** enterprise-grade, otimizada para **GPU Consumer (RTX 3060 12GB)**. A arquitetura evoluiu para focar em **Modularidade**, **Testabilidade** e **Controle Regional**.

### 🌟 Pilares da Arquitetura v3.0
1.  **Pipeline Two-Pass**: Análise (CPU) e Geração (GPU/VRAM).
2.  **Engine V3 (SD 1.5)**: Alta fidelidade de traço com ControlNet Lineart e Multiply Mode.
3.  **Global Identity**: Consistência via IP-Adapter com referência visual única por personagem.
4.  **Data-Driven**: Persistência robusta (Parquet + FAISS) e logs estruturados.
5.  **Quality Assurance**: Suite de testes automatizada com **AVQV (Automated Visual Quality Validation)**.

---

## 🏗️ Diagrama de Estrutura (Componentes)

```mermaid
graph TD
    classDef core fill:#e1f5fe,stroke:#01579b
    classDef service fill:#f3e5f5,stroke:#4a148c
    classDef data fill:#e8f5e9,stroke:#1b5e20
    classDef ext fill:#fff3e0,stroke:#e65100

    User[User Input] -->|Chapter| API[FastAPI / CLI]
    API --> Pipeline[MangaColorizationPipeline]:::core

    subgraph "Pass 1: Analysis (CPU Bound)"
        Pipeline --> P1[Pass1Analyzer]:::core
        P1 --> YOLO[YOLODetector (Manga109)]:::ext
        P1 --> Scene[SceneDetector (Narrative)]:::service
        P1 --> Palette[PaletteExtractor (CIELAB)]:::service
        P1 --> ID[HybridIdentityEncoder (CLIP+Face)]:::service
        
        P1 --> DB[(ChapterDatabase)]:::data
    end

    subgraph "Data Layer"
        DB --> HS[CharacterService]:::service
        DB --> NS[NarrativeService]:::service
        DB --> TS[TileService]:::service
        DB --> VI[VectorIndex (FAISS)]:::data
        DB --> PQ[Parquet Metadata]:::data
    end

    subgraph "Pass 2: Generation (GPU Bound)"
        Pipeline --> P2[Pass2Generator]:::core
        P2 --> E3[SD15LineartEngine]:::core
        
        E3 --> PB[MangaPromptBuilder]:::service
        E3 --> TM[TilingManager]:::service
        E3 --> TC[TextCompositor]:::service
        
        E3 --> SD15[SD 1.5 UNet]:::ext
        E3 --> IP[IP-Adapter Plus]:::ext
        E3 --> CN[ControlNet Lineart]:::ext
    end

    SD15LineartEngine -->|Bubble Masking| E3
    E3 -->|Gaussian Blur| E3
    E3 -->|Multiply Blend| Output[Colorized Page]
```

## 🔄 Fluxo de Processamento (Pipeline)

### Passo 1: Análise e Enriquecimento
O objetivo é extrair **todo** o contexto necessário antes de tocar na GPU.

1.  **Ingestão Híbrida**: O endpoint `/chapter/analyze` aceita tanto páginas P/B quanto referências coloridas (opcional).
2.  **Detecção (YOLO)**: Identifica `body`, `face`, `frame`, e `text`.
    *   *Nota*: Balões de texto (class 3) são preservados explicitamente.
3.  **Narrative Context**: Classifica a cena (ex: "flashback", "night", "outdoors") via `SceneDetector`.
4.  **Identidade Híbrida**: Extrai embeddings CLIP (global) e ArcFace (facial) para cada personagem.
5.  **Consolidação (Clustering)**: `CharacterService` agrupa detecções do mesmo personagem usando FAISS, unificando referências coloridas (se houver) com ocorrências nas páginas.
6.  **Persistência**: Tudo é salvo em `output/{chapter_id}/cache/`.

### Passo 2: Geração Tile-Aware
A geração é agnóstica à resolução e focada em eficiência de VRAM.

1.  **Prompt Building**: `MangaPromptBuilder` constrói prompts baseados em:
    *   Descrição da Cena (`SceneType`)
    *   Paletas de Cores (CIELAB)
    *   Style Presets (Config)
2.  **Generation Strategy**:
    *   **Single Pass (v3.0)**: Processamento de página inteira (Full Page) para máxima coerência global.
    *   **Resolution Handling**: SD 1.5 é nativo em 512px. Geração em 1024px (Single Pass) depende fortemente do **ControlNet** para evitar duplicação de estruturas (ex: dois corpos).
    *   *Nota*: V3.1 trará Tiling real para mitigar riscos de alucinação em alta resolução.
3.  **Identity Strategy (Global & Regional)**:
    *   **Regional IP-Adapter**: Suporta múltiplos personagens por tile. O `Pass2Generator` cria máscaras de atenção baseadas nos BBoxes, garantindo que cada referência visual condicione apenas a região correta.
    *   **Dynamic Control**: Aplica-se apenas nos primeiros 60% dos steps (configurável via `IP_ADAPTER_END_STEP`) para garantir estrutura sem comprometer detalhes finos.
    *   **Text Prompt Augmentation**: O `MangaPromptBuilder` usa o `PaletteExtractor` (CIELAB) para converter cores das referências em texto (ex: "blue hair"), reforçando a consistência.
4.  **Compositing & Bubble Masking**:
    *   **Bubble Masking**: O motor identifica regiões de texto via YOLO e as limpa (preenche com branco puro) na camada de cor gerada. Isso elimina "ghosting" e cores indesejadas dentro dos balões.
    *   **Soft Composition**: Aplica-se um leve Gaussian Blur (radius=0.5) na camada de cor antes do Multiply. Isso suaviza halos e "serrilhados" na intersecção entre cores e linhas.
    *   **Text Restoration**: O `TextCompositor` restaura o texto original com nitidez total.

---

## 🎨 Estratégia de Colorização

A engine v3.0 decide dinamicamente a fonte das cores baseada na disponibilidade de referências:

### 1. Com Referência (Character-Driven)
Quando o usuário faz upload de imagens coloridas junto com o capítulo (via **Extension UI** ou API):
*   **Ingestão:** O sistema recebe referências em campo separado da API (`color_references`), evitando confusão com páginas do mangá.
*   **Matching:** **Automático (Threshold 0.95)**. O `CharacterService` usa FAISS para agrupar referências aos personagens detectados.
    *   *Nota*: Não há interface manual de correção nesta versão.
*   **Geração:** O IP-Adapter recebe a imagem de referência do cluster para guiar a colorização daquele personagem específico.
*   **Resultado:** Consistência visual mantida (roupas, cabelo, pele) através das páginas.

### 2. Sem Referência (Zero-Shot / Style Preset)
Quando nenhuma referência é fornecida:
*   **Ingestão:** Apenas páginas P/B são analisadas.
*   **Colorização:** O sistema utiliza `ControlNet Lineart` + `Prompt Engineering`.
*   **Style Presets:** O usuário escolhe um preset (ex: "vibrant", "muted", "pastel") no momento da geração.
*   **Resultado:** A IA "alucina" cores coerentes com o estilo escolhido, mantendo o traço original perfeito, mas sem garantia de consistência de cores específicas entre páginas (ex: a camisa pode mudar de cor se não houver referência).

---

## 🧩 Componentes Chave (Decoupled Services)

A partir da v2.6, classes monolíticas foram refatoradas em serviços especializados:

### 1. SD15LineartEngine (`core/generation/engines/sd15_lineart_engine.py`)
Novo motor de geração baseado em SD 1.5.
*   **Model**: `runwayml/stable-diffusion-v1-5`.
*   **ControlNet**: `lllyasviel/control_v11p_sd15s2_lineart_anime` (Específico para Anime).
*   **Preprocessor**: Nenhum (Implicit). O sistema assume que a entrada já é um Lineart (Manga P/B), alimentando a imagem original diretamente no ControlNet.
*   **Feature**: Preservação perfeita de traço via ControlNet Lineart + Multiply Mode.
*   **Mecanismo**: Inpainting regional + Composição final em modo Multiply.
*   **Consistência**: IP-Adapter global por personagem (referência visual).

### 2. TilingManager (`core/generation/tiling.py`)
(Em desenvolvimento para v3.1)
*   Planejado para gerenciar subdivisão de páginas 4K+.
*   Atualmente, o sistema opera em modo **Single Tile** (Full Page) para garantir coerência.
*   Gerencia o "Change Map" (máscaras Gaussianas para blending).
*   Filtra quais personagens aparecem em qual tile.

### 3. TextCompositor (`core/generation/text_compositor.py`)
Responsável pela preservação de texto (SRP).
*   Recebe a imagem original e a gerada.
*   Recebe máscaras de texto (do YOLO).
*   Aplica `seamlessClone` ou alpha blending para restaurar o texto com nitidez perfeita.

### 4. MangaPromptBuilder (`core/generation/prompt_builder.py`)
Encapsula a lógica de engenharia de prompt.
*   Converte paletas CIELAB/HSL em nomes de cores.
*   Aplica modificadores de cena e estilos.

### 5. ScenePaletteService (`core/generation/scene_palette_service.py`)
Novo em v3.0.
*   **Responsabilidade**: Garantir consistência determinística para coadjuvantes (Zero-Shot).
*   **Mecanismo**: Hash(char_id) -> HSL -> Harmonização com Protagonistas -> Prompt Injection.
*   **Persistência**: `scene_palette.json` por capítulo.

### 6. AVQV: Automated Visual Quality Validation (`tests/integration/test_visual_quality_regression.py`)
Novo framework de testes para prevenir regressões visuais subjetivas.
*   **Métrica: Bubble Purity**: Analisa a variância de cor em regiões de texto. (Detecta balões sujos).
*   **Métrica: Edge Neutrality**: Compara a crominância nas bordas vs centro para detectar artefatos de VAE Tiling.
*   **Métrica: Dynamic Range**: Verifica picos de histograma para detectar "solarização".

---

## 🧪 Estratégia de Testes (Quality Assurance)

A suite de testes (v2.6.3) garante estabilidade e previne regressões.

### Estrutura
*   `tests/unit/`: Testes isolados de componentes (sem IO/GPU).
    *   Ex: `test_text_compositor.py`, `test_prompt_builder.py`.
*   `tests/integration/`: Testes de componentes reais com IO (File system/Database).
    *   Ex: `test_pass1.py` (roda análise real), `test_chapter_db.py`.
*   `tests/e2e/`: Simulação completa do pipeline (Mocked Models).
    *   Ex: `test_pipeline.py` (Simula Pass 1 -> Pass 2).

### Execução
O script `run_tests.bat` orquestra a execução no ambiente correto (`venv`):
```batch
run_tests.bat  # Roda Unit, Integration e E2E em sequência
```

---

## 📊 Performance (Benchmark RTX 3060 - v3.0)

| Modo | Resolução | Tempo Médio | VRAM |
|------|-----------|-------------|------|
| **Single Tile** | 1024x1408 | ~25s | ~8.0 GB |
| **Multi-Tile** | 2048x2816 | ~90s | ~8.5 GB |

*   **Nota**: Aumento de tempo justificado pelo salto dramático na qualidade (4 steps -> 20 steps).
*   **VRAM**: Consumo menor que SDXL, permitindo maior estabilidade.

*   **VRAM Management**: O sistema usa `enable_model_cpu_offload()` agressivo. O pico de VRAM ocorre durante o decode VAE.
*   **Concurrency**: O processamento é sequencial por GPU, mas thread-safe para API server.

---

## 📅 Histórico de Mudanças Arquiteturais

| Versão | Mudança Principal | Motivo |
|--------|-------------------|--------|
| **v2.0** | Two-Pass Architecture | Separar IO de GPU, permitir cache. |
| **v2.3** | Single Tile Optimization | SDXL nativo (1024px) é melhor que Tiling forçado. |
| **v2.5** | Regional IP-Adapter | Resolver "color bleeding" entre personagens. |
| **v2.6** | Decoupled Services | Reduzir complexidade ciclomática e acoplamento. |
| **v2.6.3**| Test Suite Upgrade | Garantir estabilidade em produção. |
| **v2.6.4**| Z-Ordering Anti-Halo | Solução para overlap de máscaras via subtração binária e blur final. |
| **v2.7** | ADR 005 - PCTC | Point Correspondence & Temporal Consistency. |
| **v3.0** | ADR 006 - Engine Replacement | Troca de SDXL por SD 1.5 + Lineart (Multiply Mode). Remoção de ADR 005. |

## ✅ Implementado: ADR 004 & ADR 006

### ADR 004: Segmentação Semântica (SAM 2.1) & Z-Buffer ✅
**Status:** IMPLEMENTADO (v2.6.5)

*   **SAM 2.1 Tiny:** Segmentação densa edge-preserving (35MB).
*   **Z-Buffer Hierárquico:** Ordenação de profundidade automática (Y + Área + Tipo).
*   **Documentação:** [ADR 004](ADR_004_SAM2_Segmentation.md).

### ADR 006: Engine Replacement (SD 1.5 + Lineart) ✅
**Status:** IMPLEMENTADO (v3.0.0)

*   **Engine:** SD 1.5 Base + ControlNet Anime Lineart.
*   **Composição:** Inpainting + Multiply Blending para preto perfeito.
*   **Consistência:** IP-Adapter Global.
*   **Documentação:** [ADR 006](ADR%20006).

### 🚫 Removido: ADR 005 (PCTC)
**Status:** REMOVIDO via ADR 006. Funcionalidades consideradas desnecessárias para o novo motor.

---

---

## ⚙️ Especificações Técnicas Críticas (V3.0)

### 1. Dependências e Compatibilidade
*   **Diffusers**: `>0.27.0` (Obrigatório para `ip_adapter_masks`). Versões anteriores causam `TypeError`.
*   **PyTorch**: `>2.0.0` (Recomendado para otimizações de memória).
*   **VRAM**: 
    *   **Mínimo**: 8GB (Single Reference).
    *   **Recomendado**: 12GB (Dual Reference + Regional).
    *   **Limite**: 2 Referências simultâneas por tile em 12GB. Acima disso, o sistema ativa fallback sequencial.
*   **Disk Space**: ~5.5GB (SD 1.5: 4GB, ControlNet: 723MB, IP-Adapter: 400MB).

### 2. Formato de Dados
*   **Máscaras IP-Adapter**:
    *   **Dimensão**: 64x64 (Latent Space do SD 1.5).
    *   **Tipo**: `torch.float32` (Suavizadas, range 0.0-1.0).
    *   **Redimensionamento**: O engine redimensiona máscaras 512x512 automaticamente usando `Image.NEAREST`.
*   **Imagens de Referência**:
    *   **Resolução Ideal**: 224x224 (CLIP padrão) ou 512x512.
    *   **Aspect Ratio**: Quadrada (1:1). Imagens retangulares sofrem squeeze/distortion.
    *   **Conteúdo**: Close-up de rosto (ArcFace) + Torso superior (CLIP/IP-Adapter) para melhor fidelidade.

### 3. Pipeline Gráfico e Determinismo
*   **Seed**: O sistema é determinístico se `seed` for fornecido. O `SD15LineartEngine` instancia `torch.Generator("cpu").manual_seed(seed)` para garantir reprodutibilidade.
*   **Ordem de Composição**:
    1.  **Geração**: SD 1.5 + Lineart + IP-Adapter -> RGB (Base Color).
    2.  **Multiply Blend**: `Base Color * Original Lineart` -> Preserva pretos absolutos.
    3.  **Text Compositing**: Restauração de balões (Original) sobre a imagem colorida.
*   **Lineart Preprocessing**:
    *   O engine **inverte automaticamente** imagens de mangá (Preto no Branco) para o formato esperado pelo ControlNet (Branco no Preto) se a média de brilho for > 127.

### 4. Limites e Escalabilidade
*   **Personagens por Capítulo**: Limitado pela memória RAM no Clustering (FAISS). 
    *   **Threshold de Merge Identity**: 0.95 (Cosine Similarity).
    *   **Edge Case**: Personagens sem rosto/costas dependem puramente do pipeline `ScenePalette` (Prompt Injection).
*   **Persistência**:
    *   `scene_palette.json`: Salvo em `output/debug/` (V3 Debug) ou diretório do capítulo.
    *   Cache de Embeddings: `data/cache/*.npy`.

### 5. Defensive Engineering & Adapters
*   **VAEDtypeAdapter** (`core/generation/engines/vae_dtype_adapter.py`):
    *   **Problema**: Em ambientes Windows + CUDA + SD 1.5 (FP16 Pipeline), o VAE (FP32) falha com `RuntimeError: Input type (struct c10::Half) and bias type (float) should be the same` durante o decode, mesmo com `vae.config.force_upcast=True` (bug/limitação de versões específicas do diffusers/torch).
    *   **Solução**: Wrapper implementado via Context Manager que intercepta a chamada `vae.decode`.
    *   **Mecanismo**: Verifica se os latents de entrada estão no mesmo dtype do VAE. Se não, realiza cast explícito (`latents.to(vae.dtype)`) antes de prosseguir.
    *   **Uso**: O `SD15LineartEngine` envolve a chamada de geração com `with VAEDtypeAdapter(self.pipe.vae):`. Isso isola o "fix" e evita monkey patching global destrutivo.

---


---

*Documento atualizado em: 14/02/2026 (v3.0.1 + Bubble Masking + AVQV)*

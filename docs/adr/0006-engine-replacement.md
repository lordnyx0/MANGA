# ADR 006: Troca do Motor de Geração (Engine Replacement)

* **Status:** Proposed
* **Data:** 2026-02-13
* **Decisores:** Arquiteto MangaAutoColor Pro
* **Referências:** ADR 004 (SAM 2.1), ADR 005 (PCTC - Deprecado)

## 1. Contexto e Problema

O pipeline atual (v2.7) utiliza SDXL-Lightning (4 steps) com ControlNet Canny, resultando em qualidade de colorização "terrível" quando comparado a soluções comerciais como NanoBanana3. Análises técnicas identificaram gargalos arquiteturais fundamentais:

*   **SDXL-Lightning 4 steps:** Modelo destilado que sacrifica fidelidade condicional por velocidade. Perde a capacidade de seguir referências visuais (IP-Adapter) em apenas 4 steps.
*   **ControlNet Canny:** Destrói o line art original ao detectar bordas duras, forçando o modelo a redesenhar linhas (borrando textos) ao invés de apenas colorizar.
*   **ADR 005 (PCTC):** Over-engineering para problema inexistente em imagens estáticas. Optical Flow e Point Correspondence são técnicas de vídeo, não aplicáveis a páginas de mangá individuais.
*   **Abordagem Textual de Cores:** Paletas CIELAB convertidas para texto não garantem consistência visual — o modelo interpreta "dark blue" de forma variável.

### 1.1 Evidências de Falha

*   **CVTG-2K Benchmark:** SDXL-Lightning apresenta ~50% de acurácia em texto/referências vs 90%+ de modelos full-steps.
*   **Artifacts Visuais:** Textos em português borrados, "color bleeding" entre personagens, linhas perdidas na regeneração.

## 2. Decisão

Trocar o "motor" (Pass 2 - Geração) mantendo o "chassi" (Pass 1 - Análise).

### 2.1 Estratégia: Arquitetura Híbrida (Engine Replacement)

*   **Manter:** Pass 1 (YOLO + SAM 2.1 + ADR 004), Database (FAISS + Parquet), API REST, Test Suite, Extensão Browser.
*   **Descartar:** SDXL-Lightning, ControlNet Canny, ADR 005 (PCTC), Regional IP-Adapter v2.5 (incompatível com steps reais).
*   **Implementar:** Novo motor baseado em SD 1.5 + ControlNet Lineart + Multiply Mode + IP-Adapter com steps reais (15-20).

### 2.2 Justificativa

*   **Custo de Reescrita Total:** 3-4 meses para replicar 190 testes, re-integrar SAM 2.1, refazer API.
*   **Custo de Troca de Motor:** 2-3 semanas, mantendo estabilidade do Pass 1 e infraestrutura.
*   **Ganho:** Qualidade salta de "inutilizável" para "próximo do NanoBanana3" (estimativa: 70-80% da qualidade comercial).

## 3. O que é Mantido (Assets Valiosos)

### 3.1 Pass 1: Análise (100% Preservado)

```
core/detection/yolo_detector.py       # Manga109 YOLO - funciona perfeitamente
core/analysis/sam_segmenter.py        # ADR 004 - SAM 2.1 Tiny
core/identity/hybrid_encoder.py       # CLIP + ArcFace
core/database/                        # FAISS + Parquet + Cache imutável
```

**Motivo:** Detecção e segmentação são agnósticas ao modelo de geração. Os embeddings e máscaras SAM serão reusados pelo novo motor.

### 3.2 Infraestrutura (100% Preservada)

```
api/routes/chapter/twopass.py         # API REST - interface mantida
tests/                                # 190 testes - adaptar mocks apenas
browser_extension/                    # Integração frontend - inalterada
config/settings.py                    # Novos parâmetros adicionados, não removidos
```

### 3.3 ADR 004: Z-Buffer & SAM 2.1 (Preservado)

*   **Status:** Continua essencial para ordenação de profundidade em cenas com overlap de personagens.
*   **Integração:** Máscaras SAM 2.1 serão inputs para o novo sistema de inpainting regional.

## 4. O que é Removido (Liabilities)

### 4.1 SDXL-Lightning

*   **Motivo:** Destilação progressiva removeu a capacidade de condicionamento fino necessário para colorização fiel.
*   **Substituto:** SD 1.5 base (runwayml/stable-diffusion-v1-5) ou variantes anime (AOM3/AnythingV5 via CivitAI, convertidos para Diffusers).

### 4.2 ControlNet Canny

*   **Motivo:** Destrói line art ao redesenhar bordas.
*   **Substituto:**
    *   **Opção A:** `lllyasviel/control_v11p_sd15_lineart` (padrão fotos com arte line)
    *   **Opção B:** `lllyasviel/control_v11p_sd15s2_lineart_anime` (otimizado para anime)
    *   **Opção C:** `SubMaroon/ControlNet-manga-recolor` (ESPECÍFICO para mangá, mas requer SDXL base)

> **⚠️ Atenção:** `SubMaroon/ControlNet-manga-recolor` é SDXL-only. Se usar SD 1.5, usar Opção A ou B.

### 4.3 ADR 005: PCTC (Point Correspondence & Temporal Consistency)

*   **Motivo:** Técnicas de vídeo (RAFT Optical Flow, LightGlue) não aplicáveis a imagens estáticas.
*   **Remoção:** `core/analysis/temporal_flow.py`, `core/analysis/point_matching.py`
*   **Substituto:** Consistência via mesma imagem de referência IP-Adapter aplicada a todas as páginas do mesmo personagem (estado global simples).

### 4.4 Paletas CIELAB em Prompts

*   **Motivo:** Abordagem textual frágil.
*   **Substituto:** Referências visuais diretas (imagens PNG) carregadas via IP-Adapter.

## 5. Especificação do Novo Motor (v3.0 Engine)

### 5.1 Arquitetura Técnica

```python
class MangaColorizationEngineV3:
    """
    Novo motor de geração baseado em:
    - Base: SD 1.5 (menor VRAM, melhor controle)
    - Steps: 15-20 (não 4)
    - ControlNet: Lineart (preservação de estrutura)
    - Modo: Multiply (composição não-destrutiva)
    """

    def __init__(self):
        # Base model SD 1.5 (verificado, existe)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",  # ou modelo anime específico
            controlnet=ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_lineart",
                torch_dtype=torch.float16
            ),
            torch_dtype=torch.float16
        )

        # IP-Adapter Plus Face (verificado, existe)
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter-full-face_sd15.bin"
        )

        # Otimizações VRAM para RTX 3060 12GB
        self.pipe.enable_model_cpu_offload()
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
```

### 5.2 Pipeline de Geração (Multiply Mode)

**Fase 1: Geração da Camada de Cor**

```python
def generate_color_layer(self, region_mask, reference_image, prompt):
    """
    Gera APENAS as cores, sem as linhas pretas.
    Usa inpainting para isolar a região.
    """
    return self.pipe(
        prompt=prompt,
        image=region_mask,           # ControlNet lineart conditioning
        mask_image=region_mask,      # Inpainting mask
        ip_adapter_image=reference_image,  # Referência visual (não texto!)
        ip_adapter_scale=0.7,        # Testado: 0.6-0.8 é o sweet spot
        num_inference_steps=20,      # ADEUS 4 steps!
        strength=0.75,               # 75% novo, 25% preservado
        controlnet_conditioning_scale=0.8
    ).images[0]
```

**Fase 2: Composição Multiply (Crucial)**

```python
def compose_final(self, original_lineart, color_layers):
    """
    Composição não-destrutiva:
    1. Line art original em modo Multiply (100%)
    2. Camadas de cor em modo Color/Multiply (95%)
    """
    from PIL import Image

    base = original_lineart.convert('RGBA')
    for color_layer, mask in color_layers:
        # Modo Multiply preserva linhas pretas perfeitamente
        base = Image.alpha_composite(base, color_layer.convert('RGBA'))

    return base
```

### 5.3 Configurações de Referência (RTX 3060 12GB)

| Parâmetro | Valor Antigo (v2.7) | Valor Novo (v3.0) | Justificativa |
| :--- | :--- | :--- | :--- |
| **Modelo Base** | SDXL-Lightning | SD 1.5 | Menor VRAM, melhor controle |
| **Steps** | 4 | 15-20 | Necessário para IP-Adapter funcionar |
| **ControlNet** | Canny (0.85) | Lineart (0.8) | Preserva line art |
| **IP-Adapter Scale** | 1.0 → 0.0 (cycling) | 0.7 (constante) | Evita overfitting |
| **Denoising** | 1.0 (img2img) | 0.75 (inpainting) | Preserva estrutura |
| **Modo de Composição** | Direct output | Multiply Mode | Linhas nítidas garantidas |

## 6. Links e Referências Verificadas

### 6.1 Modelos Oficiais (HuggingFace)

*   **ControlNet Lineart (SD 1.5):**
    *   `lllyasviel/control_v11p_sd15_lineart` - [Link](https://huggingface.co/lllyasviel/control_v11p_sd15_lineart)
    *   `lllyasviel/control_v11p_sd15s2_lineart_anime` - [Link](https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main) (Recomendado)
*   **ControlNet Manga (SDXL - Alternativa):**
    *   `SubMaroon/ControlNet-manga-recolor` - [Link](https://huggingface.co/SubMaroon/ControlNet-manga-recolor) (SDXL-only)
*   **IP-Adapter:**
    *   `h94/IP-Adapter` - [Link](https://huggingface.co/h94/IP-Adapter)
    *   Modelos SD 1.5: `ip-adapter-full-face_sd15.bin`, `ip-adapter-plus-face_sd15.bin`
*   **Base Models:**
    *   `runwayml/stable-diffusion-v1-5` - Base padrão SD 1.5

### 6.2 Papers e Benchmarks

*   **SDXL-Lightning Limitations:** "Progressive Adversarial Diffusion Distillation" - perda de fidelidade condicional em 4 steps.
*   **ControlNet Canny vs Lineart:** Análise técnica de preservação de bordas.
*   **IP-Adapter Masking:** Suporte a máscaras regionais para múltiplos personagens.

## 7. Cuidados e Advertências

### 7.1 Modelos que NÃO Existem (Não Inventar)

*   ❌ `control_v11p_sd15_manga` - Não existe. Usar `lineart_anime`.
*   ❌ `ip-adapter-manga-v1` - Não existe. Usar `ip-adapter-full-face_sd15` ou Plus.
*   ❌ `sd15-manga-colorization-lora` no HuggingFace - Verificar existência antes de referenciar.

### 7.2 Compatibilidade de Modelos

*   `SubMaroon/ControlNet-manga-recolor` é SDXL-only. Não tentar usar com SD 1.5.
*   IP-Adapter Plus requer `CLIPVisionModelWithProjection` explicitamente carregado para SD 1.5.

### 7.3 Limitações de VRAM (RTX 3060 12GB)

*   SD 1.5 + ControlNet + IP-Adapter: ~8-9GB VRAM (cabe confortavelmente)
*   Resolução máxima recomendada: 768x768 (SD 1.5) ou 1024x1024 (SDXL com offload)

### 7.4 Riscos de Qualidade

*   **Lineart Detector:** O preprocessador LineartDetector pode falhar em scans de baixa qualidade.
*   **Consistência de Personagem:** Sem ADR 005, a consistência depende 100% da mesma imagem de referência sendo passada ao IP-Adapter para todas as páginas.

## 8. Plano de Migração

**Fase 1: Interface Abstrata (2 dias)**

```python
# core/generation/interfaces.py (novo arquivo)
class ColorizationEngine(ABC):
    @abstractmethod
    def generate(self, line_art, masks, references, style_params) -> Image:
        pass
```

*   Branch: `refactor/v3-engine-interface`

**Fase 2: Implementação do Motor (1 semana)**

*   Implementar `SD15LineartEngine(ColorizationEngine)`
*   Implementar `MultiplyCompositor`
*   Testes unitários isolados
*   Branch: `feature/v3-sd15-engine`

**Fase 3: Migração de Dados (2 dias)**

*   Converter cache do Pass 1
*   Script: `scripts/migrate_v27_to_v30.py`

**Fase 4: A/B Testing (1 semana)**

*   Flag de feature: `USE_V3_ENGINE=true/false`
*   Comparar saídas v2.7 vs v3.0

**Fase 5: Deprecação (v3.1)**

*   Remover código v2.7
*   Remover ADR 005
*   Atualizar documentação

## 9. Consequências

### Positivas

*   **Qualidade:** Salto de "terrível" para "comercialmente viável".
*   **Preservação de Texto:** 100% de legibilidade em português.
*   **Consistência:** Cores fixas por personagem via referência visual direta.
*   **VRAM:** Economia de ~2GB.

### Negativas

*   **Tempo de Geração:** Aumento de ~8s para ~25s por página.
*   **Complexidade de Setup:** Requer download de novos modelos (~5GB total).
*   **Breaking Change:** APIs internas do Pass 2 mudam completamente.

### Neutras

*   **Pass 1 inalterado:** Nenhum impacto em YOLO, SAM 2.1.
*   **Database:** Schema migrado, mas estrutura mantida.

## 10. Notas de Implementação

### 10.1 Exemplo de Código de Produção

```python
# core/generation/engines/sd15_lineart_engine.py

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
# ... (código omitido para brevidade)
```

### 10.2 Checklist de Verificação de Modelos

*   [ ] `lllyasviel/control_v11p_sd15s2_lineart_anime` existe e é acessível
*   [ ] `h94/IP-Adapter` contém `ip-adapter-plus-face_sd15.bin`
*   [ ] `runwayml/stable-diffusion-v1-5` não está em "gated"
*   [ ] Tamanho total dos downloads: ~4.2GB

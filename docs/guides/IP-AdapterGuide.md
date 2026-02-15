# Guia Técnico: Regional IP-Adapter com Early-Heavy Injection para MangaAutoColor Pro

> **Documento Técnico Revisado - Arquitetura Simplificada**
> 
> **Versão:** 2.0  
> **Data:** 2026-02-05  
> **Projeto:** MangaAutoColor Pro v2.5  
> **Status:** CORREÇÃO DE ARQUITETURA - Usar API nativa do Diffusers

---

## 📋 Índice

1. [Descobertas Científicas](#descobertas-científicas)
2. [Correção de Arquitetura](#correção-de-arquitetura)
3. [Especificação Técnica](#especificação-técnica)
4. [Algoritmo de Early-Heavy Injection](#algoritmo-de-early-heavy-injection)
5. [Integração com Pipeline](#integração-com-pipeline-existente)
6. [Gestão de VRAM](#gestão-de-vram)
7. [Checklist de Implementação](#checklist-de-implementação)
8. [Referências](#referências)

---

## Descobertas Científicas

### 1.1 O Paradigma da Injeção Temporal (T-GATE - ICML 2024)

Pesquisa publicada no ICML 2024 demonstra que **cross-attention é necessária apenas nos primeiros 20% dos steps** (etapa de "semantics-planning"). Nos 80% finais ("fidelity-improving"), a injeção contínua é redundante e pode prejudicar a fidelidade estrutural.

**Implicação para SDXL-Lightning (4 steps):**

| Step | Porcentagem | Fase | Estratégia IP-Adapter |
|------|-------------|------|----------------------|
| 0 | 0-25% | Semantics-Planning | **Máxima força** (scale 1.0) - Define "quem" é o personagem |
| 1 | 25-50% | Transition | Redução (scale 0.6) ou alternância cíclica |
| 2 | 50-75% | Fidelity-Improving | **Desligado** (scale 0.0) - ControlNet domina |
| 3 | 75-100% | Refinement | **Desligado** (scale 0.0) - Finalização sem interferência |

### 1.2 Multi-Embedding Cíclico vs Simultâneo (ICAS 2025)

Estudos quantitativos mostram que injetar múltiplos embeddings **simultaneamente** (scale [0.5, 0.5]) causa:
- "Oversimplification" (simplificação excessiva)
- Vazamento de identidade entre personagens

A estratégia **cíclica** (alternar foco por step) preserva **40% mais características individuais** em cenários multi-subject.

---

## Correção de Arquitetura

### ❌ NÃO Implementar (Estratégia Anterior - Obsoleta)

```python
# NÃO CRIAR ESTA CLASSE - Reinventa a roda
class RegionalIPAdapterXL:
    def _modify_unet_attention(self): 
        # Monkey-patching desnecessário e arriscado
        ...
```

**Problemas da abordagem anterior:**
- Monkey-patching do UNet é instável
- Manutenção complexa com updates do Diffusers
- Reimplementa funcionalidade que já existe nativamente

### ✅ Implementar (Estratégia Baseada em Evidências)

Usar **API nativa do Diffusers ≥0.29.0** com `cross_attention_kwargs` e callback de otimização temporal:

```python
from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers import StableDiffusionXLControlNetPipeline
```

**Vantagens:**
- ✅ Código mantido pela HuggingFace
- ✅ Testado e otimizado
- ✅ Compatível com futuras versões
- ✅ Menos bugs, mais performance

---

## Especificação Técnica

### 2.1 Dependências

```bash
# requirements.txt - ATUALIZAR

diffusers>=0.29.0       # Necessário para ip_adapter_masks
accelerate>=0.20.0      # Para cpu_offload
torch>=2.0.0
```

### 2.2 ⚠️ Avisos Importantes sobre IP-Adapter Plus Face ViT-H

#### O "Pegadinha" Técnica - Encoder ViT-H

O modelo **ip-adapter-plus-face_sdxl_vit-h** usa o encoder **CLIP-ViT-H-14**, que é diferente do encoder padrão do SDXL (ViT-L-14).

> ⚠️ **Importante:** A biblioteca Diffusers moderna (≥0.29.0) baixa o encoder correto **automaticamente** quando você usa `load_ip_adapter()` apontando para `h94/IP-Adapter`. Não é necessário configuração manual.

#### 🎨 Ajuste Fino para Mangá (Anime vs Realismo)

O **Plus Face** foi treinado principalmente em **fotos de rostos humanos reais**. Quando aplicado em mangá (desenho 2D), pode criar o efeito **"Uncanny Valley"**:

> Rosto 3D realista em corpo 2D de anime = estranho e inconsistente

**Recomendações de Scale para Mangá:**

| Scale | Efeito | Recomendação |
|-------|--------|--------------|
| **0.5 - 0.7** | ✅ **Ponto Ideal** | Captura identidade sem estragar traço 2D |
| **0.8 - 1.0** | ⚠️ Risco | Rosto começa a parecer foto colada |
| **> 1.0** | ❌ Evitar | Efeito 3D forte, inconsistente com mangá |

**Nossa Estratégia Early-Heavy:**
- **Step 0:** Scale 1.0 (apenas neste momento crítico de semântica)
- **Step 1:** Scale 0.6 (fade)
- **Steps 2-3:** Scale 0.0 (desligado)

Isso garante que o personagem seja reconhecido sem que o estilo 2D seja corrompido.

#### 💾 Gestão de VRAM - ViT-H é Grande!

O encoder ViT-H consome aproximadamente **+600MB de VRAM** adicional:

```
SDXL-Lightning Base:        ~6.0 GB
+ ControlNet Canny:         +2.0 GB
+ IP-Adapter Plus ViT-H:    +0.6 GB
+ Buffers e máscaras:       +0.3 GB
─────────────────────────────────────
Total Estimado:             ~8.9 GB
Margem Segurança (12GB):    ~3.1 GB ✅
```

> ⚠️ **OBRIGATÓRIO:** Usar `enable_model_cpu_offload()` na RTX 3060 12GB!

---

### 2.3 Novo Módulo: `core/generation/regional_ip_adapter.py`

### 2.2 Novo Módulo: `core/generation/regional_ip_adapter.py`

```python
"""
Regional IP-Adapter com Early-Heavy Injection para SDXL-Lightning 4-Step.

Baseado em:
- T-GATE (ICML 2024): Early stopping de cross-attention em few-steps models
- ICAS (2025): Multi-embedding cyclic injection superior a simultaneous
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from PIL import Image
from dataclasses import dataclass
from diffusers.image_processor import IPAdapterMaskProcessor


@dataclass
class RegionalCharacter:
    """
    Estrutura de dados para personagem regional.
    
    Args:
        char_id: Identificador único do personagem
        embedding: Tensor CLIP do personagem (do HybridIdentitySystem)
        mask: Array numpy (H, W) com valores 0.0-1.0
        crop_image: PIL Image do crop do personagem (para IP-Adapter)
    """
    char_id: str
    embedding: torch.Tensor
    mask: np.ndarray
    crop_image: Image.Image


class EarlyHeavyRegionalIP:
    """
    Controlador de IP-Adapter com otimização temporal para 4 steps.
    
    Implementa a estratégia Early-Heavy baseada em T-GATE:
    - Steps 0-1: Injeção máxima (semantics planning)
    - Steps 2-3: Desligado (fidelity improvement sem interferência)
    
    Para múltiplos personagens, usa injeção cíclica (ICAS):
    - Step 0: Personagem A com força máxima
    - Step 1: Personagem B com força máxima (ou fade se único)
    - Steps 2-3: Desligado para todos
    """
    
    def __init__(
        self,
        pipeline,  # StableDiffusionXLControlNetPipeline
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        self.pipeline = pipeline
        self.device = device
        self.dtype = dtype
        self.mask_processor = IPAdapterMaskProcessor()
        
        # Carregar IP-Adapter Plus Face ViT-H (maior impacto por step)
        # ⚠️ IMPORTANTE: Este modelo usa encoder ViT-H, não o padrão do SDXL
        # A API moderna do Diffusers baixa o encoder correto automaticamente
        self.pipeline.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors",
            torch_dtype=dtype
        )
        
        # 🎨 AJUSTE FINO PARA MANGÁ:
        # O plus-face tende a realismo. Em mangá, pode criar "Uncanny Valley"
        # (rosto 3D em corpo 2D). Mantenha scale baixo!
        # 
        # Recomendações para mangá:
        # - 0.5 a 0.7: Ponto ideal (captura identidade sem estragar traço 2D)
        # - > 0.8: Risco de rosto parecer foto colada (evitar!)
        # - Early-Heavy usa 1.0 apenas no Step 0 (semantics), depois reduz
        
        # Otimizações de VRAM obrigatórias para RTX 3060 12GB
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.enable_vae_slicing()
    
    def generate_regional(
        self,
        prompt: str,
        negative_prompt: str = "",
        characters: Optional[List[RegionalCharacter]] = None,
        controlnet_image: Optional[Image.Image] = None,
        num_inference_steps: int = 4,  # FIXO - SDXL-Lightning
        guidance_scale: float = 1.2,
        height: int = 1024,
        width: int = 1408,
    ) -> Image.Image:
        """
        Gera imagem com IP-Adapter regional otimizado para 4 steps.
        
        Args:
            prompt: Prompt de texto
            negative_prompt: Prompt negativo
            characters: Lista de RegionalCharacter (máximo 2 para RTX 3060 12GB)
            controlnet_image: Imagem Canny para ControlNet
            num_inference_steps: Deve ser 4 (SDXL-Lightning)
            guidance_scale: Scale do CFG (1.2 para Lightning)
            height: Altura da imagem
            width: Largura da imagem
            
        Returns:
            Imagem PIL gerada
            
        Raises:
            ValueError: Se mais de 2 personagens (limitação de VRAM)
        """
        if not characters:
            return self._generate_base(prompt, negative_prompt, controlnet_image)
        
        if len(characters) > 2:
            raise ValueError(
                "RTX 3060 12GB suporta máximo 2 personagens simultâneos "
                "sem OOM. Use batching sequencial para 3+ personagens."
            )
        
        # 1. Preparar imagens de referência (crops dos personagens)
        reference_images = [char.crop_image for char in characters]
        
        # 2. Preparar máscaras regionais
        masks = [char.mask for char in characters]
        processed_masks = self.mask_processor.preprocess(
            masks, height=height, width=width
        )
        
        # Reshape para formato esperado: (batch_size, num_images, H, W)
        ip_adapter_masks = processed_masks.reshape(
            1, len(characters), processed_masks.shape[-2], processed_masks.shape[-1]
        )
        
        # 3. Configurar escala inicial (será modificada pelo callback)
        # Formato: [[scale_char1, scale_char2, ...]]
        num_chars = len(characters)
        
        # 4. Callback de Early-Heavy Injection (Fundamentado em T-GATE)
        def early_heavy_callback(pipe, step_index, timestep, callback_kwargs):
            """
            Estratégia de injeção temporal otimizada.
            
            Para 2 personagens (cíclica):
            - Step 0: [1.0, 0.0] - Personagem A "carimba" identidade
            - Step 1: [0.0, 1.0] - Personagem B "carimba" identidade
            - Steps 2-3: [0.0, 0.0] - Desligado, ControlNet domina
            
            Para 1 personagem:
            - Step 0: [1.0] - Máxima força
            - Step 1: [0.6] - Fade
            - Steps 2-3: [0.0] - Desligado
            """
            if step_index == 0:
                # Semantics planning: máxima força no primeiro personagem
                scales = [[1.0 if i == 0 else 0.0 for i in range(num_chars)]]
            elif step_index == 1:
                if num_chars > 1:
                    # Alternância cíclica para segundo personagem
                    scales = [[0.0 if i == 0 else 1.0 for i in range(num_chars)]]
                else:
                    # Fade para personagem único
                    scales = [[0.6]]
            else:
                # Fidelity improving: desliga IP-Adapter
                # Permite que ControlNet refine estrutura sem conflito
                scales = [[0.0 for _ in range(num_chars)]]
            
            pipe.set_ip_adapter_scale(scales)
            return callback_kwargs
        
        # 5. Executar geração
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=controlnet_image,
            ip_adapter_image=reference_images,
            cross_attention_kwargs={"ip_adapter_masks": ip_adapter_masks},
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            callback_on_step_end=early_heavy_callback,
        ).images[0]
        
        return result
    
    def _generate_base(
        self,
        prompt: str,
        negative_prompt: str,
        controlnet_image: Optional[Image.Image]
    ) -> Image.Image:
        """Fallback sem IP-Adapter quando não há personagens conhecidos."""
        return self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=controlnet_image,
            num_inference_steps=4,
            guidance_scale=1.2,
        ).images[0]
```

---

## Algoritmo de Early-Heavy Injection

### 3.1 Por que Funciona em 4 Steps?

Baseado em T-GATE, o processo de difusão tem duas fases distintas:

```
Step 0 (0-25%): SEMANTICS-PLANNING
├── O latent é puro ruído gaussiano
├── Cross-attention define a "direção" semântica
└── IP-Adapter deve ter máxima influência (scale 1.0)

Step 1 (25-50%): TRANSITION  
├── Latent organiza estrutura básica
├── IP-Adapter mantém identidade mas cede para ControlNet
└── Scale reduzido (0.6) ou alternância cíclica

Steps 2-3 (50-100%): FIDELITY-IMPROVING
├── Estrutura já está definida (anatomia, pose)
├── Cross-attention adicional introduz ruído/artefatos
├── ControlNet deve dominar (preservar bordas Canny)
└── IP-Adapter deve ser 0.0 para não competir
```

### 3.2 Estratégia Cíclica para 2 Personagens

**Problema com abordagem simultânea:**
```python
# ❌ RUIM: Ambos simultâneos causam vazamento
scales = [0.5, 0.5]  # Constante - oversimplification
```

**Solução com injeção cíclica:**
```python
# ✅ BOM: Alternância exclusiva preserva identidade
Step 0: [1.0, 0.0]  # A "carimba" identidade no latent
Step 1: [0.0, 1.0]  # B "carimba" identidade no latent  
Step 2: [0.0, 0.0]  # Refinamento estrutural puro
Step 3: [0.0, 0.0]  # Finalização sem interferência
```

**Resultado:** Cada personagem recebe atenção exclusiva durante o momento crítico, eliminando competição por cross-attention.

---

## Integração com Pipeline Existente

### 4.1 Modificações em `core/generation/pipeline.py`

```python
def _generate_single_tile(
    self,
    image: Image.Image,
    canny_edges: np.ndarray,
    character_embeddings: Dict[str, torch.Tensor],
    detections: List[Dict],
    options: Any,
    original_image: Optional[Image.Image] = None,
    target_size: Optional[Tuple[int, int]] = None,
    character_masks: Optional[Dict[str, np.ndarray]] = None,  # NOVO
    character_crops: Optional[Dict[str, Image.Image]] = None,  # NOVO
) -> Image.Image:
    """
    Gera tile único com suporte a Regional IP-Adapter.
    """
    # ... código existente ...
    
    # Verificar se Regional IP-Adapter está disponível e necessário
    has_regional_data = (
        character_masks is not None and 
        character_crops is not None and
        len(character_embeddings) > 0
    )
    
    if has_regional_data and REGIONAL_IP_AVAILABLE:
        # Preparar RegionalCharacters
        characters = []
        for idx, (char_id, embedding) in enumerate(character_embeddings.items()):
            if idx >= 2:  # Limitação VRAM
                break
            
            mask = character_masks.get(char_id)
            crop = character_crops.get(char_id)
            
            if mask is not None and crop is not None:
                characters.append(RegionalCharacter(
                    char_id=char_id,
                    embedding=embedding,
                    mask=mask,
                    crop_image=crop
                ))
        
        # Inicializar controller (lazy singleton)
        if not hasattr(self, '_regional_ip_controller'):
            from .regional_ip_adapter import EarlyHeavyRegionalIP
            self._regional_ip_controller = EarlyHeavyRegionalIP(
                pipeline=self.pipeline,
                device=self.device,
                dtype=self.dtype
            )
        
        # Gerar com Regional IP-Adapter
        result = self._regional_ip_controller.generate_regional(
            prompt=prompt,
            negative_prompt=negative_prompt,
            characters=characters,
            controlnet_image=canny_pil,
            height=image.height,
            width=image.width
        )
    else:
        # Fallback para geração base (sem IP-Adapter ou versão antiga)
        result = self._generate_with_standard_pipeline(...)
    
    return result
```

### 4.2 Modificações em `core/chapter_processing/pass2_generator.py`

```python
def _generate_single_tile_page(self, ...):
    # Carregar dados existentes...
    active_embeddings = {...}
    masks = {...}
    
    # NOVO: Carregar crops dos personagens para IP-Adapter
    character_crops = {}
    for char_id in tile_job.active_char_ids:
        # Carregar crop do personagem (salvo no Pass 1)
        crop_path = self.db.get_character_crop_path(char_id)
        if crop_path:
            character_crops[char_id] = Image.open(crop_path).convert('RGB')
    
    # Adicionar às opções
    options_with_masks['character_crops'] = character_crops
    
    result = self.generator.generate_image(...)
```

---

## Gestão de VRAM

### 5.1 Profile de Memória

```
Base (SDXL-Lightning + ControlNet):         ~8.2 GB
+ IP-Adapter Plus Face (1 personagem):      +0.4 GB
+ IP-Adapter Plus Face (2 personagens):     +0.8 GB
+ Máscaras e buffers temporários:           +0.3 GB
──────────────────────────────────────────────────
TOTAL ESTIMADO:                             ~9.3 GB
Margem de segurança (12GB - 9.3GB):         ~2.7 GB ✅
```

### 5.2 Otimizações Obrigatórias

```python
# No __init__ do EarlyHeavyRegionalIP
self.pipeline.enable_model_cpu_offload()  # Essencial para 2 personagens
self.pipeline.enable_vae_slicing()        # Para tiles grandes

# Durante geração de múltiplas páginas
torch.cuda.empty_cache()  # Entre páginas
```

### 5.3 Fallback de Memória

```python
try:
    result = self.generate_regional(characters=[char1, char2], ...)
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    # Fallback: Gerar um por vez
    result = self._generate_sequential(characters)
```

---

## Checklist de Implementação

### Fase 1: Setup (Dia 1)
- [ ] Atualizar `requirements.txt` para `diffusers>=0.29.0`
- [ ] Criar `core/generation/regional_ip_adapter.py`
- [ ] Baixar modelo `ip-adapter-plus-face_sdxl_vit-h.safetensors`

### Fase 2: Integração (Dia 2)
- [ ] Modificar `core/generation/pipeline.py`
- [ ] Adaptar `Pass2Generator` para fornecer crops dos personagens
- [ ] Garantir que máscaras sejam salvas em formato (H, W) float32

### Fase 3: Testes (Dia 3)
- [ ] **Teste A/B:** Early-Heavy vs Scale constante (medir qualidade)
- [ ] **Teste de Isolamento:** Dois personagens distintos - verificar vazamento
- [ ] **Teste de VRAM:** Monitorar `nvidia-smi` com 2 personagens em 1024x1024
- [ ] **Teste de Consistência:** Mesmo personagem em 3 páginas - Delta E < 5.0

### Fase 4: Otimização (Dia 4)
- [ ] Cache de embeddings em memória
- [ ] Ajustar sigma do gaussian blur nas máscaras (testar 5.0, 10.0, 20.0)
- [ ] Fine-tuning do scale inicial (0.9 vs 1.0 vs 1.1)

---

## Referências

### Papers Científicos

1. **T-GATE (ICML 2024)**: "Cross-Attention Makes Inference Cumbersome in Text-to-Image Diffusion Models"
   - Zhang et al.
   - Early stopping de cross-attention em few-steps models

2. **ICAS (2025)**: "IP-Adapter and ControlNet-based Attention Structure"
   - Yang et al.
   - Multi-embedding cyclic injection vs simultaneous

### Documentação Técnica

3. **HuggingFace Diffusers ≥0.29.0**: 
   - `IPAdapterMaskProcessor`
   - `cross_attention_kwargs` com `ip_adapter_masks`
   - Documentação: https://huggingface.co/docs/diffusers

### Repositórios

4. **IP-Adapter**: https://github.com/tencent-ailab/IP-Adapter
5. **MangaAutoColor Pro**: `docs/ARCHITECTURE.md`

---

## Notas Finais

> **"Não reinvente a roda implementando camadas de atenção customizadas. A API nativa do Diffusers já possui o mecanismo de regional masking. Foque na estratégia temporal de injeção (Early-Heavy), que é onde reside o ganho de qualidade em 4 steps."**

### Próximos Passos

1. Implementar módulo `EarlyHeavyRegionalIP`
2. Integrar com pipeline existente mantendo backward compatibility
3. Testar rigorosamente com diferentes cenários de personagens
4. Documentar resultados e métricas de qualidade

---

<p align="center">
  <strong>Documento Técnico Revisado v2.0</strong><br>
  MangaAutoColor Pro - Implementação Regional IP-Adapter com Early-Heavy Injection<br>
  2026
</p>

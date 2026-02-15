# Guia de Debugging: Cores Psicodélicas/Fritadas

Este guia explica como diagnosticar e corrigir problemas de geração onde as imagens saem com cores distorcidas, saturadas ou "psicodélicas".

---

## 🎯 Sintomas do Problema

| Sintoma | Descrição | Causa Provável |
|---------|-----------|----------------|
| Cores neon excessivas | Saturação muito alta, cores irreais | `rescale_betas_zero_snr=True` |
| Pixels aleatórios | Pontos de cores aleatórias espalhados | NaNs nos latents |
| Imagem "fritada" | Cores estouradas, perda de detalhes | VAE sem `force_upcast` |
| Padrões repetitivos | Artefatos de grid ou padrões estranhos | Scheduler instável |

---

## 🧪 Testes de Validação

### 1. Testes Rápidos (Sem GPU)

```bash
# Testa configurações críticas
pytest tests/unit/test_psychedelic_fixes.py -v
```

Estes testes verificam:
- ✅ `rescale_betas_zero_snr=False` no scheduler
- ✅ `force_upcast=True` no VAE
- ✅ VAEDtypeAdapter corrige NaNs/Infs
- ✅ Prompt negativo contém termos de proteção

### 2. Testes de Integração (Requer Modelos)

```bash
# Testes que podem usar GPU
pytest tests/integration/test_generation_quality.py -v
```

---

## 🔧 Ferramentas de Análise

### Análise de Imagem Gerada

```python
from tests.integration.test_generation_quality import TestImageAnalysisUtils
from PIL import Image

# Carrega imagem gerada
img = Image.open("output/page_001_colored.png")

# Analisa
analyzer = TestImageAnalysisUtils()
metrics = analyzer.detect_psychedelic_artifacts(img)

print(f"Saturação média: {metrics['saturation_mean']:.2f}")
print(f"Pixels extremos: {metrics['extreme_pixels_ratio']:.2%}")
print(f"É psicodélico: {metrics['is_psychedelic']}")
```

### Thresholds de Alerta

| Métrica | Valor Seguro | Valor Crítico |
|---------|--------------|---------------|
| `saturation_mean` | < 0.5 | > 0.8 |
| `extreme_pixels_ratio` | < 10% | > 30% |
| `color_variance` | < 5000 | > 10000 |

---

## 🚨 Checklist de Correção

### Verificação 1: Scheduler

```python
from diffusers import DDIMScheduler

scheduler = DDIMScheduler.from_config(
    "runwayml/stable-diffusion-v1-5",
    subfolder="scheduler",
    rescale_betas_zero_snr=False,  # DEVE SER FALSE!
    clip_sample=False
)
```

**❌ Problema:** `rescale_betas_zero_snr=True`
**✅ Correção:** Definir como `False`

### Verificação 2: VAE

```python
from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    torch_dtype=torch.float32
)
vae.config.force_upcast = True  # DEVE SER TRUE!
```

**❌ Problema:** VAE sem `force_upcast`
**✅ Correção:** Habilitar `force_upcast`

### Verificação 3: VAEDtypeAdapter

```python
from core.generation.engines.vae_dtype_adapter import VAEDtypeAdapter

with VAEDtypeAdapter(pipe.vae):
    result = pipe(...)
```

**❌ Problema:** Latents com NaNs não sendo detectados
**✅ Correção:** Adapter faz `nan_to_num` e `clamp`

### Verificação 4: Prompt Negativo

```python
negative_prompt = (
    "monochrome, greyscale, lowres, bad anatomy, worst quality, "
    "oversaturated, neon colors, psychedelic, distorted colors, "  # IMPORTANTE!
    "blurry, watermark, signature, text, cropped"
)
```

**❌ Problema:** Prompt negativo sem termos de proteção
**✅ Correção:** Adicionar "oversaturated, neon colors, psychedelic"

---

## 📊 Debug Avançado

### Verificar NaNs nos Latents

```python
import torch

# Durante geração, intercepte os latents
latents = pipe(..., output_type="latent").images

if torch.isnan(latents).any():
    print(f"⚠️  NaNs detectados: {torch.isnan(latents).sum()} pixels")
    print(f"   Localização: {torch.where(torch.isnan(latents))}")
```

### Verificar Range dos Latents

```python
print(f"Latents min: {latents.min():.2f}")
print(f"Latents max: {latents.max():.2f}")
print(f"Latents mean: {latents.mean():.2f}")

# Valores seguros: min > -5, max < 5
# Valores perigosos: min < -10 ou max > 10
```

---

## 🔄 Solução de Contingência

Se o problema persistir:

### Opção 1: Usar FP32 Completo

```python
# config/settings.py
DTYPE = torch.float32  # Mais lento mas mais estável
```

### Opção 2: Reduzir Strength

```python
# config/settings.py
V3_STRENGTH = 0.6  # Padrão é 0.75
```

### Opção 3: Aumentar Steps

```python
# config/settings.py
V3_STEPS = 30  # Padrão é 20
```

### Opção 4: Desabilitar IP-Adapter

```python
# Gere sem referência visual
options['reference_image'] = None
options['ip_adapter_scale'] = 0.0
```

---

## 📝 Histórico de Correções

| Data | Problema | Solução |
|------|----------|---------|
| 2026-02-14 | `rescale_betas_zero_snr=True` | Removido, causava instabilidade |
| 2026-02-14 | VAE sem `force_upcast` | Habilitado para evitar NaNs |
| 2026-02-14 | NaNs não detectados | VAEDtypeAdapter melhorado |
| 2026-02-14 | Prompt negativo fraco | Adicionados termos de proteção |

---

## 🔗 Referências

- [Diffusers Documentation - Scheduler](https://huggingface.co/docs/diffusers/main/en/api/schedulers)
- [Stable Diffusion Artifacts Guide](https://stable-diffusion-art.com/fix-artifacts/)
- [VAE Numerical Stability](https://huggingface.co/docs/diffusers/main/en/api/models/autoencoderkl)

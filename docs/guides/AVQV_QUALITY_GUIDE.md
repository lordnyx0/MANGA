# 🔍 Guia: Automated Visual Quality Validation (AVQV)

Este guia explica como utilizar o framework **AVQV** para garantir que a qualidade visual da colorização não sofra regressões após mudanças no código.

## Motivação
Diferente de testes unitários tradicionais, a colorização de IA pode falhar de formas sutis (ex: cores "fritadas", barras vermelhas nas bordas, balões sujos). O AVQV usa o motor real para validar métricas estatísticas da imagem gerada.

## Como Rodar
Os testes de AVQV requerem **GPU (CUDA)** e os modelos carregados.

```bash
# Ativar venv
venv\Scripts\activate

# Rodar os testes de qualidade visual
pytest tests/integration/test_visual_quality_regression.py -v -s
```

## Métricas Monitoradas

### 1. Bubble Purity (Pureza de Balões)
- **O que faz**: Analisa a variância de cor (RGB) dentro dos BBoxes de texto detectados.
- **Por que importa**: Garante que o *Bubble Masking* está funcionando. Balões brancos devem ter variância zero. 
- **Threshold**: < `0.01` (Variância média).

### 2. Edge Neutrality (Neutralidade de Bordas)
- **O que faz**: Compara a dominância do canal vermelho (Chrominance) nas bordas vs centro da imagem.
- **Por que importa**: Detecta artefatos causados por *VAE Tiling*. Se as bordas estiverem muito diferentes do centro, o teste falha.
- **Threshold**: Red-Delta < `5.0`.

### 3. Tensor Stability (NaN Check)
- **O que faz**: Verifica a presença de pixels `NaN` ou `Inf`.
- **Por que importa**: Detecta problemas de precisão numérica (FP16/FP32 mismatch) que causam imagens pretas ou artefatos coloridos.

## Adicionando Novos Testes
Para adicionar um novo critério de qualidade visual, edite o arquivo `tests/integration/test_visual_quality_regression.py` e adicione um novo método de análise estatística usando `numpy`.

> [!TIP]
> Use o AVQV sempre que modificar o `SD15LineartEngine` ou o `compose_final`.

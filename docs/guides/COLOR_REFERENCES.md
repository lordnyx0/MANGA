# 🎨 Color Reference System

## Visão Geral

O sistema de Referências de Cor permite que você forneça imagens oficiais coloridas dos personagens para garantir que as cores geradas sejam precisas e fieis ao original.

## Como Funciona

```
Usuário faz upload de imagens de referência coloridas
            ↓
Pass 1: Detecção YOLO encontra personagens nas referências
            ↓
Extração de paleta de cores real da imagem de referência
            ↓
Paleta marcada como "is_color_reference = True"
            ↓
Pass 2: Geração usa cores reais em vez de STYLE_PRESETS
            ↓
Personagens coloridos com precisão!
```

## Uso

### Via Extensão do Navegador

1. Clique na extensão MangaAutoColor Pro
2. Faça download do capítulo (basta fornecer a URL)
3. Na seção "Referências de Cor", clique em "Escolher Arquivos"
4. Selecione uma ou mais imagens coloridas dos personagens
5. Clique em "Analisar Capítulo"
6. Aguarde a análise completar
7. Clique em "Gerar Todas"

### Via API REST

```bash
# Upload com referências de cor
curl -X POST http://localhost:8000/chapters/analyze \
  -F "files=@page_001.png" \
  -F "files=@page_002.png" \
  -F "color_references=@char1_official.jpg" \
  -F "color_references=@char2_official.png"
```

### Via Python

```python
from core.chapter_processing.pass1_analyzer import Pass1Analyzer

analyzer = Pass1Analyzer(
    yolo_detector=yolo,
    palette_extractor=palette_extractor,
    identity_encoder=encoder
)

# Analisar com referências de cor
color_refs = [
    "./references/character_1.jpg",
    "./references/character_2.png"
]

summary = analyzer.analyze_chapter(
    image_files=chapter_pages,
    output_db=db,
    color_reference_paths=color_refs  # ← Referências de cor
)
```

## Requisitos das Imagens de Referência

### Qualidade

| Aspecto | Recomendação |
|---------|--------------|
| Resolução | Mínimo 512x512px |
| Formato | PNG, JPG, WEBP |
| Tamanho | Máximo 10MB por imagem |
| Quantidade | Até 5 imagens por análise |

### Conteúdo

✅ **Bom:**
- Personagem bem iluminado
- Fundo simples ou transparente
- Cores claras e definidas
- Pose que mostre o rosto e roupa

❌ **Evitar:**
- Imagens muito escuras
- Múltiplos personagens sobrepostos
- Cores alteradas por filtros
- Imagens em preto e branco

## Extração de Paleta

### Processo

O sistema extrai automaticamente:

```python
# Exemplo de paleta extraída
palette = {
    'colors': [
        {'name': 'hair', 'lab': [45.2, 12.3, -45.7], 'percentage': 0.35},
        {'name': 'skin', 'lab': [72.1, 18.5, 32.2], 'percentage': 0.25},
        {'name': 'eyes', 'lab': [35.0, 45.2, -25.3], 'percentage': 0.08},
        {'name': 'clothes_primary', 'lab': [55.3, -15.2, 42.1], 'percentage': 0.20},
        {'name': 'clothes_secondary', 'lab': [75.0, 5.2, -10.5], 'percentage': 0.12}
    ],
    'is_color_reference': True,
    'source_page': -1,  # Indica referência externa
    'extracted_at': '2025-02-07T01:15:23'
}
```

### Categorias de Cor

As cores são automaticamente categorizadas:

| Categoria | Descrição | Uso no Prompt |
|-----------|-----------|---------------|
| `hair` | Cabelo | "{color} hair" |
| `skin` | Pele | "{color} skin" |
| `eyes` | Olhos | "{color} eyes" |
| `clothes_primary` | Roupa principal | "{color} outfit" |
| `clothes_secondary` | Roupa secundária | "{color} accents" |

## Comportamento na Geração

### Com Referências de Cor

```python
# Quando há referências, STYLE_PRESETS são ignorados
options = {
    'character_palettes': {
        'ref_char_000_000': palette,  # ← Referência de cor
        'char_001': palette           # ← Paleta normal
    }
}

# O sistema detecta has_color_reference = True
# E constrói o prompt com as cores reais
prompt = "colorful manga illustration, vibrant colors, "
prompt += "blue hair, peach skin, green eyes, red outfit"
```

### Sem Referências de Cor

```python
# Quando não há referências, STYLE_PRESETS são aplicados
options = {
    'style_preset': 'vibrant',  # ← Usa preset
    'character_palettes': {}     # ← Sem referências
}

# O prompt inclui o addition do preset
prompt = "colorful manga illustration, vibrant colors, "
prompt += "vibrant saturated colors, rich tones"  # ← De STYLE_PRESETS
```

## STYLE_PRESETS

Quando não há referências de cor, você pode escolher entre 7 presets:

| Preset | Prompt Addition | Uso |
|--------|-----------------|-----|
| `default` | (nenhum) | Equilibrado |
| `vibrant` | `vibrant saturated colors, rich tones` | Cores intensas |
| `muted` | `muted desaturated colors, soft tones` | Tons suaves |
| `sepia` | `sepia tone, warm vintage colors` | Vintage |
| `flashback` | `mostly black and white with selective color` | Flashback |
| `dream` | `dreamy pastel colors, ethereal atmosphere` | Sonho |
| `nightmare` | `dark desaturated colors, deep shadows` | Pesadelo |

## Debugging

### Verificar Paletas Extraídas

```bash
# Os logs são salvos em output/{chapter_id}/logs/
cat output/ch_d8d4c0757039/logs/generation_log.json | jq '.steps[0].prompt'
```

### Visualizar Paleta

```python
import json
from PIL import Image
import matplotlib.pyplot as plt

# Carregar paleta
with open('output/ch_xxx/embeddings/ref_char_000_000_palette.json') as f:
    palette = json.load(f)

# Visualizar
colors = [c['lab'] for c in palette['colors']]
names = [c['name'] for c in palette['colors']]

# Converter LAB para RGB para visualização
from skimage.color import lab2rgb
rgb_colors = [lab2rgb([[c]])[0][0] for c in colors]

plt.figure(figsize=(8, 2))
for i, (color, name) in enumerate(zip(rgb_colors, names)):
    plt.subplot(1, len(colors), i+1)
    plt.imshow([[color]])
    plt.title(name)
    plt.axis('off')
plt.show()
```

## Troubleshooting

### Cores incorretas

Se as cores geradas não correspondem às referências:

```python
# Verificar se as referências foram processadas
db = ChapterDatabase(chapter_id)
ref_palettes = db.load_reference_palettes()

if not ref_palettes:
    print("Nenhuma referência de cor encontrada!")
    print("Certifique-se de enviar as imagens na análise.")

# Verificar conteúdo
for char_id, palette in ref_palettes.items():
    print(f"{char_id}: {len(palette.colors)} cores")
    for c in palette.colors:
        print(f"  - {c['name']}: {c['lab']}")
```

### Referência não detectada

Se o YOLO não detectou o personagem na referência:

1. **Use imagens mais claras** - Iluminação adequada
2. **Corte mais próximo** - Personagem ocupando >50% da imagem
3. **Evite fundos complexos** - Fundo sólido ou simples
4. **Verifique a resolução** - Mínimo 512x512px

### Conflito com STYLE_PRESET

As referências de cor sempre têm prioridade sobre STYLE_PRESETS:

```python
# Isso é automático - não precisa configurar
# Se has_color_reference == True, STYLE_PRESET é ignorado
```

## Exemplos

### Exemplo 1: Personagem Único

```bash
# Upload de referência
curl -X POST http://localhost:8000/chapters/analyze \
  -F "files=@manga_pages.zip" \
  -F "color_references=@goku_official.png"

# Resultado: Goku será colorido com cores oficiais (cabelo laranja, roupa laranja/azul)
```

### Exemplo 2: Múltiplos Personagens

```bash
# Upload de múltiplas referências
curl -X POST http://localhost:8000/chapters/analyze \
  -F "files=@chapter.zip" \
  -F "color_references=@naruto_official.jpg" \
  -F "color_references=@sasuke_official.jpg" \
  -F "color_references=@sakura_official.jpg"

# Resultado: Cada personagem terá suas cores específicas
```

### Exemplo 3: Comparação

Sem referência:
```
Prompt: "colorful manga illustration, vibrant colors"
Resultado: Cores arbitrárias, possivelmente laranja cabelo
```

Com referência (Sailor Moon):
```
Prompt: "colorful manga illustration, vibrant colors, 
         blonde hair, blue eyes, white and blue outfit"
Resultado: Cores oficiais da Sailor Moon
```

## Limitações

1. **Qualidade da Referência**: A qualidade da extração depende da qualidade da imagem de referência
2. **Iluminação**: Diferenças de iluminação podem afetar a extração
3. **Complexidade**: Personagens com muitos detalhes podem ter extração imprecisa
4. **Matching**: O sistema associa referências a detecções baseado em similaridade visual

## Melhores Práticas

1. **Use imagens oficiais** - Artwork oficial tem cores mais precisas
2. **Múltiplos ângulos** - Forneça diferentes poses para melhor matching
3. **Verifique a extração** - Consulte os logs para confirmar as cores extraídas
4. **Ajuste se necessário** - Se necessário, edite os arquivos JSON das paletas manualmente

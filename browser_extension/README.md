# 🎨 MangaAutoColor Pro - Browser Extension

Extensão para Microsoft Edge/Chrome que coloriza páginas de mangá em tempo real usando o backend MangaAutoColor Pro.

## 🏗️ Arquitetura Client-Side Capture

Esta extensão usa uma arquitetura especial para contornar proteções como Cloudflare:

```
┌─────────────────────────────────────────────────────────────┐
│                    NAVEGADOR (Edge/Chrome)                   │
│  ┌─────────────────┐      ┌──────────────────────────────┐  │
│  │  Site de Mangá  │      │     Content Script           │  │
│  │  (Manganato)    │─────▶│  1. Captura imagem do DOM    │  │
│  │                 │      │     - fetch() com cookies    │  │
│  │  ┌───────────┐  │      │     - canvas (fallback)      │  │
│  │  │ Imagem    │  │      │  2. Converte para Blob       │  │
│  │  │ Manga     │  │      │  3. Envia bytes para API     │  │
│  │  └───────────┘  │      │     (NUNCA envia URL!)       │  │
│  └─────────────────┘      └──────────────────────────────┘  │
│                                        │                    │
│                                        ▼                    │
│                              ┌──────────────────┐          │
│                              │  localhost:8000  │          │
│                              │  /realtime/      │          │
│                              │  colorize        │          │
│                              └──────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

**Princípio fundamental**: A extensão NUNCA envia a URL da imagem para o backend. Em vez disso, extrai os bytes da imagem diretamente do DOM e envia como `multipart/form-data`.

## 📁 Estrutura

```
browser_extension/
├── manifest.json       # Configuração da extensão (Manifest V3)
├── background.js       # Service worker (menu de contexto)
├── content_script.js   # Script injetado nas páginas (captura)
├── popup.html          # Interface do popup
├── popup.js            # Lógica do popup
├── styles.css          # Estilos
└── icons/              # Ícones
```

## 🚀 Como Usar

### 1. Instalar a Extensão

1. Abra Edge/Chrome e vá para `edge://extensions/` (ou `chrome://extensions/`)
2. Ative "Modo de desenvolvedor"
3. Clique em "Carregar sem pacote"
4. Selecione a pasta `browser_extension/`

### 2. Iniciar o Backend

```bash
# Na pasta raiz do projeto MangaAutoColor
start_server.bat
```

### 3. Colorizar Mangá

1. Navegue até um site de mangá (ex: Manganato, MangaDex)
2. **Clique direito** em uma página de mangá
3. Selecione **"🎨 Colorize this Image"**
4. Aguarde o processamento (~30s em RTX 3060)

## 🔧 Como Funciona

### Captura da Imagem

O `content_script.js` usa dois métodos para capturar a imagem:

#### Método 1: Fetch com Credenciais
```javascript
const response = await fetch(imgElement.src, {
  credentials: 'include',  // Usa cookies do usuário
  headers: { 'Accept': 'image/webp,image/apng,image/*' }
});
const blob = await response.blob();
```

**Vantagem**: Usa a sessão autenticada do usuário, bypassando Cloudflare.

#### Método 2: Canvas (Fallback)
```javascript
const canvas = document.createElement('canvas');
canvas.width = img.naturalWidth;
canvas.height = img.naturalHeight;
const ctx = canvas.getContext('2d');
ctx.drawImage(imgElement, 0, 0);
canvas.toBlob((blob) => resolve(blob), 'image/png');
```

**Vantagem**: Funciona mesmo com CORS restritivo.

### Envio para API

```javascript
const formData = new FormData();
formData.append('file', imageBlob, 'manga_page.png');
formData.append('style_preset', 'default');

const response = await fetch('http://localhost:8000/realtime/colorize', {
  method: 'POST',
  body: formData
});
```

### Substituição da Imagem

```javascript
const coloredBlob = await response.blob();
const coloredUrl = URL.createObjectURL(coloredBlob);
imgElement.src = coloredUrl;
```

## 🛠️ Solução de Problemas

### "Backend Offline"
- Verifique se o servidor Python está rodando em `http://localhost:8000`
- Verifique o firewall

### "Failed to colorize"
- A imagem pode estar protegida por CORS
- Tente recarregar a página
- Verifique se é uma imagem de mangá válida (tamanho mínimo 400x600)

### Extensão não aparece no menu de contexto
- Recarregue a extensão em `edge://extensions/`
- Recarregue a página do mangá

## 📝 Notas Técnicas

- A extensão só funciona em páginas HTTP/HTTPS (não em arquivos locais)
- O backend deve estar acessível em `localhost:8000`
- Imagens são processadas uma a uma para não sobrecarregar a GPU
- O URL local (`blob:`) é revogado automaticamente quando a página é fechada

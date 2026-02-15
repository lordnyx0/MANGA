# 📦 Instalação da Extensão MangaAutoColor Pro

## Pré-requisitos

1. **Backend rodando**: A extensão requer o servidor Python em execução
2. **Navegador compatível**: Microsoft Edge ou Google Chrome
3. **Modo desenvolvedor**: Necessário para carregar extensão não empacotada

---

## Passo 1: Iniciar o Backend

### Usando o script batch:

```bash
# Na pasta raiz do projeto
cd C:\caminho\para\manga-autocolor-pro
start_server.bat
```

### Ou manualmente:

```bash
venv\Scripts\activate
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Verificar se está rodando:

Abra no navegador: `http://localhost:8000/health`

Deve retornar:
```json
{
  "status": "healthy",
  "device": "cuda",
  "vram_gb": 12.0
}
```

---

## Passo 2: Carregar a Extensão

### Microsoft Edge

1. Abra Edge e digite na barra de endereço: `edge://extensions/`
2. No canto inferior esquerdo, **ative "Modo de desenvolvedor"**
3. Clique no botão **"Carregar sem pacote"** que aparecerá
4. Selecione a pasta `browser_extension/` (não os arquivos individuais)
5. A extensão 🎨 aparecerá na lista

### Google Chrome

1. Abra Chrome e digite: `chrome://extensions/`
2. No canto superior direito, **ative "Modo do desenvolvedor"**
3. Clique em **"Carregar sem compactação"**
4. Selecione a pasta `browser_extension/`

---

## Passo 3: Fixar na Barra de Ferramentas

1. Clique no ícone de **quebra-cabeça** (Edge) ou **extensões** (Chrome)
2. Encontre "MangaAutoColor Pro"
3. Clique no **alfinete** para fixar na barra

---

## Passo 4: Testar

1. Abra um site de mangá (ex: `https://manganato.com`)
2. Navegue até um capítulo
3. **Clique direito** em uma página de mangá em preto e branco
4. Selecione **"🎨 Colorize this Image"**
5. Aguarde o processamento (~30s em RTX 3060)
6. A imagem será substituída pela versão colorizada!

---

## Solução de Problemas

### "Backend Offline"
- Verifique se o servidor Python está rodando
- Confira se a URL da API está correta no painel da extensão
- Verifique se não há firewall bloqueando porta 8000

### "Failed to colorize: Image not found"
- A extensão só detecta imagens grandes (mínimo 400x600 pixels)
- Tente esperar a imagem carregar completamente
- Alguns sites usam lazy loading; role a página para carregar a imagem

### "Failed to colorize: Canvas draw failed"
- A imagem pode estar protegida por CORS
- Tente usar o modo de captura alternativo (a extensão tenta automaticamente)

### Conteúdo Misto (HTTPS→HTTP)
A extensão converte automaticamente para Blob URL, contornando o bloqueio.

---

## Atualizar a Extensão

Após modificar o código:

1. Vá para `edge://extensions/` ou `chrome://extensions/`
2. Encontre "MangaAutoColor Pro"
3. Clique no ícone 🔄 (atualizar)
4. **Recarregue a página do mangá** (F5)

---

## Desinstalar

### Remover Extensão
1. Vá para `edge://extensions/` ou `chrome://extensions/`
2. Encontre "MangaAutoColor Pro"
3. Clique em "Remover"

### Parar Backend
- Pressione `Ctrl+C` no terminal onde rodou o servidor

# 🚀 Scripts de Inicialização - MangaAutoColor Pro

Arquivos `.bat` para iniciar o backend facilmente no Windows.

## 📁 Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `start_server.bat` | **Simples** - Inicia o servidor com reload automático |
| `start_server_debug.bat` | **Debug** - Modo debug (salva imagens em `output/debug/`) |
| `start_server_advanced.bat` | **Avançado** - Menu com múltiplas opções |
| `check_and_install_deps.bat` | **Dependências** - Verifica e instala pacotes faltantes |

---

## 🎯 Uso Rápido

### Antes de começar: Verifique as dependências

Se for a primeira vez ou se der erros de `ModuleNotFoundError`:

1. Execute `check_and_install_deps.bat`
2. Aguarde a instalação (5-10 minutos)
3. Depois execute o servidor

### Método 1: Clique Duplo
1. Navegue até a pasta do projeto no Explorer
2. Dê **duplo clique** em `start_server.bat`
3. O servidor iniciará automaticamente

### Método 2: Prompt de Comando
```cmd
cd C:\caminho\para\manga-autocolor-pro
start_server.bat
```

### Modo Debug (Para Análise)
```cmd
start_server_debug.bat
```
O modo debug salva automaticamente todas as imagens intermediárias em `output/debug/`:
- `01_input.png` - Imagem recebida
- `02_canny.png` - Bordas detectadas  
- `03_detections.png` - Visualização das detecções
- `crops/` - Crops de personagens (body/face)
- `04_result.png` - Resultado final

Útil para diagnosticar problemas de qualidade!

---

## 🔧 Script de Dependências

O `check_and_install_deps.bat` verifica se todas as bibliotecas estão instaladas:

### O que ele faz:
1. Ativa o ambiente virtual (ou cria um novo)
2. Verifica dependências críticas:
   - scipy, scikit-learn, numpy
   - torch, diffusers, transformers
   - fastapi, uvicorn, opencv-python
   - pillow
3. Mostra versões instaladas
4. Oferece para instalar/atualizar tudo

### Quando usar:
- **Primeira instalação** do projeto
- Após erros como `ModuleNotFoundError: No module named 'scipy'`
- Quando atualizar o `requirements.txt`

---

## 📋 Opções do Script Avançado

O `start_server_advanced.bat` oferece 5 modos:

### 1️⃣ Modo Normal (Desenvolvimento)
```
Recarrega automaticamente quando você edita o código.
Ideal para desenvolvimento e testes.
```
**Comando equivalente:**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2️⃣ Modo Produção
```
Mais rápido, sem reload automático.
Menos mensagens de log.
```
**Comando equivalente:**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --no-access-log
```

### 3️⃣ Modo Warm Start ⭐
```
Pré-carrega os modelos SDXL na VRAM.
Reduz tempo de primeira colorização de ~35s para ~30s.
Ideal quando vai colorizar várias imagens seguidas.
⚠️ Consome ~11GB de VRAM constantemente!
```

### 4️⃣ Modo Worker
```
Usa múltiplos processos Python.
Melhor para processar capítulos inteiros em paralelo.
```
**Comando equivalente:**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
```

### 5️⃣ Verificar Dependências
```
Mostra versões instaladas do Python, PyTorch, CUDA, etc.
Útil para troubleshooting.
```

---

## 🌐 Acessando a API

Após iniciar o servidor, acesse:

| URL | Descrição |
|-----|-----------|
| `http://localhost:8000` | Health check - verifica se está rodando |
| `http://localhost:8000/docs` | Documentação interativa (Swagger UI) |
| `http://localhost:8000/realtime/status` | Status do modo realtime |

---

## 🛑 Parar o Servidor

### Método 1: CTRL+C
1. Clique na janela do terminal
2. Pressione `CTRL+C`
3. Confirme com `Y` se perguntar

### Método 2: Fechar a Janela
- Apenas feche a janela do terminal
- O servidor será encerrado automaticamente

---

## 🔧 Solução de Problemas

### "Não encontrei api\main.py"
- Certifique-se de executar o `.bat` na **pasta raiz** do projeto
- Não execute de dentro de subpastas

### Erros de `ModuleNotFoundError` (scipy, etc.)
Execute o verificador de dependências:
```
check_and_install_deps.bat
```
Ou manualmente:
```cmd
venv\Scripts\pip install -r requirements.txt
```

### "Python não encontrado"
- Instale Python 3.10 ou 3.11: https://python.org
- Marque "Add to PATH" durante a instalação

### "Ambiente virtual não encontrado"
- O script `check_and_install_deps.bat` pode criar um automaticamente
- Ou crie manualmente:
  ```cmd
  python -m venv venv
  venv\Scripts\pip install -r requirements.txt
  ```

### "Porta 8000 em uso"
- Outro programa está usando a porta
- Feche o outro programa, ou
- Mude a porta no arquivo `.bat` (substitua `8000` por outro número)

### "CUDA out of memory" no Warm Start
- Sua GPU não tem VRAM suficiente (~11GB necessários)
- Use o **Modo Normal** (opção 1) em vez de Warm Start

---

## 📊 Comparação de Modos

| Modo | Tempo 1ª Requisição | Uso VRAM | Ideal Para |
|------|---------------------|----------|------------|
| Normal | ~35s | Libera após uso | Uso ocasional |
| Warm Start | ~30s | ~11GB constante | Leitura de mangá |
| Produção | ~35s | Libera após uso | API pública |
| Worker | ~35s | Libera após uso | Processamento batch |

---

## 🎨 Fluxo Típico de Uso

### Para ler mangá online com a extensão:

1. Execute `start_server_advanced.bat`
2. Escolha opção **3 (Warm Start)**
3. Aguarde carregar os modelos (~10s)
4. Abra o navegador e instale a extensão
5. Acesse MangaDex e comece a colorizar!
6. Ao terminar, feche a janela do servidor

### Para processar capítulos inteiros:

1. Execute `start_server_advanced.bat`
2. Escolha opção **1 (Normal)** ou **4 (Worker)**
3. Use o CLI ou interface Gradio
4. Processe o capítulo

---

## 💡 Dicas

- **Mantenha a janela visível** para ver logs de erro
- **Use Warm Start** se vai colorizar mais de 5 imagens
- **CTRL+C duas vezes** força o encerramento imediato
- A janela não fecha sozinha para você poder ver erros

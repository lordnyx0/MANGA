# Modo Debug - MangaAutoColor Pro

## Para Ativar o Modo Debug

O modo debug salva automaticamente todas as imagens intermediárias para análise:
- Imagem recebida da extensão
- Bordas Canny detectadas
- Detecções de personagens (com bounding boxes)
- Crops de bodies e faces
- Resultado final da colorização

## Como Ativar

Defina a variável de ambiente `MANGA_DEBUG=true` antes de iniciar o servidor.

### Windows (CMD)

```batch
set MANGA_DEBUG=true
scripts\windows\run.bat
```

### Windows (PowerShell)

```powershell
$env:MANGA_DEBUG="true"
.\scripts\windows\run.bat
```

### Linux/Mac

```bash
export MANGA_DEBUG=true
./scripts/run.sh
```

## Verificar se o Debug está Ativo

Acesse: http://localhost:8000/health

Deve retornar:
```json
{
  "status": "healthy",
  "debug_mode": true,
  ...
}
```

## Local das Imagens de Debug

As imagens são salvas em:
```
output/debug/YYYYMMDD_HHMMSS_MMM/
├── 01_input.png          # Imagem recebida
├── 02_canny.png          # Bordas Canny
├── 03_detections.png     # Detecções visualizadas
├── crops/
│   ├── body_00.png       # Crop do body 1
│   ├── body_01.png       # Crop do body 2
│   └── face_00.png       # Crop da face 1
└── 04_result.png         # Resultado final
```

## Desativar o Debug

Basta iniciar o servidor normalmente (sem definir a variável):

```batch
scripts\windows\run.bat
```

Ou, se a variável estiver definida na sessão:

```batch
set MANGA_DEBUG=false
scripts\windows\run.bat
```

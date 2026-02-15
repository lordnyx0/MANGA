# MangaAutoColor Pro - Test Suite

Suíte de testes automatizados para o MangaAutoColor Pro.

## 📁 Estrutura

```
tests/
├── high/           # Testes obrigatórios (CI por padrão)
├── medium/         # Testes úteis (CPU)
├── low/            # Testes pré-release
├── conftest.py     # Fixtures e configuração compartilhada
└── README.md       # Este arquivo
```

## 🚀 Execução

### Rodar todos os testes (CPU)
```bash
pytest -q
```

### Rodar pulando GPU
```bash
pytest -q -m "not gpu"
```

### Rodar apenas alta prioridade
```bash
pytest -q tests/high/
```

### Rodar com markers específicos
```bash
# Apenas testes high
pytest -q -m high

# Apenas testes medium (sem GPU)
pytest -q -m "medium and not gpu"

# Apenas testes low
pytest -q -m low
```

### Rodar testes específicos
```bash
pytest tests/high/test_determinism_seed.py -v
```

## 🏷️ Markers

| Marker | Descrição | Como rodar |
|--------|-----------|------------|
| `high` | Alta prioridade (obrigatório) | `pytest -m high` |
| `medium` | Prioridade média | `pytest -m medium` |
| `low` | Baixa prioridade | `pytest -m low` |
| `gpu` | Requer GPU | `pytest -m gpu` |
| `slow` | Testes lentos | `pytest -m "not slow"` |

## ⚙️ Configuração

### Variáveis de ambiente

```bash
# Threshold de similaridade cosseno (padrão: 0.98)
export COSINE_THRESHOLD=0.98

# Threshold de I/O em ms (padrão: 50)
export IO_THRESHOLD_MS=50
```

### pytest.ini

Configuração padrão em `pytest.ini`:
- Testes GPU e slow são pulados por padrão
- Saída verbosa com traceback curto

## 🧪 Tipos de Testes

### High Priority

1. **Determinismo** (`test_determinism_seed.py`)
   - Seed fixo produz resultados idênticos
   - 3 runs com mesma seed = hash idêntico

2. **Cache/Imutabilidade** (`test_cache_immutability.py`)
   - Embeddings salvos são imutáveis
   - Pass 2 não recalcula embeddings

3. **Top-K Seleção** (`test_topk_selection.py`)
   - Seleção correta por prominence
   - Ordem decrescente garantida

4. **Propriedades de Máscara** (`test_mask_properties.py`)
   - Máscara gaussiana: max=1.0, min≈0
   - Monotonicidade radial
   - Suavidade (sem degraus)

5. **Temporal Decay** (`test_temporal_decay.py`)
   - IP-Adapter scale = 0 após 60% dos steps
   - Decaimento monotônico

### Medium Priority

1. **Estabilidade sob Compressão** (`test_embedding_stability_compression.py`)
   - Embeddings estáveis após JPG compression
   - Similaridade >= 0.98

2. **Máscaras Sobrepostas** (`test_overlapping_masks.py`)
   - Soma clamped em [0,1]
   - Background mask >= 0

3. **Fallback** (`test_fallback_on_missing_pt.py`)
   - Pass 2 não crasha sem cache
   - Warnings apropriados

4. **Endurance de Memória** (`test_memory_endurance.py`)
   - Memória estável após 100+ tiles
   - Sem vazamentos (requer GPU)

### Low Priority

1. **Concorrência** (`test_concurrency.py`)
   - 4 workers sem duplicatas
   - Thread-safe cache access

2. **Performance I/O** (`test_io_perf.py`)
   - Load embedding.pt < 50ms
   - Load mask.png < 50ms

3. **Scheduler Timestep** (`test_scheduler_timestep.py`)
   - Mapeamento correto de frações
   - Respeito ao end_idx

## 🔧 Fixtures Disponíveis

Ver `conftest.py`:

```python
dummy_page()          # PIL.Image 1024×1024 sintético
dummy_embedding()     # torch.Tensor (768,) normalizado
dummy_detections()    # Lista de 5 detecções fake
dummy_tile_bbox()     # (0, 0, 1024, 1024)
mock_detector         # Mock do YOLODetector
mock_encoder          # Mock do HybridIdentitySystem
temp_dir              # Diretório temporário
```

## 🛠️ Helpers

Ver `core/test_utils.py`:

```python
make_dummy_page(size, seed)
make_dummy_embedding(dim, seed)
make_dummy_bbox(image_size, seed)
create_gaussian_mask(shape, center, sigma)
img_hash(pil_image)
cosine_similarity(a, b)
calculate_prominence(bbox, image_size)
get_ip_adapter_scale_at_step(step, total, end_frac)
```

## 📝 Adicionando Novos Testes

### Estrutura mínima

```python
import pytest

# Escolha o marker apropriado
@pytest.mark.high  # ou medium, low
class TestMinhaFeature:
    
    def test_feature_scenario_expected(self):
        """Descrição clara do teste."""
        # Arrange
        input_data = ...
        
        # Act
        result = minha_funcao(input_data)
        
        # Assert
        assert result == esperado
```

### Usando fixtures

```python
def test_com_dummy_page(dummy_page, dummy_embedding):
    """Usa fixtures do conftest.py."""
    # dummy_page é PIL.Image
    # dummy_embedding é torch.Tensor
    pass
```

### Mocking

```python
def test_com_mock(mocker):
    """Usa pytest-mock."""
    mock = mocker.patch('core.modulo.funcao')
    mock.return_value = 42
    # ...
```

## 🐛 Debug

### Ver logs detalhados
```bash
pytest -v --log-cli-level=DEBUG
```

### Parar no primeiro erro
```bash
pytest -x
```

### Mostrar variáveis locais no traceback
```bash
pytest -l
```

## 📊 CI/GitHub Actions

O workflow `.github/workflows/test.yml` roda:
1. **test_cpu**: Testes high + medium em Python 3.10/3.11
2. **test_low_priority**: Em releases
3. **lint**: flake8 + black
4. **test_gpu**: Opcional (requer self-hosted runner)

## 📈 Métricas Esperadas

| Métrica | Valor Mínimo |
|---------|--------------|
| High tests pass rate | 100% |
| Medium tests pass rate | 100% |
| Cobertura de código | >70% |
| Tempo de teste (CPU) | <2 min |

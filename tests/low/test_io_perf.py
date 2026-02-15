"""
MangaAutoColor Pro - Low Priority Test: I/O Performance

Medir timeit para load(embedding.pt) e load(mask.png).
"""

import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from core.test_utils import create_gaussian_mask, make_dummy_embedding


@pytest.mark.low
class TestIOPerf:
    """Testes de performance de I/O."""
    
    IO_THRESHOLD_MS = 50  # Configurável via variável de ambiente
    
    @classmethod
    def setup_class(cls):
        """Lê threshold de variável de ambiente."""
        env_threshold = os.environ.get('IO_THRESHOLD_MS')
        if env_threshold:
            cls.IO_THRESHOLD_MS = float(env_threshold)
    
    def test_embedding_pt_load_time(self, tmp_path):
        """
        Tempo para carregar embedding.pt deve ser < 50ms.
        
        Aceite: tempo <= threshold.
        """
        # Cria arquivo de teste
        embedding = make_dummy_embedding(dim=768, seed=42)
        pt_path = tmp_path / "test_embedding.pt"
        torch.save(embedding, pt_path)
        
        # Mede tempo de carregamento
        times = []
        for _ in range(10):  # Média de 10 runs
            start = time.perf_counter()
            loaded = torch.load(pt_path)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
        
        avg_time = np.mean(times)
        
        print(f"\n[Embedding Load Test]")
        print(f"File size: {pt_path.stat().st_size} bytes")
        print(f"Average load time: {avg_time:.2f} ms")
        print(f"Threshold: {self.IO_THRESHOLD_MS} ms")
        print(f"Times: {[f'{t:.2f}' for t in times]}")
        
        assert avg_time <= self.IO_THRESHOLD_MS, \
            f"Tempo médio {avg_time:.2f}ms acima do threshold {self.IO_THRESHOLD_MS}ms"
    
    def test_mask_png_load_time(self, tmp_path):
        """
        Tempo para carregar mask.png deve ser < 50ms.
        
        Aceite: tempo <= threshold.
        """
        from PIL import Image
        
        # Cria máscara de teste
        mask = create_gaussian_mask((1024, 1024), center=(512, 512), sigma=200)
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        png_path = tmp_path / "test_mask.png"
        Image.fromarray(mask_uint8).save(png_path)
        
        # Mede tempo de carregamento
        times = []
        for _ in range(10):
            start = time.perf_counter()
            loaded = Image.open(png_path)
            loaded_array = np.array(loaded)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
        
        avg_time = np.mean(times)
        
        print(f"\n[Mask Load Test]")
        print(f"File size: {png_path.stat().st_size} bytes")
        print(f"Average load time: {avg_time:.2f} ms")
        print(f"Threshold: {self.IO_THRESHOLD_MS} ms")
        
        assert avg_time <= self.IO_THRESHOLD_MS, \
            f"Tempo médio {avg_time:.2f}ms acima do threshold {self.IO_THRESHOLD_MS}ms"
    
    def test_embedding_save_time(self, tmp_path):
        """
        Tempo para salvar embedding.pt deve ser rápido.
        
        Aceite: tempo <= threshold.
        """
        embedding = make_dummy_embedding(dim=768, seed=42)
        pt_path = tmp_path / "test_save.pt"
        
        times = []
        for _ in range(10):
            start = time.perf_counter()
            torch.save(embedding, pt_path)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
        
        avg_time = np.mean(times)
        
        print(f"\n[Embedding Save Test]")
        print(f"Average save time: {avg_time:.2f} ms")
        
        # Save pode ser mais lento que load, mas ainda deve ser rápido
        assert avg_time <= self.IO_THRESHOLD_MS * 2, \
            f"Tempo de save {avg_time:.2f}ms muito alto"
    
    def test_batch_embedding_load(self, tmp_path):
        """
        Carregar múltiplos embeddings em batch deve ser eficiente.
        
        Aceite: tempo total proporcional ao número de arquivos.
        """
        # Cria 10 embeddings
        for i in range(10):
            emb = make_dummy_embedding(dim=768, seed=i)
            torch.save(emb, tmp_path / f"emb_{i}.pt")
        
        # Carrega todos
        start = time.perf_counter()
        for i in range(10):
            _ = torch.load(tmp_path / f"emb_{i}.pt")
        total_ms = (time.perf_counter() - start) * 1000
        
        avg_per_file = total_ms / 10
        
        print(f"\n[Batch Load Test]")
        print(f"Total time (10 files): {total_ms:.2f} ms")
        print(f"Average per file: {avg_per_file:.2f} ms")
        
        # Média por arquivo deve ser próxima do tempo individual
        assert avg_per_file <= self.IO_THRESHOLD_MS * 1.5, \
            f"Degradação em batch: {avg_per_file:.2f}ms por arquivo"

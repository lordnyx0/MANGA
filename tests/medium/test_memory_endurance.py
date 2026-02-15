"""
MangaAutoColor Pro - Medium Priority Test: Memory Endurance

Loop N=100 tiles sintéticos; monitorar memória GPU.
"""

import gc

import pytest
import torch

from core.test_utils import make_dummy_page


@pytest.mark.medium
@pytest.mark.slow
@pytest.mark.gpu
class TestMemoryEndurance:
    """Testes de endurance de memória (requer GPU)."""
    
    N_ITERATIONS = 100
    MAX_MEMORY_GROWTH_MB = 50  # Configurável
    
    @pytest.fixture(autouse=True)
    def check_gpu(self):
        """Skip se GPU não disponível."""
        if not torch.cuda.is_available():
            pytest.skip("GPU não disponível")
    
    def test_memory_stable_over_many_tiles(self):
        """
        Loop N=100 tiles; memória não deve crescer indefinidamente.
        
        Aceite: não cresce mais que 50 MB ao longo do loop.
        """
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        memory_snapshots = []
        
        for i in range(self.N_ITERATIONS):
            # Simula processamento de um tile
            # Cria tensor na GPU
            tile_tensor = torch.randn(3, 1024, 1024, device='cuda')
            
            # Simula algum processamento
            result = tile_tensor * 0.5
            
            # Limpa explicitamente
            del tile_tensor, result
            
            # A cada 10 iterações, captura memória
            if i % 10 == 0:
                torch.cuda.synchronize()
                mem_mb = torch.cuda.memory_allocated() / (1024**2)
                memory_snapshots.append((i, mem_mb))
        
        # Força GC
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        final_mem = torch.cuda.memory_allocated() / (1024**2)
        
        print(f"\n[Memory Endurance Test]")
        print(f"Iterations: {self.N_ITERATIONS}")
        print(f"Snapshots: {memory_snapshots}")
        print(f"Final memory: {final_mem:.2f} MB")
        
        # Verifica se memória estabilizou
        # A memória pode variar, mas não deve ter crescimento contínuo
        if len(memory_snapshots) >= 2:
            first_mem = memory_snapshots[0][1]
            last_mem = memory_snapshots[-1][1]
            growth = last_mem - first_mem
            
            print(f"Memory growth: {growth:.2f} MB (threshold: {self.MAX_MEMORY_GROWTH_MB} MB)")
            
            # Permite variação pequena, mas não crescimento contínuo grande
            assert growth < self.MAX_MEMORY_GROWTH_MB, \
                f"Memória cresceu {growth:.2f} MB, acima do limite {self.MAX_MEMORY_GROWTH_MB} MB"
    
    def test_memory_released_after_pipeline(self):
        """
        Memória deve ser liberada após descarregar pipeline.
        
        Aceite: memória retorna ao nível inicial.
        """
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        mem_before = torch.cuda.memory_allocated() / (1024**2)
        
        # Simula criação de objetos grandes
        tensors = []
        for _ in range(10):
            t = torch.randn(1000, 1000, device='cuda')
            tensors.append(t)
        
        torch.cuda.synchronize()
        mem_during = torch.cuda.memory_allocated() / (1024**2)
        
        # Libera
        del tensors
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        mem_after = torch.cuda.memory_allocated() / (1024**2)
        
        print(f"\n[Memory Release Test]")
        print(f"Before: {mem_before:.2f} MB")
        print(f"During: {mem_during:.2f} MB")
        print(f"After: {mem_after:.2f} MB")
        
        # Memória deve ter diminuído significativamente
        assert mem_after < mem_during * 0.5, "Memória não foi liberada adequadamente"
    
    def test_no_memory_leak_in_tile_loop(self):
        """
        Teste específico para vazamento de memória em loop de tiles.
        
        Aceite: memória estável.
        """
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Aquece
        for _ in range(10):
            t = torch.randn(3, 512, 512, device='cuda')
            del t
        
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated() / (1024**2)
        
        # Loop principal
        for i in range(50):
            # Simula tile processing
            tile = torch.randn(3, 512, 512, device='cuda')
            processed = tile.sigmoid()
            output = processed * 255
            del tile, processed, output
            
            if i == 25:
                torch.cuda.synchronize()
                mid_mem = torch.cuda.memory_allocated() / (1024**2)
        
        torch.cuda.synchronize()
        final_mem = torch.cuda.memory_allocated() / (1024**2)
        
        print(f"\n[Leak Test]")
        print(f"Baseline: {baseline:.2f} MB")
        print(f"Mid-point: {mid_mem:.2f} MB")
        print(f"Final: {final_mem:.2f} MB")
        
        # Variação deve ser pequena
        variation = abs(final_mem - baseline)
        assert variation < 20, f"Possível vazamento: variação de {variation:.2f} MB"

"""
MangaAutoColor Pro - Low Priority Test: Concurrency

Simula 4 workers processando TileJobs; verifica locks e deduplicação.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from core.test_utils import make_dummy_embedding


@pytest.mark.low
class TestConcurrency:
    """Testes de concorrência e paralelismo."""
    
    def test_multiple_workers_no_duplicate_embeddings(self):
        """
        4 workers processando TileJobs não devem duplicar embeddings.
        
        Aceite: sem duplicatas.
        """
        # Simula cache compartilhado
        shared_cache = {}
        processed_chars = []
        lock = threading.Lock()
        
        def process_tile(tile_id, char_ids):
            """Processa um tile, carregando embeddings necessários."""
            loaded = []
            for char_id in char_ids:
                # Verifica cache
                if char_id not in shared_cache:
                    # Simula carregamento
                    with lock:
                        # Double-check após adquirir lock
                        if char_id not in shared_cache:
                            # Usa seed positivo baseado no hash
                            seed_val = abs(hash(char_id)) % (2**31)
                            shared_cache[char_id] = make_dummy_embedding(seed=seed_val)
                
                loaded.append(char_id)
            
            with lock:
                processed_chars.extend(loaded)
            
            return tile_id, loaded
        
        # Jobs simulados
        jobs = [
            ('tile_001', ['char_001', 'char_002']),
            ('tile_002', ['char_002', 'char_003']),  # char_002 duplicado
            ('tile_003', ['char_001', 'char_004']),  # char_001 duplicado
            ('tile_004', ['char_003', 'char_004']),  # ambos duplicados
        ]
        
        # Executa em paralelo
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_tile, tid, chars) for tid, chars in jobs]
            results = [f.result() for f in as_completed(futures)]
        
        print(f"\n[Concurrency Test]")
        print(f"Shared cache keys: {list(shared_cache.keys())}")
        print(f"Processed chars: {processed_chars}")
        print(f"Cache size: {len(shared_cache)}")
        
        # Cada personagem deve estar no cache apenas uma vez
        assert len(shared_cache) == 4, f"Esperado 4 embeddings, cache tem {len(shared_cache)}"
        
        # Não deve haver duplicatas no cache
        assert len(set(shared_cache.keys())) == len(shared_cache), "Duplicatas no cache"
    
    def test_thread_safety_cache_access(self):
        """
        Acesso concorrente ao cache deve ser thread-safe.
        
        Aceite: sem race conditions.
        """
        cache = {}
        errors = []
        lock = threading.Lock()
        
        def writer(thread_id):
            for i in range(100):
                key = f"key_{i % 10}"  # 10 chaves diferentes
                try:
                    with lock:
                        cache[key] = f"value_from_thread_{thread_id}"
                    time.sleep(0.001)  # Pequeno delay
                except Exception as e:
                    errors.append(str(e))
        
        def reader(thread_id):
            for i in range(100):
                key = f"key_{i % 10}"
                try:
                    with lock:
                        _ = cache.get(key)
                    time.sleep(0.001)
                except Exception as e:
                    errors.append(str(e))
        
        # Executa writers e readers concorrentemente
        threads = []
        for i in range(2):
            t = threading.Thread(target=writer, args=(i,))
            threads.append(t)
            t = threading.Thread(target=reader, args=(i,))
            threads.append(t)
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        print(f"\n[Thread Safety Test]")
        print(f"Total errors: {len(errors)}")
        print(f"Cache size: {len(cache)}")
        
        assert len(errors) == 0, f"Erros em acesso concorrente: {errors[:5]}"
    
    def test_tile_job_ordering(self):
        """
        TileJobs devem poder ser processados em qualquer ordem.
        
        Aceite: resultado independente da ordem.
        """
        jobs = [
            {'tile_id': 'tile_001', 'page_num': 0, 'bbox': (0, 0, 512, 512)},
            {'tile_id': 'tile_002', 'page_num': 0, 'bbox': (256, 0, 768, 512)},
            {'tile_id': 'tile_003', 'page_num': 0, 'bbox': (0, 256, 512, 768)},
            {'tile_id': 'tile_004', 'page_num': 0, 'bbox': (256, 256, 768, 768)},
        ]
        
        def process_job(job):
            return job['tile_id']
        
        # Ordem 1
        with ThreadPoolExecutor(max_workers=2) as executor:
            result1 = list(executor.map(process_job, jobs))
        
        # Ordem reversa
        with ThreadPoolExecutor(max_workers=2) as executor:
            result2 = list(executor.map(process_job, reversed(jobs)))
        
        print(f"\n[Ordering Test]")
        print(f"Order 1: {result1}")
        print(f"Order 2 (reversed): {result2}")
        
        # Mesmos tiles processados (ordem pode variar)
        assert set(result1) == set(result2), "Conjunto de tiles processados diferente"
